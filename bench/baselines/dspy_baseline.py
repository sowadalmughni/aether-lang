#!/usr/bin/env python3
"""
DSPy Baseline Benchmark (real `dspy` package, pinned to 2.6.27).

Mirrors bench/baselines/langchain_baseline.py section-by-section so a
reviewer can diff the two files side-by-side. The output JSON schema is
identical to bench/results/langchain_v1.json so all three systems
(LangChain, DSPy, Aether) can be compared field-by-field.

Acceptance criteria (per session task spec):
  1. `import dspy` works at the top of this script. (See imports below.)
  2. Real `dspy.Signature` and `dspy.Module` are used (no shimmed equivalents).
  3. Output JSON matches the langchain_v1.json / aether_mock_v1.json schema.

Mock latency is 50 ms flat -- parity with
  - aether-runtime/src/llm.rs:273 (latency_ms: 50)
  - bench/baselines/langchain_baseline.py:72 (LATENCY_MS_MOCK = 50)
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Real DSPy imports (proof per acceptance criterion 1).
import dspy
from dspy import (
    BaseLM,
    ChainOfThought,
    ChatAdapter,
    InputField,
    Module,
    OutputField,
    Predict,
    Prediction,
    Signature,
)

# Reuse Aether's aggregation primitives (same pattern as langchain_baseline.py:55-64).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_benchmark import (  # noqa: E402
    _bootstrap_ci,  # noqa: F401  (re-exported for parity with langchain baseline)
    _trial_percentiles,
    aggregate_trials,
    get_git_version,
    get_hardware_info,
)


# =============================================================================
# Mock LM -- 50 ms flat latency, no jitter, no prompt scaling.
# Mirrors aether-runtime/src/llm.rs:273 (latency_ms: 50) and
# bench/baselines/langchain_baseline.py:72 (LATENCY_MS_MOCK = 50) exactly.
# =============================================================================

LATENCY_MS_MOCK = 50

# Module-level call recorder, drained per-item by the trial runner so we can
# compute parallelization_factor empirically (sum_of_per_call / item_wall).
# Mirrors langchain_baseline.py:79-92.
_CALL_DURATIONS_MS: list[float] = []
_CALL_DURATIONS_LOCK = threading.Lock()


def _record_call_ms(elapsed_ms: float) -> None:
    with _CALL_DURATIONS_LOCK:
        _CALL_DURATIONS_MS.append(elapsed_ms)


def _drain_call_durations_ms() -> list[float]:
    with _CALL_DURATIONS_LOCK:
        out = list(_CALL_DURATIONS_MS)
        _CALL_DURATIONS_MS.clear()
        return out


# Parses the system message produced by dspy.ChatAdapter for the lines:
#   Your output fields are:
#   1. `urgency` (str):
#   2. `category` (str):
#   ...
# and recovers the output field names. ChainOfThought injects an extra
# `reasoning` field which appears in the same numbered list, so it is
# discovered by the same regex with no special-casing.
_OUTPUT_FIELDS_BLOCK_RE = re.compile(
    r"Your output fields are:\n((?:\s*\d+\.\s*`\w+`.*\n?)+)"
)
_FIELD_NAME_RE = re.compile(r"`(\w+)`")


class DeterministicMockLM(BaseLM):
    """Flat 50ms mock for DSPy 2.6.27.

    Subclasses dspy.BaseLM (the minimal seam): one abstract method
    `forward(prompt, messages, **kwargs)` returning an OpenAI-shaped response.
    See https://github.com/stanfordnlp/dspy/blob/2.6.27/dspy/clients/base_lm.py.

    Latency: 50 ms flat (no jitter).
    Determinism: response derived from sha256 of the formatted prompt.
    Output format: dspy.ChatAdapter markers --
      [[ ## field_name ## ]]\\n<value>
      ...
      [[ ## completed ## ]]
    """

    def __init__(self, model: str = "deterministic-mock") -> None:
        super().__init__(model=model)
        self.model_type = "chat"
        self.kwargs = {"temperature": 0.0, "max_tokens": 4000}
        self.history = []

    @staticmethod
    def _output_fields(system_msg: str) -> list[str]:
        m = _OUTPUT_FIELDS_BLOCK_RE.search(system_msg)
        if not m:
            return []
        return _FIELD_NAME_RE.findall(m.group(1))

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        if messages:
            sys_msg = next(
                (m.get("content", "") for m in messages if m.get("role") == "system"),
                "",
            )
            text = "\n".join((m.get("content") or "") for m in messages)
        else:
            text = prompt or ""
            sys_msg = text

        out_fields = self._output_fields(sys_msg)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        body_parts = []
        for i, name in enumerate(out_fields):
            value = f"mock_{name}_{digest[i * 4:(i + 1) * 4]}"
            body_parts.append(f"[[ ## {name} ## ]]\n{value}")
        body_parts.append("[[ ## completed ## ]]")
        content = "\n\n".join(body_parts)

        t0 = time.perf_counter()
        time.sleep(LATENCY_MS_MOCK / 1000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_call_ms(elapsed_ms)

        # Token model -- matches langchain_baseline.py:130-131 and
        # aether-runtime/src/llm.rs:377-378.
        prompt_tokens = max(1, len(text) // 4)
        completion_tokens = 50 + prompt_tokens // 4
        total_tokens = prompt_tokens + completion_tokens
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        # OpenAI-shaped response object that BaseLM._process_lm_response can
        # read: it accesses .choices[i].message.content, .usage (as dict()),
        # .model, ._hidden_params["response_cost"].
        msg = types.SimpleNamespace(content=content, tool_calls=None)
        choice = types.SimpleNamespace(message=msg, logprobs=None)
        resp = types.SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=self.model,
            _hidden_params={"response_cost": 0.0},
        )
        return resp


# =============================================================================
# Signatures
# =============================================================================
# Per the task spec: TriageSignature is a single conceptual contract with
# input `query` and outputs `urgency`, `category`, `context`. The Module
# below decomposes this into three sub-Predict/CoT calls (one per step) so
# the call count and parallelization shape match the LangChain baseline's
# three-step triage chain.

class TriageSignature(Signature):
    """Triage a customer support query: classify urgency, classify category, generate response."""

    query: str = InputField()
    urgency: str = OutputField(desc="Low | Medium | High | Critical")
    category: str = OutputField(
        desc="authentication, billing, bug, feature-request, how-to, outage, security, etc."
    )
    context: str = OutputField(desc="A concise, helpful customer support response")


class _UrgencySig(Signature):
    """Classify the urgency of this customer query as Low, Medium, High, or Critical."""

    query: str = InputField()
    customer_tier: str = InputField()
    urgency: str = OutputField(desc="Low | Medium | High | Critical")


class _CategorySig(Signature):
    """Classify this customer query into a category (authentication, billing, bug, feature-request, how-to, outage, security, etc.)."""

    query: str = InputField()
    category: str = OutputField()


class _ResponseSig(Signature):
    """Generate a helpful customer support response given the query, urgency, and category."""

    query: str = InputField()
    urgency: str = InputField()
    category: str = InputField()
    context: str = OutputField(desc="Concise, helpful customer-support response")


class ExtractionSignature(Signature):
    """Analyse a document: extract entities, summarise, classify domain."""

    document: str = InputField()
    entities: str = OutputField(desc="JSON array of named entities")
    summary: str = OutputField(desc="2-3 sentence summary")
    domain: str = OutputField(
        desc="technology, finance, medical, legal, machine-learning, etc."
    )


class _EntitiesSig(Signature):
    """Extract all named entities from this document. Return as a JSON array of strings."""

    document: str = InputField()
    entities: str = OutputField(desc="JSON array of strings")


class _SummarySig(Signature):
    """Summarize this document in 2-3 sentences."""

    document: str = InputField()
    summary: str = OutputField()


class _DomainSig(Signature):
    """Classify the domain of this document (e.g., technology, finance, medical, legal, etc.)."""

    document: str = InputField()
    domain: str = OutputField()


# =============================================================================
# Modules
# =============================================================================

class TriageModule(Module):
    """Three-step triage: urgency, category (independent), then response (depends on both)."""

    def __init__(self, parallel: bool) -> None:
        super().__init__()
        self._parallel = parallel
        # dspy.Predict for single-label classification, dspy.ChainOfThought
        # for the generative response step (per design decision recorded in
        # plan: "Mixed -- Predict for classification, CoT for generation").
        self.urgency_pred = Predict(_UrgencySig)
        self.category_pred = Predict(_CategorySig)
        self.response_cot = ChainOfThought(_ResponseSig)

    def forward(self, query: str, customer_tier: str = "free") -> Prediction:  # type: ignore[override]
        if self._parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fu = pool.submit(self.urgency_pred, query=query, customer_tier=customer_tier)
                fc = pool.submit(self.category_pred, query=query)
                u = fu.result().urgency
                c = fc.result().category
        else:
            u = self.urgency_pred(query=query, customer_tier=customer_tier).urgency
            c = self.category_pred(query=query).category
        ctx = self.response_cot(query=query, urgency=u, category=c).context
        return Prediction(urgency=u, category=c, context=ctx)


class ExtractionModule(Module):
    """Three independent analyzers: entities, summary, domain."""

    def __init__(self, parallel: bool) -> None:
        super().__init__()
        self._parallel = parallel
        self.entities_cot = ChainOfThought(_EntitiesSig)
        self.summary_cot = ChainOfThought(_SummarySig)
        self.domain_pred = Predict(_DomainSig)

    def forward(self, document: str) -> Prediction:  # type: ignore[override]
        if self._parallel:
            with ThreadPoolExecutor(max_workers=3) as pool:
                fe = pool.submit(self.entities_cot, document=document)
                fs = pool.submit(self.summary_cot, document=document)
                fd = pool.submit(self.domain_pred, document=document)
                e = fe.result().entities
                s = fs.result().summary
                d = fd.result().domain
        else:
            e = self.entities_cot(document=document).entities
            s = self.summary_cot(document=document).summary
            d = self.domain_pred(document=document).domain
        return Prediction(entities=e, summary=s, domain=d)


# =============================================================================
# Explicit prompt-keyed cache (for parallel_cached config).
# Interface mirrors langchain_baseline.py:190-237 ExplicitCache.
# =============================================================================

class ExplicitCache:
    """Opt-in prompt-keyed cache. Stores serialized Prediction outputs under a
    JSON-serialised input dict; tracks hit/miss counts so per-trial
    cache_hit_rate can be reported.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[dict[str, Any]]:
        v = self._store.get(key)
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def _cache_key(step: str, payload: dict) -> str:
    return step + "::" + json.dumps(payload, sort_keys=True, default=str)


class _CachedPredictor:
    """Wraps a dspy.Predict / dspy.ChainOfThought instance with a prompt-keyed
    cache. On hit, returns a Prediction reconstructed from the cached field
    dict (no LM call). On miss, calls the inner predictor and stores all
    fields from the resulting Prediction.
    """

    def __init__(self, step_name: str, inner, cache: ExplicitCache) -> None:
        self.step_name = step_name
        self._inner = inner
        self._cache = cache

    def __call__(self, **kwargs) -> Prediction:
        key = _cache_key(self.step_name, kwargs)
        hit = self._cache.get(key)
        if hit is not None:
            return Prediction(**hit)
        result = self._inner(**kwargs)
        # dspy.Prediction supports .keys() / dict-like iteration in 2.6.x.
        fields = {k: result[k] for k in result.keys()}
        self._cache.put(key, fields)
        return result


def _wrap_module_with_cache(module: Module, cache: ExplicitCache) -> Module:
    """Replace each step-predictor on a TriageModule / ExtractionModule with
    a _CachedPredictor wrapper. Mutates and returns the same module.
    """
    if isinstance(module, TriageModule):
        module.urgency_pred = _CachedPredictor("triage.urgency", module.urgency_pred, cache)
        module.category_pred = _CachedPredictor("triage.category", module.category_pred, cache)
        module.response_cot = _CachedPredictor("triage.response", module.response_cot, cache)
    elif isinstance(module, ExtractionModule):
        module.entities_cot = _CachedPredictor("extract.entities", module.entities_cot, cache)
        module.summary_cot = _CachedPredictor("extract.summary", module.summary_cot, cache)
        module.domain_pred = _CachedPredictor("extract.domain", module.domain_pred, cache)
    else:
        raise TypeError(f"unsupported module type for caching: {type(module).__name__}")
    return module


# =============================================================================
# Trial runner -- per-trial measurements dict matches langchain baseline:
#   {latencies_ms, tokens_total, tokens_saved, cache_hits, cache_misses,
#    cache_hit_rate, errors, successful, total_time_ms,
#    parallelization_factor_mean, level_execution_times_ms_mean}
# =============================================================================

def _build_module(scenario: str, parallel: bool, cache: Optional[ExplicitCache]) -> Module:
    if scenario == "triage":
        m: Module = TriageModule(parallel=parallel)
    elif scenario == "extraction":
        m = ExtractionModule(parallel=parallel)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    if cache is not None:
        m = _wrap_module_with_cache(m, cache)
    return m


def _item_kwargs(scenario: str, item: dict) -> dict:
    if scenario == "triage":
        ctx = item.get("context", {}) or {}
        return {"query": item["query"], "customer_tier": ctx.get("customer_tier", "free")}
    if scenario == "extraction":
        return {"document": item["document"]}
    raise ValueError(f"unknown scenario: {scenario}")


def run_one_trial(
    scenario: str,
    items: list[dict],
    parallel: bool,
    cache: Optional[ExplicitCache],
) -> dict:
    """Mirror of langchain_baseline.py:359-436."""
    cache_hits_start = cache.hits if cache else 0
    cache_misses_start = cache.misses if cache else 0

    module = _build_module(scenario, parallel=parallel, cache=cache)

    latencies_ms: list[float] = []
    parallelization_factors: list[float] = []
    successful = 0
    failed = 0

    start = time.perf_counter()
    for item in items:
        _drain_call_durations_ms()  # reset per-item recorder
        item_start = time.perf_counter()
        try:
            module(**_item_kwargs(scenario, item))
            successful += 1
        except Exception as exc:
            print(f"item failed: {exc}", file=sys.stderr)
            failed += 1
        item_wall_ms = (time.perf_counter() - item_start) * 1000.0
        per_call = _drain_call_durations_ms()
        latencies_ms.append(item_wall_ms)
        if item_wall_ms > 0:
            pf = sum(per_call) / item_wall_ms if per_call else 0.0
            parallelization_factors.append(pf)

    total_time_ms = (time.perf_counter() - start) * 1000.0

    if cache is not None:
        cache_hits = cache.hits - cache_hits_start
        cache_misses = cache.misses - cache_misses_start
    else:
        cache_hits = 0
        cache_misses = 0
    cache_hit_rate = (
        cache_hits / (cache_hits + cache_misses)
        if (cache_hits + cache_misses) > 0
        else 0.0
    )
    pf_mean = (
        sum(parallelization_factors) / len(parallelization_factors)
        if parallelization_factors
        else (1.0 if not parallel else 0.0)
    )

    return {
        "latencies_ms": latencies_ms,
        "tokens_total": 0,
        "tokens_saved": 0,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hit_rate,
        "errors": failed,
        "successful": successful,
        "total_time_ms": total_time_ms,
        "parallelization_factor_mean": pf_mean,
        "level_execution_times_ms_mean": [],  # not measured for this baseline
    }


# =============================================================================
# Suite driver: dataset x config x N trials, with parallel_cached warmup
# protocol mirroring langchain_baseline.py:461-501.
# =============================================================================

SUITE_DATASETS = [
    ("customer_support_100", "triage",     "bench/datasets/customer_support_100.jsonl"),
    ("document_analysis_50", "extraction", "bench/datasets/document_analysis_50.jsonl"),
]
SUITE_CONFIGS = ["sequential", "parallel", "parallel_cached"]


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_config_trials(
    scenario: str,
    items: list[dict],
    config: str,
    n_trials: int,
) -> list[dict]:
    assert config in SUITE_CONFIGS, f"unknown config: {config}"
    parallel = (config != "sequential")
    trials: list[dict] = []

    if config == "parallel_cached":
        cache = ExplicitCache()
        # Warmup pass (discarded). Cache freshly empty -> ~0% hit rate.
        warmup = run_one_trial(scenario, items, parallel=True, cache=cache)
        if warmup["cache_hit_rate"] > 0.05:
            raise RuntimeError(
                f"parallel_cached warmup expected ~0% hit rate but saw "
                f"{warmup['cache_hit_rate']:.3f} -- ExplicitCache state leaking"
            )
        for i in range(n_trials):
            print(f"    trial {i+1}/{n_trials} ({config})...", flush=True)
            trial = run_one_trial(scenario, items, parallel=True, cache=cache)
            trial["trial"] = i
            p50, p95, p99 = _trial_percentiles(trial["latencies_ms"])
            trial["p50"], trial["p95"], trial["p99"] = p50, p95, p99
            trials.append(trial)
        if trials and trials[0]["cache_hit_rate"] < 0.5:
            raise RuntimeError(
                f"parallel_cached first measured trial hit_rate "
                f"{trials[0]['cache_hit_rate']:.3f}; expected > 0.5 after warmup"
            )
    else:
        for i in range(n_trials):
            print(f"    trial {i+1}/{n_trials} ({config})...", flush=True)
            trial = run_one_trial(scenario, items, parallel=parallel, cache=None)
            trial["trial"] = i
            p50, p95, p99 = _trial_percentiles(trial["latencies_ms"])
            trial["p50"], trial["p95"], trial["p99"] = p50, p95, p99
            trials.append(trial)
    return trials


def _trial_for_json(t: dict) -> dict:
    """Slim per-trial dict for raw_trial_results (mirrors langchain_baseline.py:504-520)."""
    return {
        "trial": t["trial"],
        "latencies_ms": [round(x, 4) for x in t["latencies_ms"]],
        "p50": round(t["p50"], 4),
        "p95": round(t["p95"], 4),
        "p99": round(t["p99"], 4),
        "cache_hit_rate": round(t["cache_hit_rate"], 6),
        "cache_hits": t["cache_hits"],
        "cache_misses": t["cache_misses"],
        "tokens_total": t["tokens_total"],
        "tokens_saved": t["tokens_saved"],
        "errors": t["errors"],
        "parallelization_factor_mean": round(t["parallelization_factor_mean"], 4),
        "level_execution_times_ms_mean": [round(x, 4) for x in t["level_execution_times_ms_mean"]],
    }


# =============================================================================
# Cost guard -- DSPy 2.6 uses litellm under dspy.LM. We do not invoke it in
# mock mode, so the cost guard only matters for --mode openai.
# =============================================================================

COST_PER_INPUT_TOKEN_USD = 0.15e-6   # gpt-4o-mini public pricing
COST_PER_OUTPUT_TOKEN_USD = 0.60e-6
COST_GUARD_USD = 5.0


def estimate_cost_usd(
    scenario_items_pairs: list[tuple[str, list[dict]]],
    n_trials: int,
    n_configs: int,
    expected_output_tokens_per_call: int = 30,
) -> dict:
    """Crude cost estimate. We don't render exact DSPy prompts here; we use
    the same input-text-length heuristic as langchain_baseline.py.
    """
    total_input_tokens = 0
    total_calls = 0
    for scenario, items in scenario_items_pairs:
        if scenario == "triage":
            steps = 3
            for item in items:
                ctx = item.get("context", {}) or {}
                base = (item["query"] or "") + (ctx.get("customer_tier") or "")
                total_input_tokens += (len(base) // 4) * steps
                total_calls += steps
        elif scenario == "extraction":
            steps = 3
            for item in items:
                base = item.get("document", "") or ""
                total_input_tokens += (len(base) // 4) * steps
                total_calls += steps
    total_input_tokens *= n_trials * n_configs
    total_calls *= n_trials * n_configs
    total_output_tokens = total_calls * expected_output_tokens_per_call
    cost = (
        total_input_tokens * COST_PER_INPUT_TOKEN_USD
        + total_output_tokens * COST_PER_OUTPUT_TOKEN_USD
    )
    return {
        "calls": total_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost_usd": cost,
    }


# =============================================================================
# CLI
# =============================================================================

def _self_check() -> int:
    """Verify mock latency parity: a single Predict call should be ~50 ms."""
    print("Imports verified:")
    print("  import dspy")
    print("  from dspy import BaseLM, ChainOfThought, ChatAdapter, InputField, Module, OutputField, Predict, Prediction, Signature")
    print(f"\ndspy.__version__ = {dspy.__version__}")
    print(f"importlib.metadata.version('dspy-ai') = {importlib.metadata.version('dspy-ai')}")
    print(f"importlib.metadata.version('dspy')    = {importlib.metadata.version('dspy')}")

    dspy.configure(lm=DeterministicMockLM(), adapter=ChatAdapter())
    p = Predict(_UrgencySig)
    # Warmup once (signature compilation, etc.)
    p(query="warmup", customer_tier="free")
    samples = []
    for _ in range(5):
        _drain_call_durations_ms()
        t0 = time.perf_counter()
        p(query="ping", customer_tier="free")
        samples.append((time.perf_counter() - t0) * 1000.0)
    median = sorted(samples)[len(samples) // 2]
    print(f"\nMock single-Predict median wall time: {median:.2f} ms (samples: {[round(s,2) for s in samples]})")
    if 45.0 <= median <= 75.0:
        print("50ms mock parity OK (target: aether-runtime/src/llm.rs:273 latency_ms=50)")
        return 0
    print(f"PARITY FAILURE: median {median:.2f} outside [45, 75] ms window", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="DSPy baseline benchmark")
    parser.add_argument(
        "--scenario",
        choices=["triage", "extraction", "all"],
        default="all",
        help="Which workflow to benchmark",
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials per (dataset, config)")
    parser.add_argument(
        "--mode",
        choices=["mock", "openai"],
        default=os.environ.get("BASELINE_PROVIDER", "mock"),
        help="Provider mode (default: $BASELINE_PROVIDER or 'mock')",
    )
    parser.add_argument(
        "--config",
        choices=SUITE_CONFIGS + ["all"],
        default="all",
        help="Restrict to one config or run all three",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "bench" / "results" / "dspy_v1.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Required to actually call the real API in --mode openai (after cost guard)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Print imports and verify mock latency parity, then exit",
    )
    args = parser.parse_args()

    if args.self_check:
        return _self_check()

    selected_datasets = (
        SUITE_DATASETS
        if args.scenario == "all"
        else [d for d in SUITE_DATASETS if d[1] == args.scenario]
    )
    if not selected_datasets:
        print(f"No datasets matched scenario '{args.scenario}'", file=sys.stderr)
        return 1
    selected_configs = SUITE_CONFIGS if args.config == "all" else [args.config]

    scenario_items: list[tuple[str, str, list[dict]]] = []
    for dataset_name, scenario, rel_path in selected_datasets:
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"Dataset not found: {path}", file=sys.stderr)
            return 1
        items = load_jsonl(path)
        scenario_items.append((dataset_name, scenario, items))
        print(f"Loaded {len(items)} items from {dataset_name}", flush=True)

    if args.mode == "mock":
        dspy.configure(lm=DeterministicMockLM(), adapter=ChatAdapter())
        provider_label = "mock"
        version_label = importlib.metadata.version("dspy-ai")
    elif args.mode == "openai":
        est = estimate_cost_usd(
            [(s[1], s[2]) for s in scenario_items],
            n_trials=args.trials,
            n_configs=len(selected_configs),
        )
        print(
            f"\n[cost guard] estimated: {est['calls']} calls, "
            f"~{est['input_tokens']} input toks, ~{est['output_tokens']} output toks, "
            f"~${est['cost_usd']:.4f} USD",
            flush=True,
        )
        if est["cost_usd"] > COST_GUARD_USD:
            print(
                f"[cost guard] aborting: estimated ${est['cost_usd']:.2f} > ${COST_GUARD_USD:.2f}",
                file=sys.stderr,
            )
            return 2
        if not args.confirm_cost:
            print(
                "[cost guard] add --confirm-cost to actually spend money. Aborting.",
                file=sys.stderr,
            )
            return 3
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set", file=sys.stderr)
            return 5
        dspy.configure(
            lm=dspy.LM("openai/gpt-4o-mini", temperature=0, cache=False),
            adapter=ChatAdapter(),
        )
        provider_label = "openai"
        version_label = importlib.metadata.version("dspy-ai")
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        return 1

    results: list[dict] = []
    for dataset_name, scenario, items in scenario_items:
        for config in selected_configs:
            print(f"  {dataset_name} / {config} ({args.trials} trials)...", flush=True)
            trials = run_config_trials(scenario, items, config, args.trials)
            agg = aggregate_trials(trials)
            entry = {
                "dataset": dataset_name,
                "config": config,
                "trials": args.trials,
                **agg,
                "raw_trial_results": [_trial_for_json(t) for t in trials],
            }
            results.append(entry)

    out = {
        "system": "dspy",
        "version": version_label,
        "provider": provider_label,
        "hardware": get_hardware_info(),
        "datasets": [d[0] for d in selected_datasets],
        "configs": selected_configs,
        "trials_per_config": args.trials,
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "results": results,
        "git_version": get_git_version(REPO_ROOT),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
