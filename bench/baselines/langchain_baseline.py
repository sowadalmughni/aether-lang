#!/usr/bin/env python3
"""LangChain baseline benchmark (real LangChain, not a stub).

Mirrors Aether's two benchmark scenarios using idiomatic LangChain Expression
Language (LCEL):

  * triage      — three-step customer-support workflow
                  (extract context not needed; urgency || category -> response).
                  Sequential variant chains the three steps; parallel variant
                  uses RunnableParallel for urgency+category, then feeds the
                  result into the response chain.
  * extraction  — three independent analyzers (entities, summary, domain) plus
                  a non-LLM combine step. Parallel variant uses RunnableParallel.

Mock-mode LLM (default) is a BaseChatModel subclass with a flat 50 ms latency,
matching Aether's runtime mock provider exactly (aether-runtime/src/llm.rs:273
-> latency_ms: 50; line 375 sleeps for that constant). This is the fair-
comparison baseline -- changing it invalidates side-by-side claims.

Real-API mode (BASELINE_PROVIDER=openai or --mode openai) uses ChatOpenAI with
gpt-4o-mini, gated by a cost guard that aborts if the estimated USD cost
exceeds $5 (cheap default; user must pass --confirm-cost to actually spend).

Output: bench/results/langchain_v1.json in the same suite schema as
bench/results/aether_mock_v1.json (system / version / provider / hardware /
datasets / configs / trials_per_config / measured_at / results[]).
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# --- Real LangChain imports (proof per acceptance criterion 3) ---------------
from langchain_core.caches import InMemoryCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.utils import AddableDict

# --- Reuse Aether's aggregation primitives ----------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_benchmark import (  # noqa: E402
    _bootstrap_ci,
    _trial_percentiles,
    aggregate_trials,
    get_git_version,
    get_hardware_info,
)


# =============================================================================
# Mock LLM -- 50 ms flat latency, no jitter, no prompt scaling.
# Mirrors aether-runtime/src/llm.rs:273 (latency_ms: 50) exactly.
# =============================================================================

LATENCY_MS_MOCK = 50

# Module-level call recorder. Cleared per-item by run_one_trial so we can
# compute parallelization_factor empirically (sum_of_per_call_durations /
# wall_clock). BaseChatModel is a pydantic model, so attaching mutable state
# to instances is awkward -- a module-level deque + lock is simpler and
# thread-safe (RunnableParallel uses ThreadPoolExecutor).
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


# Per-trial token usage recorder. Captures real prompt/completion token counts
# from ChatOpenAI response metadata so we can compute the actual USD cost from
# response usage (gpt-4o-mini: $0.15/1M input, $0.60/1M output).
#
# Attach this directly to the ChatOpenAI instance (NOT via chain.invoke config),
# because cached_step calls inner.invoke(payload) without propagating the outer
# chain's callbacks — config-attached trackers would miss warmup API calls in
# the parallel_cached config. LLM-attached callbacks fire for every model call
# regardless of how the chain wraps it. We snapshot/diff per trial to recover
# per-trial counts.
class UsageTracker(BaseCallbackHandler):
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs) -> None:  # type: ignore[override]
        # ChatOpenAI surfaces usage in two places depending on version:
        #   1. response.llm_output["token_usage"]
        #   2. each generation.message.usage_metadata (langchain >= 0.3)
        # Try both to stay robust across pinned langchain versions.
        usage = None
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
        if usage:
            with self._lock:
                self.input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                self.output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            return
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg else None
                if meta:
                    with self._lock:
                        self.input_tokens += int(meta.get("input_tokens", 0) or 0)
                        self.output_tokens += int(meta.get("output_tokens", 0) or 0)

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.input_tokens, self.output_tokens


class DeterministicMockChat(BaseChatModel):
    """Flat-latency mock that mirrors Aether's runtime mock provider.

    Reference: aether-runtime/src/llm.rs:270-345 (MockLlmClient).
    Latency: 50 ms flat (no jitter, no prompt-length scaling).
    Token model: input_tokens = len(prompt)//4; output_tokens = 50 + input//4
    (matches aether-runtime/src/llm.rs:377-378).
    Response: deterministic SHA256-prefixed string (matches the runtime's
    "[Mock Response]\\nModel: ...\\nPrompt hash: <16 hex>\\n..." format).
    """

    @property
    def _llm_type(self) -> str:
        return "deterministic-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(getattr(m, "content", "") or "" for m in messages)
        t0 = time.perf_counter()
        time.sleep(LATENCY_MS_MOCK / 1000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_call_ms(elapsed_ms)

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        content = (
            "[Mock Response]\n"
            f"Model: deterministic-mock\n"
            f"Prompt hash: {prompt_hash}\n"
            "This is a deterministic mock response for testing."
        )
        input_tokens = len(prompt) // 4
        output_tokens = 50 + (input_tokens // 4)
        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(
            generations=[gen],
            llm_output={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )


# =============================================================================
# Prompt templates -- VERBATIM from scripts/run_benchmark.py:125-188.
# Keep these byte-identical so triage/extraction prompts hash the same way
# the Aether runtime sees them (matters for cache parity comparisons).
# =============================================================================

TRIAGE_URGENCY_TMPL = (
    "Classify the urgency of this customer query as Low, Medium, High, or Critical.\n\n"
    "Query: {query}\n\n"
    "Customer tier: {customer_tier}\n\n"
    "Respond with only the urgency level."
)
TRIAGE_CATEGORY_TMPL = (
    "Classify this customer query into a category (authentication, billing, bug, "
    "feature-request, how-to, outage, security, etc.).\n\n"
    "Query: {query}\n\n"
    "Respond with only the category."
)
TRIAGE_RESPONSE_TMPL = (
    "Generate a helpful customer support response for this query.\n\n"
    "Query: {query}\n"
    "Urgency: {urgency}\n"
    "Category: {category}\n\n"
    "Provide a concise, helpful response."
)

EXTRACT_ENTITIES_TMPL = (
    "Extract all named entities from this document. "
    "Return as a JSON array of strings.\n\nDocument: {document}"
)
EXTRACT_SUMMARY_TMPL = (
    "Summarize this document in 2-3 sentences.\n\nDocument: {document}"
)
EXTRACT_DOMAIN_TMPL = (
    "Classify the domain of this document "
    "(e.g., technology, finance, medical, legal, etc.).\n\n"
    "Document: {document}\n\nRespond with only the domain."
)


# =============================================================================
# Explicit, opt-in cache. Wraps langchain_core.caches.InMemoryCache (real
# LangChain symbol) and is invoked via RunnableLambda -- matches the user's
# explicit instruction: "use ... AddableDict and a RunnableLambda for any
# caching simulation, but make caching opt-in and explicit".
# =============================================================================

class ExplicitCache:
    """Opt-in prompt-keyed cache backed by langchain_core's InMemoryCache.

    Stores responses under a JSON-serialised input dict; tracks hit/miss
    counts so per-trial cache_hit_rate can be reported.
    """

    LLM_STRING = "langchain-baseline-mock"

    def __init__(self) -> None:
        self._inner = InMemoryCache()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        result = self._inner.lookup(key, self.LLM_STRING)
        if result:
            self.hits += 1
            return result[0].text
        self.misses += 1
        return None

    def put(self, key: str, value: str) -> None:
        self._inner.update(key, self.LLM_STRING, [Generation(text=value)])

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def _cache_key(step: str, payload: dict) -> str:
    return step + "::" + json.dumps(payload, sort_keys=True, default=str)


def cached_step(step_name: str, inner, cache: ExplicitCache):
    """Wrap an LCEL chain with explicit cache lookup via RunnableLambda."""

    def _run(payload: dict) -> str:
        key = _cache_key(step_name, payload)
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = inner.invoke(payload)
        cache.put(key, result)
        return result

    return RunnableLambda(_run)


# =============================================================================
# Triage workflow: 3 LLM steps, mirrors create_triage_dag in
# scripts/run_benchmark.py:125. urgency and category are independent; response
# depends on both. Sequential variant chains them via RunnablePassthrough.assign;
# parallel variant uses RunnableParallel for urgency+category.
# =============================================================================

def _triage_step_chains(llm):
    urgency = ChatPromptTemplate.from_template(TRIAGE_URGENCY_TMPL) | llm | StrOutputParser()
    category = ChatPromptTemplate.from_template(TRIAGE_CATEGORY_TMPL) | llm | StrOutputParser()
    response_prompt = ChatPromptTemplate.from_template(TRIAGE_RESPONSE_TMPL)
    response = response_prompt | llm | StrOutputParser()
    return urgency, category, response


def build_triage_chain(llm, parallel: bool, cache: Optional[ExplicitCache] = None):
    urgency, category, response = _triage_step_chains(llm)

    if cache is not None:
        urgency = cached_step("urgency", urgency, cache)
        category = cached_step("category", category, cache)
        response = cached_step("response", response, cache)

    if parallel:
        classify = RunnableParallel(urgency=urgency, category=category)
        merge = RunnableLambda(
            lambda x: AddableDict(
                query=x["query"],
                urgency=x["classifications"]["urgency"],
                category=x["classifications"]["category"],
            )
        )
        return (
            RunnablePassthrough.assign(classifications=classify)
            | merge
            | response
        )

    # Sequential: urgency, then category, then response. Each .assign blocks
    # for one full LLM call before the next runs.
    return (
        RunnablePassthrough.assign(urgency=urgency)
        .assign(category=category)
        | response
    )


# =============================================================================
# Document analysis: 3 independent LLM analyzers + 1 non-LLM combine.
# Mirrors create_extraction_dag in scripts/run_benchmark.py:155.
# =============================================================================

def _extraction_step_chains(llm):
    entities = ChatPromptTemplate.from_template(EXTRACT_ENTITIES_TMPL) | llm | StrOutputParser()
    summary = ChatPromptTemplate.from_template(EXTRACT_SUMMARY_TMPL) | llm | StrOutputParser()
    domain = ChatPromptTemplate.from_template(EXTRACT_DOMAIN_TMPL) | llm | StrOutputParser()
    return entities, summary, domain


def _combine_extraction(d: dict) -> AddableDict:
    out = AddableDict()
    out["entities"] = d["entities"]
    out["summary"] = d["summary"]
    out["domain"] = d["domain"]
    out["combined"] = (
        f"Combine: entities={d['entities']}, "
        f"summary={d['summary']}, domain={d['domain']}"
    )
    return out


def build_extraction_chain(llm, parallel: bool, cache: Optional[ExplicitCache] = None):
    entities, summary, domain = _extraction_step_chains(llm)

    if cache is not None:
        entities = cached_step("entities", entities, cache)
        summary = cached_step("summary", summary, cache)
        domain = cached_step("domain", domain, cache)

    if parallel:
        analyzers = RunnableParallel(entities=entities, summary=summary, domain=domain)
    else:
        def _seq(x: dict) -> AddableDict:
            d = AddableDict()
            d["entities"] = entities.invoke(x)
            d["summary"] = summary.invoke(x)
            d["domain"] = domain.invoke(x)
            return d

        analyzers = RunnableLambda(_seq)

    return analyzers | RunnableLambda(_combine_extraction)


# =============================================================================
# Trial runner -- one full pass over a dataset; collects per-item latencies,
# per-item parallelization_factor (empirical: sum_call_ms / wall_clock_ms),
# and cache hit/miss counters.
# =============================================================================

def _build_chain(scenario: str, llm, parallel: bool, cache):
    if scenario == "triage":
        return build_triage_chain(llm, parallel=parallel, cache=cache)
    if scenario == "extraction":
        return build_extraction_chain(llm, parallel=parallel, cache=cache)
    raise ValueError(f"unknown scenario: {scenario}")


def _item_inputs(scenario: str, item: dict) -> dict:
    if scenario == "triage":
        return {
            "query": item["query"],
            "customer_tier": item.get("context", {}).get("customer_tier", "free"),
        }
    if scenario == "extraction":
        return {"document": item["document"]}
    raise ValueError(f"unknown scenario: {scenario}")


def _llm_usage_tracker(llm) -> UsageTracker:
    """Return the single UsageTracker attached to llm.callbacks, creating one
    if it isn't already attached. Attaching to the LLM (vs passing via
    chain.invoke config) ensures cached_step's inner.invoke() — which doesn't
    propagate config — still records usage during the parallel_cached warmup.
    """
    cbs = getattr(llm, "callbacks", None)
    if cbs:
        for cb in cbs:
            if isinstance(cb, UsageTracker):
                return cb
    tracker = UsageTracker()
    new_cbs = list(cbs or []) + [tracker]
    try:
        llm.callbacks = new_cbs
    except (AttributeError, TypeError):
        # ChatOpenAI is pydantic — assignment usually works, but DeterministicMockChat
        # subclasses BaseChatModel which is also pydantic. If assignment fails, fall
        # back to setattr through __dict__ (only used in mock-mode self-check anyway).
        object.__setattr__(llm, "callbacks", new_cbs)
    return tracker


def run_one_trial(
    scenario: str,
    items: list[dict],
    llm,
    parallel: bool,
    cache: Optional[ExplicitCache],
) -> dict:
    """Execute the chain over all items and return a per-trial measurements dict.

    Schema matches scripts/run_benchmark.py:_run_single_trial so it composes
    cleanly with aggregate_trials() / _trial_for_json().
    """
    cache_hits_start = cache.hits if cache else 0
    cache_misses_start = cache.misses if cache else 0

    chain = _build_chain(scenario, llm, parallel=parallel, cache=cache)
    tracker = _llm_usage_tracker(llm)
    in_start, out_start = tracker.snapshot()

    latencies_ms: list[float] = []
    parallelization_factors: list[float] = []
    tokens_total = 0
    tokens_saved = 0
    successful = 0
    failed = 0

    start = time.perf_counter()
    for item in items:
        _drain_call_durations_ms()  # reset per-item recorder
        item_start = time.perf_counter()
        try:
            chain.invoke(_item_inputs(scenario, item))
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

    in_end, out_end = tracker.snapshot()
    tokens_input = in_end - in_start
    tokens_output = out_end - out_start
    tokens_total = tokens_input + tokens_output

    return {
        "latencies_ms": latencies_ms,
        "tokens_total": tokens_total,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_saved": tokens_saved,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hit_rate,
        "errors": failed,
        "successful": successful,
        "total_time_ms": total_time_ms,
        "parallelization_factor_mean": pf_mean,
        "level_execution_times_ms_mean": [],  # not measured for LangChain baseline
    }


# =============================================================================
# Suite driver: dataset x config x N trials, with parallel_cached warmup
# protocol (mirrors scripts/run_benchmark.py:run_config_trials:492-512).
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
    llm,
    config: str,
    n_trials: int,
) -> tuple[list[dict], int, int]:
    """Returns (trials, warmup_tokens_input, warmup_tokens_output).

    For parallel_cached, the warmup pass IS a real API hit (cache is empty),
    so its tokens are real cost even though the warmup is otherwise discarded
    from per-trial latency aggregates. We surface the warmup token counts so
    the merge step can include them in actual_cost_usd. For sequential and
    parallel, no warmup happens and the returned warmup tokens are 0.
    """
    assert config in SUITE_CONFIGS, f"unknown config: {config}"
    parallel = (config != "sequential")
    trials: list[dict] = []
    warmup_tokens_input = 0
    warmup_tokens_output = 0

    if config == "parallel_cached":
        cache = ExplicitCache()
        # Warmup pass (discarded for latency, but its tokens are real cost).
        warmup = run_one_trial(scenario, items, llm, parallel=True, cache=cache)
        warmup_tokens_input = int(warmup.get("tokens_input", 0))
        warmup_tokens_output = int(warmup.get("tokens_output", 0))
        if warmup["cache_hit_rate"] > 0.05:
            raise RuntimeError(
                f"parallel_cached warmup expected ~0% hit rate but saw "
                f"{warmup['cache_hit_rate']:.3f} -- ExplicitCache state leaking"
            )
        for i in range(n_trials):
            print(f"    trial {i+1}/{n_trials} ({config})...", flush=True)
            trial = run_one_trial(scenario, items, llm, parallel=True, cache=cache)
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
            trial = run_one_trial(scenario, items, llm, parallel=parallel, cache=None)
            trial["trial"] = i
            p50, p95, p99 = _trial_percentiles(trial["latencies_ms"])
            trial["p50"], trial["p95"], trial["p99"] = p50, p95, p99
            trials.append(trial)
    return trials, warmup_tokens_input, warmup_tokens_output


def _trial_for_json(t: dict) -> dict:
    """Slim per-trial dict for raw_trial_results (mirrors run_benchmark.py)."""
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
        "tokens_input": t.get("tokens_input", 0),
        "tokens_output": t.get("tokens_output", 0),
        "tokens_saved": t["tokens_saved"],
        "errors": t["errors"],
        "parallelization_factor_mean": round(t["parallelization_factor_mean"], 4),
        "level_execution_times_ms_mean": [round(x, 4) for x in t["level_execution_times_ms_mean"]],
    }


# =============================================================================
# Cost guard for real-API mode (BASELINE_PROVIDER=openai or --mode openai)
# =============================================================================

# gpt-4o-mini public pricing (as of 2026-05): $0.15/1M input, $0.60/1M output.
COST_PER_INPUT_TOKEN_USD = 0.15e-6
COST_PER_OUTPUT_TOKEN_USD = 0.60e-6
COST_GUARD_USD = 5.0


def estimate_cost_usd(
    scenario_items_pairs: list[tuple[str, list[dict]]],
    n_trials: int,
    n_configs: int,
    expected_output_tokens_per_call: int = 30,
) -> dict:
    """Estimate USD cost for a real-API run.

    Sum input chars across all (scenario, item, step), //4 for tokens.
    Multiply by trials * configs.
    """
    total_input_tokens = 0
    total_calls = 0
    for scenario, items in scenario_items_pairs:
        if scenario == "triage":
            tmpls = [TRIAGE_URGENCY_TMPL, TRIAGE_CATEGORY_TMPL, TRIAGE_RESPONSE_TMPL]
            for item in items:
                ctx = item.get("context", {})
                for tmpl in tmpls:
                    rendered = tmpl.format(
                        query=item["query"],
                        customer_tier=ctx.get("customer_tier", "free"),
                        urgency="<placeholder>",
                        category="<placeholder>",
                    )
                    total_input_tokens += len(rendered) // 4
                    total_calls += 1
        elif scenario == "extraction":
            tmpls = [EXTRACT_ENTITIES_TMPL, EXTRACT_SUMMARY_TMPL, EXTRACT_DOMAIN_TMPL]
            for item in items:
                for tmpl in tmpls:
                    rendered = tmpl.format(document=item["document"])
                    total_input_tokens += len(rendered) // 4
                    total_calls += 1
    # Multiply by trial count and config count (each config re-runs all calls)
    total_input_tokens *= n_trials * n_configs
    total_calls *= n_trials * n_configs
    total_output_tokens = total_calls * expected_output_tokens_per_call
    cost_usd = (
        total_input_tokens * COST_PER_INPUT_TOKEN_USD
        + total_output_tokens * COST_PER_OUTPUT_TOKEN_USD
    )
    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "calls": total_calls,
        "cost_usd": cost_usd,
    }


# =============================================================================
# CLI
# =============================================================================

def _self_check() -> int:
    """Verify mock latency parity: a single .invoke() should be ~50 ms."""
    print("Imports verified:")
    print("  from langchain_core.language_models.chat_models import BaseChatModel")
    print("  from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough")
    print("  from langchain_core.runnables.utils import AddableDict")
    print("  from langchain_core.caches import InMemoryCache")
    print("  from langchain_core.globals import set_llm_cache")
    print("  from langchain_core.prompts import ChatPromptTemplate")
    print("  from langchain_core.output_parsers import StrOutputParser")

    llm = DeterministicMockChat()
    chain = ChatPromptTemplate.from_template("{q}") | llm | StrOutputParser()
    # Warmup once (LangChain compiles graph on first call)
    chain.invoke({"q": "warmup"})
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        chain.invoke({"q": "ping"})
        samples.append((time.perf_counter() - t0) * 1000.0)
    median = sorted(samples)[len(samples) // 2]
    print(f"\nMock single-invoke median wall time: {median:.2f} ms (samples: {[round(s,2) for s in samples]})")
    if 45.0 <= median <= 75.0:
        print(f"50ms mock parity OK (target: aether-runtime/src/llm.rs:273 latency_ms=50)")
        return 0
    print(f"PARITY FAILURE: median {median:.2f} outside [45, 75] ms window", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="LangChain baseline benchmark")
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
        default=str(REPO_ROOT / "bench" / "results" / "langchain_v1.json"),
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

    # Pick scenarios + datasets
    selected_datasets = (
        SUITE_DATASETS
        if args.scenario == "all"
        else [d for d in SUITE_DATASETS if d[1] == args.scenario]
    )
    if not selected_datasets:
        print(f"No datasets matched scenario '{args.scenario}'", file=sys.stderr)
        return 1
    selected_configs = SUITE_CONFIGS if args.config == "all" else [args.config]

    # Load datasets
    scenario_items: list[tuple[str, str, list[dict]]] = []
    for dataset_name, scenario, rel_path in selected_datasets:
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"Dataset not found: {path}", file=sys.stderr)
            return 1
        items = load_jsonl(path)
        scenario_items.append((dataset_name, scenario, items))
        print(f"Loaded {len(items)} items from {dataset_name}", flush=True)

    # Build LLM
    if args.mode == "mock":
        llm = DeterministicMockChat()
        provider_label = "mock"
        version_label = _installed_langchain_version()
    elif args.mode == "openai":
        # Cost guard
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
        try:
            from langchain_openai import ChatOpenAI  # noqa: F401
        except ImportError:
            print("langchain-openai not installed", file=sys.stderr)
            return 4
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set", file=sys.stderr)
            return 5
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        provider_label = "openai"
        version_label = _installed_langchain_version()
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        return 1

    # Run suite
    results: list[dict] = []
    for dataset_name, scenario, items in scenario_items:
        for config in selected_configs:
            print(f"  {dataset_name} / {config} ({args.trials} trials)...", flush=True)
            trials, warmup_in, warmup_out = run_config_trials(
                scenario, items, llm, config, args.trials
            )
            agg = aggregate_trials(trials)
            entry = {
                "dataset": dataset_name,
                "config": config,
                "trials": args.trials,
                **agg,
                "warmup_tokens_input": warmup_in,
                "warmup_tokens_output": warmup_out,
                "raw_trial_results": [_trial_for_json(t) for t in trials],
            }
            results.append(entry)

    # Write JSON
    out = {
        "system": "langchain",
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


def _installed_langchain_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("langchain")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
