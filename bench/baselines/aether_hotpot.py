#!/usr/bin/env python3
"""Aether HotpotQA RAG runner.

Drives the Aether runtime (HTTP /execute) over the HotpotQA dev_500
dataset using a 3-node DAG that mirrors examples/rag_qa.aether:

    extract_subqueries (llm_fn) -> retrieve_context (compute) -> answer_question (llm_fn)

`retrieve_context` is a `compute` node so the runtime doesn't bill it
as an LLM call -- in mock mode the runtime simply returns
"Function retrieve_context executed", which is enough to express the
DAG dependency. The static fields (question, paragraphs) are inlined
into the prompt strings before submission so the runtime mock can
reproduce them deterministically without context-variable plumbing.

The runner uses the same launch / health-check / spawn protocol as
scripts/run_benchmark.py: if no runtime is reachable at AETHER_RUNTIME_URL,
it spawns `cargo run --release -p aether-runtime` with AETHER_PROVIDER=mock,
waits for /health, and tears it down on exit.

Output: bench/results/hotpotqa_aether_v1.json with the same EM/F1-aware
schema as hotpotqa_langchain_v1.json / hotpotqa_dspy_v1.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "bench" / "baselines"))

from run_benchmark import (  # noqa: E402
    DEFAULT_RUNTIME_URL,
    _bootstrap_ci,
    _trial_percentiles,
    _shutdown_runtime,
    aggregate_trials,
    check_runtime,
    clear_cache,
    get_cache_stats,
    get_git_version,
    get_hardware_info,
    launch_runtime,
)
from _hotpot_common import (  # noqa: E402
    DEFAULT_DATASET,
    RAG_PROMPTS,
    evaluate_qa,
    extract_short_answer,
    load_hotpot_dataset,
    per_item_em_f1,
    serialize_paragraphs,
)


# -----------------------------------------------------------------------------
# DAG construction
# -----------------------------------------------------------------------------

def build_rag_dag(item_id: str, question: str, paragraphs: str) -> dict:
    """Build the 3-node HotpotQA RAG DAG for one item.

    Static fields (question, paragraphs) are inlined into the prompt
    strings; cross-node references use {{node.X.output}} so the runtime
    template engine substitutes them at execution time.
    """
    decompose_prompt = RAG_PROMPTS["decompose"].format(question=question)
    # The answer node references the prior subquery output via {{node.X.output}}.
    # Paragraphs and question are baked in. Subqueries arrive as the literal
    # output of extract_subqueries (which in mock mode is the [Mock Response]
    # string -- that's fine, EM/F1 in mock mode is 0 by construction).
    answer_prompt = (
        "Answer the multi-hop question using only the provided context. "
        "Give the shortest possible answer (yes/no or a short noun phrase). "
        "Do not explain your reasoning.\n\n"
        f"Context:\n{paragraphs}\n\n"
        "Subqueries you should resolve:\n{{node.extract_subqueries.output}}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return {
        "id": f"rag_qa_{item_id}",
        "nodes": [
            {
                "id": "extract_subqueries",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 200,
                "prompt": decompose_prompt,
                "dependencies": [],
            },
            {
                "id": "retrieve_context",
                "node_type": "compute",
                # The compute node body is documentation only -- the runtime
                # treats compute nodes as no-ops for now (returns "Function
                # retrieve_context executed"). Keeping the dependency makes
                # the DAG shape match the .aether file.
                "prompt": "Mock retriever passthrough; subqueries={{node.extract_subqueries.output}}",
                "dependencies": ["extract_subqueries"],
            },
            {
                "id": "answer_question",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 80,
                "prompt": answer_prompt,
                "dependencies": ["retrieve_context"],
            },
        ],
    }


# -----------------------------------------------------------------------------
# Runtime invocation
# -----------------------------------------------------------------------------

def execute_dag(url: str, dag: dict, timeout_s: float = 60.0) -> dict:
    payload = {"dag": dag}
    data = json.dumps(payload).encode("utf-8")
    req = Request(f"{url}/execute", data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    start = time.perf_counter()
    with urlopen(req, timeout=timeout_s) as resp:
        result = json.loads(resp.read().decode())
    result["client_latency_ms"] = (time.perf_counter() - start) * 1000.0
    return result


def find_node_output(result: dict, node_id: str) -> str:
    """Locate the named node's output string in a /execute response.

    The runtime returns `{ "results": [ {node_id, output, ...}, ... ] }`.
    """
    for nr in result.get("results") or []:
        if nr.get("node_id") == node_id or nr.get("id") == node_id:
            return str(nr.get("output", ""))
    return ""


# -----------------------------------------------------------------------------
# Trial runner
# -----------------------------------------------------------------------------

def run_one_trial(runtime_url: str, items: list[dict]) -> dict:
    initial_stats = get_cache_stats(runtime_url)

    latencies_ms: list[float] = []
    predictions: dict[str, str] = {}
    golds: dict[str, str] = {}
    em_per_item: list[float] = []
    f1_per_item: list[float] = []
    tokens_input = 0
    tokens_output = 0
    tokens_total = 0
    successful = 0
    failed = 0

    start = time.perf_counter()
    for item in items:
        try:
            dag = build_rag_dag(
                item_id=item["id"],
                question=item["query"],
                paragraphs=serialize_paragraphs(item["context_paragraphs"]),
            )
            result = execute_dag(runtime_url, dag)
            wall_ms = result.get("total_execution_time_ms", result["client_latency_ms"])
            latencies_ms.append(float(wall_ms))
            tokens_total += int(result.get("total_token_cost", 0) or 0)
            for nr in result.get("results") or []:
                tokens_input += int(nr.get("input_tokens", 0) or 0)
                tokens_output += int(nr.get("output_tokens", 0) or 0)
            raw_answer = find_node_output(result, "answer_question")
            ans = extract_short_answer(raw_answer)
            predictions[item["id"]] = ans
            golds[item["id"]] = item["expected_answer"]
            em, f1 = per_item_em_f1(ans, item["expected_answer"])
            em_per_item.append(em)
            f1_per_item.append(f1)
            successful += 1
        except Exception as exc:
            print(f"item {item.get('id')} failed: {exc}", file=sys.stderr)
            failed += 1
            em_per_item.append(0.0)
            f1_per_item.append(0.0)
            latencies_ms.append(0.0)

    total_time_ms = (time.perf_counter() - start) * 1000.0

    final_stats = get_cache_stats(runtime_url)
    cache_hits = final_stats.get("hits", 0) - initial_stats.get("hits", 0)
    cache_misses = final_stats.get("misses", 0) - initial_stats.get("misses", 0)
    cache_hit_rate = (
        cache_hits / (cache_hits + cache_misses)
        if (cache_hits + cache_misses) > 0
        else 0.0
    )

    agg = evaluate_qa(predictions, golds)
    return {
        "latencies_ms": latencies_ms,
        "em_per_item": em_per_item,
        "f1_per_item": f1_per_item,
        "em_mean": agg["em"],
        "f1_mean": agg["f1"],
        "prec_mean": agg["prec"],
        "recall_mean": agg["recall"],
        "n_eval": agg["n"],
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_total": tokens_total,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hit_rate,
        "errors": failed,
        "successful": successful,
        "total_time_ms": total_time_ms,
    }


def run_trials(runtime_url: str, items: list[dict], n_trials: int) -> list[dict]:
    trials: list[dict] = []
    for i in range(n_trials):
        clear_cache(runtime_url)  # cold each trial -- HotpotQA queries are unique
        print(f"    trial {i+1}/{n_trials}...", flush=True)
        t = run_one_trial(runtime_url, items)
        t["trial"] = i
        p50, p95, p99 = _trial_percentiles(t["latencies_ms"])
        t["p50"], t["p95"], t["p99"] = p50, p95, p99
        trials.append(t)
    return trials


def aggregate_qa_trials(trials: list[dict]) -> dict:
    base = aggregate_trials(
        [
            {
                "p50": t["p50"], "p95": t["p95"], "p99": t["p99"],
                "cache_hit_rate": t.get("cache_hit_rate", 0.0),
                "tokens_saved": 0,
            }
            for t in trials
        ]
    )
    base["em"] = _bootstrap_ci([t["em_mean"] for t in trials])
    base["f1"] = _bootstrap_ci([t["f1_mean"] for t in trials])
    base["prec"] = _bootstrap_ci([t["prec_mean"] for t in trials])
    base["recall"] = _bootstrap_ci([t["recall_mean"] for t in trials])
    return base


def _trial_for_json(t: dict) -> dict:
    return {
        "trial": t["trial"],
        "em": round(t["em_mean"], 6),
        "f1": round(t["f1_mean"], 6),
        "prec": round(t["prec_mean"], 6),
        "recall": round(t["recall_mean"], 6),
        "n_eval": t["n_eval"],
        "p50": round(t["p50"], 4),
        "p95": round(t["p95"], 4),
        "p99": round(t["p99"], 4),
        "latencies_ms": [round(x, 4) for x in t["latencies_ms"]],
        "tokens_input": t["tokens_input"],
        "tokens_output": t["tokens_output"],
        "tokens_total": t["tokens_total"],
        "cache_hits": t["cache_hits"],
        "cache_misses": t["cache_misses"],
        "cache_hit_rate": round(t["cache_hit_rate"], 6),
        "errors": t["errors"],
        "successful": t["successful"],
        "total_time_ms": round(t["total_time_ms"], 4),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Aether HotpotQA RAG runner")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=0, help=">0: only run first N items")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "bench" / "results" / "hotpotqa_aether_v1.json"),
    )
    parser.add_argument("--runtime-url", default=DEFAULT_RUNTIME_URL)
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Don't spawn the runtime; assume operator already started it",
    )
    args = parser.parse_args()

    items = load_hotpot_dataset(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"Loaded {len(items)} items from {args.dataset}", flush=True)

    spawned: Optional[object] = None
    if check_runtime(args.runtime_url):
        print(f"Reusing runtime at {args.runtime_url}", flush=True)
    elif args.no_autostart:
        print(
            f"Runtime not available at {args.runtime_url} and --no-autostart was set",
            file=sys.stderr,
        )
        return 1
    else:
        try:
            spawned = launch_runtime(REPO_ROOT, args.runtime_url)
        except RuntimeError as exc:
            print(f"Failed to launch runtime: {exc}", file=sys.stderr)
            return 1

    try:
        trials = run_trials(args.runtime_url, items, args.trials)
    finally:
        if spawned is not None:
            _shutdown_runtime(spawned)

    agg = aggregate_qa_trials(trials)

    entry = {
        "dataset": "hotpotqa_dev_500",
        "config": "rag_sequential",
        "trials": args.trials,
        "em_mean": agg["em"]["mean"],
        "f1_mean": agg["f1"]["mean"],
        "prec_mean": agg["prec"]["mean"],
        "recall_mean": agg["recall"]["mean"],
        "em": agg["em"],
        "f1": agg["f1"],
        "prec": agg["prec"],
        "recall": agg["recall"],
        "latency_p50_ms": agg["latency_p50_ms"],
        "latency_p95_ms": agg["latency_p95_ms"],
        "latency_p99_ms": agg["latency_p99_ms"],
        "raw_trial_results": [_trial_for_json(t) for t in trials],
    }

    out = {
        "system": "aether",
        "version": get_git_version(REPO_ROOT),
        "provider": os.environ.get("AETHER_PROVIDER", "mock"),
        "hardware": get_hardware_info(),
        "dataset": "hotpotqa_dev_500",
        "dataset_size": len(items),
        "trials_per_config": args.trials,
        "configs": ["rag_sequential"],
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "results": [entry],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"\nWrote {out_path}\n"
        f"  EM mean: {agg['em']['mean']:.4f}\n"
        f"  F1 mean: {agg['f1']['mean']:.4f}\n"
        f"  p50:     {agg['latency_p50_ms']['mean']:.2f} ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
