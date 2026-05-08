#!/usr/bin/env python3
"""LangChain HotpotQA RAG baseline.

Three-step pipeline mirroring examples/rag_qa.aether:

  1. extract_subqueries  -- LLM call: decompose multi-hop question
  2. retrieve_context    -- non-LLM passthrough of the dataset's distractor
                            paragraphs (mock retriever)
  3. answer_question     -- LLM call: short answer over retrieved context

The mock LLM is the same DeterministicMockChat from langchain_baseline.py
(50 ms flat latency, deterministic SHA-prefixed output) so the latency
parity claim from the existing langchain baseline carries over.

Output: bench/results/hotpotqa_langchain_v1.json with the schema:
  {
    system, version, provider, hardware, dataset, dataset_size,
    trials_per_config, configs, measured_at, results: [
      {dataset, config, trials, em_mean, f1_mean, latency_p50_ms, ...,
       raw_trial_results: [{trial, em, f1, latencies_ms, ...}, ...]}
    ]
  }

Usage:
  py -3.13 bench/baselines/langchain_rag.py --trials 3 --mode mock \
      --output bench/results/hotpotqa_langchain_v1.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "bench" / "baselines"))

# Reuse the deterministic mock + helpers from langchain_baseline.py
from langchain_baseline import (  # noqa: E402
    DeterministicMockChat,
    UsageTracker,
    _llm_usage_tracker,
    _drain_call_durations_ms,
)
from run_benchmark import (  # noqa: E402
    _trial_percentiles,
    aggregate_trials,
    get_git_version,
    get_hardware_info,
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

# LangChain runnables
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: E402


# -----------------------------------------------------------------------------
# Chain construction
# -----------------------------------------------------------------------------

def _decompose_chain(llm):
    return (
        ChatPromptTemplate.from_template(RAG_PROMPTS["decompose"])
        | llm
        | StrOutputParser()
    )


def _answer_chain(llm):
    return (
        ChatPromptTemplate.from_template(RAG_PROMPTS["answer"])
        | llm
        | StrOutputParser()
    )


def build_rag_chain(llm):
    """Sequential 3-step chain. Each step depends on the previous, so there's
    no parallelism opportunity here -- this matches the .aether flow shape.

    The 'retrieve' step is a non-LLM RunnableLambda that just forwards the
    dataset's pre-supplied paragraphs (mock retriever). The chain output is
    the post-processed short answer string.
    """
    decompose = _decompose_chain(llm)
    answer = _answer_chain(llm)

    def _retrieve(payload: dict) -> dict:
        # Mock retriever: forwards pre-serialised paragraphs as `context`.
        # No LLM call, no latency simulation -- the actual retrieval would
        # plug in here (BM25 / dense vector / hybrid).
        return {
            "question": payload["question"],
            "subqueries": payload["subqueries"],
            "context": payload["paragraphs"],
        }

    chain = (
        RunnablePassthrough.assign(subqueries=decompose)
        | RunnableLambda(_retrieve)
        | answer
        | RunnableLambda(extract_short_answer)
    )
    return chain


# -----------------------------------------------------------------------------
# Trial runner
# -----------------------------------------------------------------------------

def run_one_trial(items: list[dict], llm) -> dict:
    """One pass over all items. Returns a per-trial measurements dict that
    composes with aggregate_trials() / _trial_percentiles().
    """
    chain = build_rag_chain(llm)
    tracker = _llm_usage_tracker(llm)
    in_start, out_start = tracker.snapshot()

    latencies_ms: list[float] = []
    predictions: dict[str, str] = {}
    golds: dict[str, str] = {}
    em_per_item: list[float] = []
    f1_per_item: list[float] = []
    successful = 0
    failed = 0

    start = time.perf_counter()
    for item in items:
        _drain_call_durations_ms()
        item_start = time.perf_counter()
        try:
            payload = {
                "question": item["query"],
                "paragraphs": serialize_paragraphs(item["context_paragraphs"]),
            }
            ans = chain.invoke(payload)
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
        latencies_ms.append((time.perf_counter() - item_start) * 1000.0)
    total_time_ms = (time.perf_counter() - start) * 1000.0

    in_end, out_end = tracker.snapshot()
    tokens_input = in_end - in_start
    tokens_output = out_end - out_start

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
        "tokens_total": tokens_input + tokens_output,
        "errors": failed,
        "successful": successful,
        "total_time_ms": total_time_ms,
    }


def run_trials(items: list[dict], llm, n_trials: int) -> list[dict]:
    trials: list[dict] = []
    for i in range(n_trials):
        print(f"    trial {i+1}/{n_trials}...", flush=True)
        t = run_one_trial(items, llm)
        t["trial"] = i
        p50, p95, p99 = _trial_percentiles(t["latencies_ms"])
        t["p50"], t["p95"], t["p99"] = p50, p95, p99
        trials.append(t)
    return trials


# -----------------------------------------------------------------------------
# Aggregation -- adds em_mean / f1_mean across-trial bootstrap CIs to the
# generic latency aggregation from run_benchmark.py.
# -----------------------------------------------------------------------------

def aggregate_qa_trials(trials: list[dict]) -> dict:
    """Like aggregate_trials() but also includes em_mean and f1_mean."""
    base = aggregate_trials(
        [
            {
                "p50": t["p50"], "p95": t["p95"], "p99": t["p99"],
                # cache is not used in this RAG baseline; reuse the same
                # dict shape so aggregate_trials doesn't blow up.
                "cache_hit_rate": 0.0,
                "tokens_saved": 0,
            }
            for t in trials
        ]
    )
    from run_benchmark import _bootstrap_ci  # local import to avoid circularity at top
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
        "errors": t["errors"],
        "successful": t["successful"],
        "total_time_ms": round(t["total_time_ms"], 4),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="LangChain HotpotQA RAG baseline")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument(
        "--mode",
        choices=["mock", "openai"],
        default="mock",
        help="LLM provider mode",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to bench/datasets/hotpotqa_dev_500.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run the first N items (debugging)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "bench" / "results" / "hotpotqa_langchain_v1.json"),
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Required for --mode openai (cost guard)",
    )
    args = parser.parse_args()

    items = load_hotpot_dataset(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"Loaded {len(items)} items from {args.dataset}", flush=True)

    if args.mode == "mock":
        llm = DeterministicMockChat()
        provider_label = "mock"
    elif args.mode == "openai":
        if not args.confirm_cost:
            print("--mode openai requires --confirm-cost (cost guard)", file=sys.stderr)
            return 3
        try:
            from langchain_openai import ChatOpenAI  # noqa: F401
        except ImportError:
            print("langchain-openai not installed", file=sys.stderr)
            return 4
        import os as _os
        if not _os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set", file=sys.stderr)
            return 5
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        provider_label = "openai"
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        return 1

    trials = run_trials(items, llm, args.trials)
    agg = aggregate_qa_trials(trials)

    try:
        version_label = importlib.metadata.version("langchain")
    except Exception:
        version_label = "unknown"

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
        "system": "langchain",
        "version": version_label,
        "provider": provider_label,
        "hardware": get_hardware_info(),
        "dataset": "hotpotqa_dev_500",
        "dataset_size": len(items),
        "trials_per_config": args.trials,
        "configs": ["rag_sequential"],
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "git_version": get_git_version(REPO_ROOT),
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
