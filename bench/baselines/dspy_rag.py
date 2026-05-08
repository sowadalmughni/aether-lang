#!/usr/bin/env python3
"""DSPy HotpotQA RAG baseline.

Three-step pipeline mirroring examples/rag_qa.aether:

  1. SubqueryDecomposer (dspy.Predict)  -- decompose multi-hop question
  2. retrieve_context  -- non-LM Python passthrough of dataset paragraphs
                          (mock retriever; swap for dspy.Retrieve later)
  3. AnswerQuestion (dspy.Predict)  -- short answer over retrieved context

Reuses DeterministicMockLM from dspy_baseline.py (50 ms flat latency,
ChatAdapter-formatted SHA-prefixed output) so latency-parity claims with
LangChain and Aether mock providers carry over.

Output: bench/results/hotpotqa_dspy_v1.json with the same EM/F1-aware
schema as bench/baselines/langchain_rag.py.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "bench" / "baselines"))

import dspy  # noqa: E402
from dspy import ChatAdapter, InputField, Module, OutputField, Predict, Prediction, Signature  # noqa: E402

from dspy_baseline import DeterministicMockLM, _drain_call_durations_ms  # noqa: E402
from run_benchmark import (  # noqa: E402
    _bootstrap_ci,
    _trial_percentiles,
    aggregate_trials,
    get_git_version,
    get_hardware_info,
)
from _hotpot_common import (  # noqa: E402
    DEFAULT_DATASET,
    evaluate_qa,
    extract_short_answer,
    load_hotpot_dataset,
    per_item_em_f1,
    serialize_paragraphs,
)


# -----------------------------------------------------------------------------
# DSPy signatures
# -----------------------------------------------------------------------------

class SubqueryDecomposer(Signature):
    """Decompose a multi-hop question into 2-3 single-hop subqueries that, answered together, would answer the original question. Return as a JSON array of strings."""

    question: str = InputField()
    subqueries: str = OutputField(desc="JSON array of 2-3 single-hop subqueries")


class AnswerQuestion(Signature):
    """Answer the multi-hop question using only the provided context. Give the shortest possible answer (yes/no or a short noun phrase). Do not explain your reasoning."""

    question: str = InputField()
    context: str = InputField(desc="Concatenated paragraphs containing the answer")
    subqueries: str = InputField(desc="The decomposed single-hop subqueries")
    answer: str = OutputField(desc="The shortest possible answer")


# -----------------------------------------------------------------------------
# DSPy module: 3-step RAG (decompose -> retrieve -> answer)
# -----------------------------------------------------------------------------

class HotpotRAG(Module):
    """Sequential RAG. retrieve_context is a non-LM passthrough so its latency
    is ~0 -- in real deployments swap it for a dspy.Retrieve subclass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.decompose = Predict(SubqueryDecomposer)
        self.answer_step = Predict(AnswerQuestion)

    @staticmethod
    def retrieve_context(subqueries: str, paragraphs: str) -> str:
        # Mock retriever: returns the dataset's pre-supplied paragraphs.
        # The `subqueries` arg is unused in mock mode but kept in the
        # signature so the conceptual data flow matches the .aether file.
        del subqueries
        return paragraphs

    def forward(self, question: str, paragraphs: str) -> Prediction:  # type: ignore[override]
        sub_pred = self.decompose(question=question)
        subqueries = sub_pred.subqueries
        context = self.retrieve_context(subqueries=subqueries, paragraphs=paragraphs)
        ans_pred = self.answer_step(
            question=question,
            context=context,
            subqueries=subqueries,
        )
        return Prediction(
            subqueries=subqueries,
            context=context,
            answer=ans_pred.answer,
        )


# -----------------------------------------------------------------------------
# Trial runner
# -----------------------------------------------------------------------------

def _lm_history_len() -> int:
    lm = getattr(dspy.settings, "lm", None)
    hist = getattr(lm, "history", None) if lm else None
    return len(hist) if hist is not None else 0


def _sum_lm_usage(start_idx: int) -> tuple[int, int]:
    lm = getattr(dspy.settings, "lm", None)
    hist = getattr(lm, "history", None) if lm else None
    if not hist:
        return 0, 0
    inp = out = 0
    for entry in hist[start_idx:]:
        u = (entry or {}).get("usage") or {}
        inp += int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
        out += int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
    return inp, out


def run_one_trial(items: list[dict]) -> dict:
    rag = HotpotRAG()
    history_start = _lm_history_len()

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
            pred = rag(
                question=item["query"],
                paragraphs=serialize_paragraphs(item["context_paragraphs"]),
            )
            ans = extract_short_answer(pred.answer)
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

    tokens_input, tokens_output = _sum_lm_usage(history_start)
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


def run_trials(items: list[dict], n_trials: int) -> list[dict]:
    trials: list[dict] = []
    for i in range(n_trials):
        print(f"    trial {i+1}/{n_trials}...", flush=True)
        t = run_one_trial(items)
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
                "cache_hit_rate": 0.0,
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
        "errors": t["errors"],
        "successful": t["successful"],
        "total_time_ms": round(t["total_time_ms"], 4),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="DSPy HotpotQA RAG baseline")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--mode", choices=["mock", "openai"], default="mock")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "bench" / "results" / "hotpotqa_dspy_v1.json"),
    )
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    items = load_hotpot_dataset(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"Loaded {len(items)} items from {args.dataset}", flush=True)

    if args.mode == "mock":
        dspy.configure(lm=DeterministicMockLM(), adapter=ChatAdapter())
        provider_label = "mock"
    elif args.mode == "openai":
        if not args.confirm_cost:
            print("--mode openai requires --confirm-cost (cost guard)", file=sys.stderr)
            return 3
        import os as _os
        if not _os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set", file=sys.stderr)
            return 5
        dspy.configure(
            lm=dspy.LM("openai/gpt-4o-mini", temperature=0, cache=False),
            adapter=ChatAdapter(),
        )
        provider_label = "openai"
    else:
        return 1

    trials = run_trials(items, args.trials)
    agg = aggregate_qa_trials(trials)

    try:
        version_label = importlib.metadata.version("dspy-ai")
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
        "system": "dspy",
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
