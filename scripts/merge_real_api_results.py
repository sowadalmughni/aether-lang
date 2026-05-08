#!/usr/bin/env python3
"""Merge aether/langchain/dspy real-API benchmark JSONs into one combined
artifact at bench/results/real_api_v1.json, and compute total USD cost from
the per-trial `tokens_input` / `tokens_output` fields recorded by:
  - bench/baselines/langchain_baseline.py:UsageTracker (LangChain callback
    on response.llm_output['token_usage'])
  - bench/baselines/dspy_baseline.py:_sum_lm_usage (reads dspy.settings.lm.history)
  - scripts/run_benchmark.py:_run_single_trial (sums per-node input_tokens /
    output_tokens from the runtime's DagExecutionResponse)

Cost is *measured*, not estimated: it is computed as
    in_tok * input_rate + out_tok * output_rate
where in_tok and out_tok come straight from the OpenAI response usage that
each baseline persisted into its JSON.

The combined JSON also records the upfront estimate (passed in by the wrapper
script) so reviewers can see whether actual cost stayed under the budget the
script was started with.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: missing {path}", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_tokens_from_results(system_doc: dict) -> tuple[int, int, int, int]:
    """Sum tokens across all (dataset, config) entries.

    Returns (input_total, output_total, warmup_input_total, warmup_output_total).
    Per-trial counts come from the OpenAI response usage that each baseline
    persisted into raw_trial_results[].tokens_input/output. Warmup counts come
    from the entry-level warmup_tokens_input/output fields written by
    run_config_trials() — those API calls happen during parallel_cached warmup
    and are real cost even though they don't appear in the per-trial latency
    aggregates.
    """
    inp = out = 0
    warm_in = warm_out = 0
    for entry in system_doc.get("results", []) or []:
        for trial in entry.get("raw_trial_results", []) or []:
            inp += int(trial.get("tokens_input", 0) or 0)
            out += int(trial.get("tokens_output", 0) or 0)
        warm_in  += int(entry.get("warmup_tokens_input", 0) or 0)
        warm_out += int(entry.get("warmup_tokens_output", 0) or 0)
    return inp, out, warm_in, warm_out


def main() -> int:
    p = argparse.ArgumentParser(description="Merge real-API benchmark JSONs")
    p.add_argument("--aether",    required=True, type=Path)
    p.add_argument("--langchain", required=True, type=Path)
    p.add_argument("--dspy",      required=True, type=Path)
    p.add_argument("--output",    required=True, type=Path)
    p.add_argument("--budget",    required=True, type=float)
    p.add_argument("--price-input-per-m",  required=True, type=float,
                   help="USD per 1M input tokens")
    p.add_argument("--price-output-per-m", required=True, type=float,
                   help="USD per 1M output tokens")
    p.add_argument("--model",     required=True, type=str,
                   help="OpenAI model name used (e.g. gpt-4o-mini)")
    p.add_argument("--estimated-cost-usd", required=True, type=float,
                   help="The upfront estimate the wrapper script computed")
    args = p.parse_args()

    aether    = _load(args.aether)
    langchain = _load(args.langchain)
    dspy      = _load(args.dspy)

    per_system: dict[str, Any] = {}
    grand_in = grand_out = 0
    for label, doc in [("aether", aether), ("langchain", langchain), ("dspy", dspy)]:
        trial_in, trial_out, warm_in, warm_out = _sum_tokens_from_results(doc)
        in_t = trial_in + warm_in
        out_t = trial_out + warm_out
        cost = in_t * args.price_input_per_m / 1e6 + out_t * args.price_output_per_m / 1e6
        per_system[label] = {
            "system":          doc.get("system", label),
            "version":         doc.get("version"),
            "provider":        doc.get("provider"),
            "trials_per_config": doc.get("trials_per_config"),
            "datasets":        doc.get("datasets"),
            "configs":         doc.get("configs"),
            "tokens_input":    in_t,
            "tokens_output":   out_t,
            "tokens_total":    in_t + out_t,
            "trial_tokens_input":   trial_in,
            "trial_tokens_output":  trial_out,
            "warmup_tokens_input":  warm_in,
            "warmup_tokens_output": warm_out,
            "cost_usd":        round(cost, 6),
        }
        grand_in  += in_t
        grand_out += out_t

    grand_cost = grand_in * args.price_input_per_m / 1e6 + grand_out * args.price_output_per_m / 1e6

    # Sanity-check that all three systems hit the same datasets.
    datasets_seen = {tuple(per_system[k]["datasets"] or []) for k in per_system}
    same_datasets = (len(datasets_seen) == 1)

    out = {
        "schema": "aether-real-api-v1",
        "produced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "model": args.model,
        "pricing": {
            "input_per_million_usd": args.price_input_per_m,
            "output_per_million_usd": args.price_output_per_m,
            "source": "OpenAI public list price for " + args.model,
        },
        "budget_usd": args.budget,
        "estimated_cost_usd": args.estimated_cost_usd,
        "actual_cost_usd": round(grand_cost, 6),
        "actual_under_budget": grand_cost <= args.budget,
        "tokens_input_total":  grand_in,
        "tokens_output_total": grand_out,
        "tokens_total":        grand_in + grand_out,
        "all_systems_same_datasets": same_datasets,
        "per_system": per_system,
        "embedded_full_results": {
            "aether":    aether,
            "langchain": langchain,
            "dspy":      dspy,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Console summary -- also goes into the run log so reviewers can grep it.
    print("")
    print("=" * 64)
    print("Real-API benchmark — actual cost (from response usage)")
    print("=" * 64)
    print(f"  model:           {args.model}")
    print(f"  rates:           ${args.price_input_per_m}/1M in, "
          f"${args.price_output_per_m}/1M out")
    for k in ("aether", "langchain", "dspy"):
        s = per_system[k]
        print(f"  [{k:9}] tokens_in={s['tokens_input']:>9}  "
              f"tokens_out={s['tokens_output']:>9}  cost=${s['cost_usd']:.6f}")
    print(f"  {'TOTAL':>11}  tokens_in={grand_in:>9}  "
          f"tokens_out={grand_out:>9}  cost=${grand_cost:.6f}")
    print(f"  budget: ${args.budget:.2f}, estimated upfront: "
          f"${args.estimated_cost_usd:.4f}, actual: ${grand_cost:.6f}")
    if same_datasets:
        print(f"  datasets identical across systems: YES "
              f"({sorted(next(iter(datasets_seen)))})")
    else:
        print(f"  datasets identical across systems: NO  ({datasets_seen})")
    print(f"  wrote: {args.output}")
    if not out["actual_under_budget"]:
        print("WARNING: actual cost exceeds budget", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
