#!/usr/bin/env python3
"""Parallelization ablation runner.

Two modes per dataset, 5 trials each:

  sequential  POST /execute?sequential=true. Runtime executes DAG nodes in
              declared order, no parallelism (parallelization_factor == 1.0).
  parallel    POST /execute (default). Runtime parallelizes across DAG levels
              that have no data dependencies.

Datasets:
  customer_support_100   2-way fan-out (urgency || category -> response)
  document_analysis_50   3-way fan-out (entities || summary || domain -> combine)

Cache is cleared before every trial in both modes so the parallelization signal
is not confounded with caching effects.

Per-trial captured: latencies_ms, p50/p95/p99, parallelization_factor (read
from the runtime response field of the same name -- single definition across
both datasets and both modes; see the plan file for rationale),
max_concurrency_used, level_execution_times_ms, errors.

Aggregates per (dataset, mode): latency_p{50,95,99} and parallelization_factor
each with mean / std / 95% bootstrap CI.

Speedup metric (per dataset): paired-trial bootstrap CI on the ratio
sequential.p50 / parallel.p50 across the 5 trial pairs.

Output: bench/results/ablation_parallel_v1.json.

Usage (run from repo root):

    target/release/aether-runtime &
    python3 bench/runners/ablation_parallel.py --trials 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_benchmark import (  # noqa: E402
    _bootstrap_ci,
    _trial_percentiles,
    aggregate_trials,
    check_runtime,
    clear_cache,
    create_extraction_dag,
    create_triage_dag,
    execute_dag,
    get_cache_stats,
    get_git_version,
    get_hardware_info,
    launch_runtime,
    load_dataset,
)

# (dataset_name, scenario, jsonl_path)
DATASETS: list[tuple[str, str, str]] = [
    ("customer_support_100", "triage",     "bench/datasets/customer_support_100.jsonl"),
    ("document_analysis_50", "extraction", "bench/datasets/document_analysis_50.jsonl"),
]
MODES: list[str] = ["sequential", "parallel"]


def _build_dag(scenario: str, item: dict) -> dict:
    if scenario == "triage":
        return create_triage_dag(item["query"], item.get("context", {}))
    if scenario == "extraction":
        return create_extraction_dag(item["document"])
    raise ValueError(f"unknown scenario {scenario!r}")


def _run_trial(
    runtime_url: str,
    scenario: str,
    items: list[dict],
    sequential: bool,
    trial_idx: int,
) -> dict:
    """Single-pass trial. Cache cleared at start so caching doesn't confound
    parallelization measurements."""
    clear_cache(runtime_url)
    initial = get_cache_stats(runtime_url)

    latencies: list[float] = []
    pfactors: list[float] = []
    max_concurrency: list[int] = []
    level_times: list[list[float]] = []
    tokens_total = 0
    tokens_input = 0
    tokens_output = 0
    tokens_saved = 0
    successful = 0
    failed = 0

    t0 = time.perf_counter()
    for item in items:
        try:
            dag = _build_dag(scenario, item)
            result = execute_dag(runtime_url, dag, sequential=sequential)
            latencies.append(
                result.get("total_execution_time_ms", result.get("client_latency_ms", 0))
            )
            tokens_total += int(result.get("total_token_cost", 0) or 0)
            tokens_saved += int(result.get("tokens_saved", 0) or 0)
            for nr in result.get("results") or []:
                tokens_input += int(nr.get("input_tokens", 0) or 0)
                tokens_output += int(nr.get("output_tokens", 0) or 0)
            pf = result.get("parallelization_factor")
            if pf is not None:
                pfactors.append(float(pf))
            mc = result.get("max_concurrency_used")
            if mc is not None:
                max_concurrency.append(int(mc))
            lvls = result.get("level_execution_times_ms")
            if lvls:
                level_times.append([float(x) for x in lvls])
            successful += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  request failed: {exc}", file=sys.stderr)
            failed += 1
    total_ms = (time.perf_counter() - t0) * 1000.0

    final = get_cache_stats(runtime_url)
    hits = final.get("hits", 0) - initial.get("hits", 0)
    misses = final.get("misses", 0) - initial.get("misses", 0)
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

    if level_times:
        max_levels = max(len(v) for v in level_times)
        sums = [0.0] * max_levels
        counts = [0] * max_levels
        for vec in level_times:
            for i, t in enumerate(vec):
                sums[i] += t
                counts[i] += 1
        level_mean = [(sums[i] / counts[i]) if counts[i] else 0.0 for i in range(max_levels)]
    else:
        level_mean = []

    pfactor_mean = sum(pfactors) / len(pfactors) if pfactors else (1.0 if sequential else 0.0)
    mc_mean = sum(max_concurrency) / len(max_concurrency) if max_concurrency else (
        1.0 if sequential else 0.0
    )

    p50, p95, p99 = _trial_percentiles(latencies) if latencies else (0.0, 0.0, 0.0)
    return {
        "trial": trial_idx,
        "latencies_ms": latencies,
        "p50": p50, "p95": p95, "p99": p99,
        "parallelization_factor_mean": pfactor_mean,
        "max_concurrency_used_mean": mc_mean,
        "level_execution_times_ms_mean": level_mean,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": hit_rate,
        "tokens_total": tokens_total,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_saved": tokens_saved,
        "errors": failed,
        "successful": successful,
        "total_time_ms": total_ms,
    }


def _paired_speedup_bootstrap(
    seq_p50: list[float], par_p50: list[float], n_resamples: int = 10_000, seed: int = 42,
) -> dict:
    """Paired-trial bootstrap CI on the ratio sequential.p50 / parallel.p50.

    Pairs by trial index. Speedup > 1 means parallel is faster. Per-trial
    ratios -> mean + 95% bootstrap CI."""
    if len(seq_p50) != len(par_p50) or len(seq_p50) < 2:
        m = (sum(seq_p50) / sum(par_p50)) if (par_p50 and sum(par_p50)) else 0.0
        return {"mean": m, "ci95": [m, m], "notes": "n<2; CI degenerate"}
    ratios = np.asarray(seq_p50, dtype=float) / np.asarray(par_p50, dtype=float)
    mean = float(ratios.mean())
    if float(ratios.std()) == 0.0:
        return {"mean": mean, "ci95": [mean, mean], "notes": "constant per-trial; CI degenerate"}
    notes: Optional[str] = None
    try:
        res = scipy_stats.bootstrap(
            (ratios,), np.mean,
            confidence_level=0.95,
            n_resamples=n_resamples,
            method="BCa",
            random_state=seed,
        )
        ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
    except Exception as exc:  # noqa: BLE001
        res = scipy_stats.bootstrap(
            (ratios,), np.mean,
            confidence_level=0.95,
            n_resamples=n_resamples,
            method="percentile",
            random_state=seed,
        )
        ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
        notes = f"BCa failed ({exc.__class__.__name__}); used percentile fallback"
    out = {"mean": mean, "ci95": ci}
    if notes:
        out["notes"] = notes
    return out


def _agg_with_pfactor(trials: list[dict]) -> dict:
    """aggregate_trials() from run_benchmark.py covers latency + cache + tokens.
    Add parallelization_factor on top."""
    base = aggregate_trials(trials)
    pf_vals = [float(t["parallelization_factor_mean"]) for t in trials]
    base["parallelization_factor"] = _bootstrap_ci(pf_vals)
    return base


def _trial_for_json(t: dict) -> dict:
    return {
        "trial": t["trial"],
        "latencies_ms": [round(x, 4) for x in t["latencies_ms"]],
        "p50": round(t["p50"], 4),
        "p95": round(t["p95"], 4),
        "p99": round(t["p99"], 4),
        "parallelization_factor_mean": round(t["parallelization_factor_mean"], 4),
        "max_concurrency_used_mean": round(t["max_concurrency_used_mean"], 4),
        "level_execution_times_ms_mean": [round(x, 4) for x in t["level_execution_times_ms_mean"]],
        "cache_hit_rate": round(t["cache_hit_rate"], 6),
        "cache_hits": t["cache_hits"],
        "cache_misses": t["cache_misses"],
        "tokens_total": t["tokens_total"],
        "tokens_input": t["tokens_input"],
        "tokens_output": t["tokens_output"],
        "tokens_saved": t["tokens_saved"],
        "errors": t["errors"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aether parallelization ablation")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--output", default="bench/results/ablation_parallel_v1.json")
    ap.add_argument("--runtime-url", default="http://127.0.0.1:3000")
    ap.add_argument("--no-autostart", action="store_true")
    args = ap.parse_args()

    spawned: Optional[subprocess.Popen] = None
    if check_runtime(args.runtime_url):
        print(f"reusing runtime at {args.runtime_url}", flush=True)
    elif args.no_autostart:
        print(f"error: runtime not available at {args.runtime_url}", file=sys.stderr)
        return 1
    else:
        spawned = launch_runtime(REPO_ROOT, args.runtime_url)

    try:
        results: list[dict] = []
        for ds_name, scenario, rel_path in DATASETS:
            ds_path = REPO_ROOT / rel_path
            if not ds_path.exists():
                print(f"error: missing dataset {ds_path}", file=sys.stderr)
                return 1
            items = load_dataset(str(ds_path))
            print(f"\n[{ds_name} ({scenario})] {len(items)} items", flush=True)

            mode_trials: dict[str, list[dict]] = {m: [] for m in MODES}
            for mode in MODES:
                seq = (mode == "sequential")
                print(f"  mode={mode}: running {args.trials} trials", flush=True)
                for trial_idx in range(args.trials):
                    t = _run_trial(args.runtime_url, scenario, items, seq, trial_idx)
                    mode_trials[mode].append(t)
                    print(f"    trial {trial_idx}: p50={t['p50']:.2f}ms "
                          f"pfactor={t['parallelization_factor_mean']:.3f} "
                          f"max_conc={t['max_concurrency_used_mean']:.2f}", flush=True)

            for mode in MODES:
                trials = mode_trials[mode]
                agg = _agg_with_pfactor(trials)
                entry = {
                    "dataset": ds_name,
                    "config": mode,
                    "trials": args.trials,
                    **agg,
                    "warmup_tokens_input": 0,
                    "warmup_tokens_output": 0,
                    "raw_trial_results": [_trial_for_json(t) for t in trials],
                }
                results.append(entry)

            seq_p50 = [t["p50"] for t in mode_trials["sequential"]]
            par_p50 = [t["p50"] for t in mode_trials["parallel"]]
            results.append({
                "speedup": {
                    "dataset": ds_name,
                    "speedup_p50": _paired_speedup_bootstrap(seq_p50, par_p50),
                    "definition": (
                        "ratio sequential.p50 / parallel.p50 per paired trial; "
                        "BCa bootstrap, n_resamples=10000, seed=42; "
                        "parallelization_factor read from runtime response field of "
                        "the same name (sum(node_execution_times_ms) / "
                        "total_execution_time_ms)"
                    ),
                },
            })

        envelope = {
            "system": "aether",
            "version": get_git_version(REPO_ROOT),
            "provider": os.environ.get("AETHER_PROVIDER", "mock"),
            "hardware": get_hardware_info(),
            "datasets": [d[0] for d in DATASETS],
            "configs": MODES,
            "trials_per_config": args.trials,
            "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "results": results,
        }
        out_path = REPO_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")
        return 0
    finally:
        if spawned is not None:
            try:
                spawned.terminate()
                spawned.wait(timeout=10)
            except Exception:  # noqa: BLE001
                spawned.kill()


if __name__ == "__main__":
    raise SystemExit(main())
