#!/usr/bin/env python3
"""Caching ablation runner.

Three modes per dataset, 5 trials each:

  no_cache        POST /cache/clear before every /execute. Hit rate is
                  guaranteed 0%; the per-request /cache/clear adds a small
                  (~ms) overhead documented in the JSON `notes`.
  l1_exact_match  POST /cache/clear once at trial start, single pass over the
                  dataset. This is the runtime's default L1 behavior. On a
                  unique-query dataset the hit rate is ~0% (L1 caching is a
                  no-op). On a repeat-injected dataset the hit rate climbs
                  with each repeat.
  repeat_warm     POST /cache/clear at trial start, run the dataset once as
                  warmup (latencies discarded, tokens recorded as
                  warmup_tokens_*), then run a second pass as the measured
                  trial. Hit rate ~100% on the measured pass.

Datasets:
  customer_support_100         100 unique queries
  customer_support_repeat_100  30 unique queries replayed ~3.33x (deterministic
                               seed=42; see bench/datasets/_make_repeat_dataset.py)

The output JSON adds a `cross_mode_deltas` field per dataset with paired-trial
bootstrap CIs on the cache_hit_rate / tokens_saved / latency_p50 deltas
between modes.

Reuses the request payload, helpers and JSON envelope conventions from
scripts/run_benchmark.py.

Usage (run from repo root):

    target/release/aether-runtime &              # listening on :3000
    python3 bench/runners/ablation_cache.py --trials 5
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
    create_triage_dag,
    execute_dag,
    get_cache_stats,
    get_git_version,
    get_hardware_info,
    launch_runtime,
    load_dataset,
)

DATASETS: list[tuple[str, str]] = [
    ("customer_support_100",        "bench/datasets/customer_support_100.jsonl"),
    ("customer_support_repeat_100", "bench/datasets/customer_support_repeat_100.jsonl"),
]
MODES: list[str] = ["no_cache", "l1_exact_match", "repeat_warm"]


def _execute_with_optional_clear(
    runtime_url: str,
    dag: dict,
    *,
    clear_before: bool,
) -> dict:
    """Execute one DAG; optionally clear the cache first.

    The pre-execute /cache/clear is the no_cache mode's enforcement mechanism
    (the runtime has no per-request cache-disable knob; see plan file).
    """
    if clear_before:
        clear_cache(runtime_url)
    return execute_dag(runtime_url, dag, sequential=False)


def _do_pass(
    runtime_url: str,
    dataset_items: list[dict],
    *,
    clear_each_request: bool,
) -> dict:
    """Run one full pass over the dataset; capture per-request latency, tokens
    and a cache-hits delta from /cache/stats.

    Returns the same per-trial dict shape that scripts/run_benchmark.py emits
    so the existing _bootstrap_ci / aggregate_trials helpers can be reused
    unmodified.
    """
    initial = get_cache_stats(runtime_url)

    latencies: list[float] = []
    tokens_total = 0
    tokens_input = 0
    tokens_output = 0
    tokens_saved = 0
    failed = 0
    successful = 0

    t0 = time.perf_counter()
    for item in dataset_items:
        try:
            dag = create_triage_dag(item["query"], item.get("context", {}))
            result = _execute_with_optional_clear(
                runtime_url, dag, clear_before=clear_each_request
            )
            latencies.append(
                result.get("total_execution_time_ms", result.get("client_latency_ms", 0))
            )
            tokens_total += int(result.get("total_token_cost", 0) or 0)
            tokens_saved += int(result.get("tokens_saved", 0) or 0)
            for nr in result.get("results") or []:
                tokens_input += int(nr.get("input_tokens", 0) or 0)
                tokens_output += int(nr.get("output_tokens", 0) or 0)
            successful += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  request failed: {exc}", file=sys.stderr)
            failed += 1
    total_ms = (time.perf_counter() - t0) * 1000.0

    final = get_cache_stats(runtime_url)
    hits = final.get("hits", 0) - initial.get("hits", 0)
    misses = final.get("misses", 0) - initial.get("misses", 0)
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

    p50, p95, p99 = _trial_percentiles(latencies) if latencies else (0.0, 0.0, 0.0)
    return {
        "latencies_ms": latencies,
        "p50": p50, "p95": p95, "p99": p99,
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
        # parallelization fields kept for envelope parity with run_benchmark.py output
        "parallelization_factor_mean": 0.0,
        "level_execution_times_ms_mean": [],
    }


def _run_trial(
    runtime_url: str,
    dataset_items: list[dict],
    mode: str,
    trial_idx: int,
) -> tuple[dict, dict]:
    """Run one trial in the given mode. Returns (measured_trial, warmup_summary).

    warmup_summary has keys tokens_input/tokens_output for the discarded
    warmup pass (only populated for repeat_warm; otherwise zeros).
    """
    warmup = {"tokens_input": 0, "tokens_output": 0}
    if mode == "no_cache":
        # The first request will MISS regardless; clearing per-request just
        # enforces the no-cache invariant.
        clear_cache(runtime_url)
        trial = _do_pass(runtime_url, dataset_items, clear_each_request=True)
    elif mode == "l1_exact_match":
        clear_cache(runtime_url)
        trial = _do_pass(runtime_url, dataset_items, clear_each_request=False)
    elif mode == "repeat_warm":
        clear_cache(runtime_url)
        warm = _do_pass(runtime_url, dataset_items, clear_each_request=False)
        warmup = {
            "tokens_input": warm["tokens_input"],
            "tokens_output": warm["tokens_output"],
        }
        trial = _do_pass(runtime_url, dataset_items, clear_each_request=False)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    trial["trial"] = trial_idx
    return trial, warmup


def _paired_bootstrap_delta(
    a: list[float], b: list[float], n_resamples: int = 10_000, seed: int = 42
) -> dict:
    """Paired-trial bootstrap CI for (a - b) where a[i], b[i] come from the
    same trial index. Used for cross-mode delta metrics."""
    if len(a) != len(b) or len(a) < 2:
        diff = float(np.mean(a) - np.mean(b)) if a and b else 0.0
        return {"mean": diff, "ci95": [diff, diff], "notes": "n<2; CI degenerate"}
    arr = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean = float(arr.mean())
    if float(arr.std()) == 0.0:
        return {"mean": mean, "ci95": [mean, mean], "notes": "constant per-trial; CI degenerate"}
    notes: Optional[str] = None
    try:
        res = scipy_stats.bootstrap(
            (arr,), np.mean,
            confidence_level=0.95,
            n_resamples=n_resamples,
            method="BCa",
            random_state=seed,
        )
        ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
    except Exception as exc:  # noqa: BLE001
        res = scipy_stats.bootstrap(
            (arr,), np.mean,
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


def _trial_for_json(t: dict) -> dict:
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
        "tokens_input": t["tokens_input"],
        "tokens_output": t["tokens_output"],
        "tokens_saved": t["tokens_saved"],
        "errors": t["errors"],
        "parallelization_factor_mean": round(t["parallelization_factor_mean"], 4),
        "level_execution_times_ms_mean": [round(x, 4) for x in t["level_execution_times_ms_mean"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aether caching ablation")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--output", default="bench/results/ablation_cache_v1.json")
    ap.add_argument("--runtime-url", default="http://127.0.0.1:3000")
    ap.add_argument("--no-autostart", action="store_true",
                    help="error out if runtime is not already on --runtime-url")
    args = ap.parse_args()

    spawned: Optional[subprocess.Popen] = None
    if check_runtime(args.runtime_url):
        print(f"reusing runtime at {args.runtime_url}", flush=True)
    elif args.no_autostart:
        print(f"error: runtime not available at {args.runtime_url} and --no-autostart was set",
              file=sys.stderr)
        return 1
    else:
        spawned = launch_runtime(REPO_ROOT, args.runtime_url)

    try:
        results: list[dict] = []
        for ds_name, rel_path in DATASETS:
            ds_path = REPO_ROOT / rel_path
            if not ds_path.exists():
                print(f"error: missing dataset {ds_path}", file=sys.stderr)
                return 1
            items = load_dataset(str(ds_path))
            print(f"\n[{ds_name}] {len(items)} items, "
                  f"{len({i['query'] for i in items})} distinct queries", flush=True)

            mode_trials: dict[str, list[dict]] = {m: [] for m in MODES}
            mode_warmup: dict[str, list[dict]] = {m: [] for m in MODES}

            for mode in MODES:
                print(f"  mode={mode}: running {args.trials} trials", flush=True)
                for trial_idx in range(args.trials):
                    trial, warm = _run_trial(args.runtime_url, items, mode, trial_idx)
                    mode_trials[mode].append(trial)
                    mode_warmup[mode].append(warm)
                    print(f"    trial {trial_idx}: p50={trial['p50']:.2f}ms "
                          f"hit_rate={trial['cache_hit_rate']:.4f} "
                          f"tokens_saved={trial['tokens_saved']}", flush=True)

            for mode in MODES:
                trials = mode_trials[mode]
                agg = aggregate_trials(trials)
                warmup_in = sum(w["tokens_input"] for w in mode_warmup[mode])
                warmup_out = sum(w["tokens_output"] for w in mode_warmup[mode])
                entry = {
                    "dataset": ds_name,
                    "config": mode,
                    "trials": args.trials,
                    **agg,
                    "warmup_tokens_input": warmup_in,
                    "warmup_tokens_output": warmup_out,
                    "raw_trial_results": [_trial_for_json(t) for t in trials],
                }
                if mode == "no_cache":
                    entry["notes"] = (
                        "POST /cache/clear before every /execute; small per-request "
                        "overhead included in latency."
                    )
                results.append(entry)

            no_cache_trials = mode_trials["no_cache"]
            l1_trials = mode_trials["l1_exact_match"]
            warm_trials = mode_trials["repeat_warm"]

            def _series(trials: list[dict], key: str) -> list[float]:
                return [float(t[key]) for t in trials]

            cross = {
                "dataset": ds_name,
                "cache_hit_rate_delta_l1_vs_no_cache": _paired_bootstrap_delta(
                    _series(l1_trials, "cache_hit_rate"),
                    _series(no_cache_trials, "cache_hit_rate"),
                ),
                "cache_hit_rate_delta_warm_vs_no_cache": _paired_bootstrap_delta(
                    _series(warm_trials, "cache_hit_rate"),
                    _series(no_cache_trials, "cache_hit_rate"),
                ),
                "tokens_saved_delta_l1_vs_no_cache": _paired_bootstrap_delta(
                    _series(l1_trials, "tokens_saved"),
                    _series(no_cache_trials, "tokens_saved"),
                ),
                "tokens_saved_delta_warm_vs_no_cache": _paired_bootstrap_delta(
                    _series(warm_trials, "tokens_saved"),
                    _series(no_cache_trials, "tokens_saved"),
                ),
                "latency_p50_delta_warm_vs_no_cache": _paired_bootstrap_delta(
                    _series(warm_trials, "p50"),
                    _series(no_cache_trials, "p50"),
                ),
            }
            results.append({"cross_mode_deltas": cross})

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
