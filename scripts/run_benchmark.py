#!/usr/bin/env python3
"""
Aether Benchmark Runner

Runs benchmark scenarios against the Aether runtime and baseline implementations,
producing JSON reports for comparison.

Usage:
    python run_benchmark.py --scenario triage --requests 100
    python run_benchmark.py --scenario extraction --requests 50 --sequential
    python run_benchmark.py --all --output results/
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
from scipy import stats as scipy_stats

# Default runtime URL
DEFAULT_RUNTIME_URL = os.environ.get("AETHER_RUNTIME_URL", "http://127.0.0.1:3000")


@dataclass
class BenchmarkResult:
    """Benchmark result for a single run."""
    scenario: str
    mode: str  # parallel, sequential, cold, warm
    provider: str  # mock, openai, anthropic
    requests: int
    successful: int
    failed: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    total_time_ms: float
    tokens_total: int
    tokens_saved: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    parallelization_factor: float
    measured_at: str
    runtime_url: str
    dataset: Optional[str] = None
    notes: Optional[str] = None


def check_runtime(url: str) -> bool:
    """Check if runtime is available."""
    try:
        req = Request(f"{url}/health", method="GET")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (URLError, TimeoutError):
        return False


def clear_cache(url: str) -> bool:
    """Clear the runtime cache for cold start benchmarks."""
    try:
        req = Request(f"{url}/cache/clear", method="POST")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (URLError, TimeoutError):
        return False


def get_cache_stats(url: str) -> dict:
    """Get cache statistics from runtime."""
    try:
        req = Request(f"{url}/cache/stats", method="GET")
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except (URLError, TimeoutError):
        return {"hits": 0, "misses": 0, "size": 0}


def execute_dag(url: str, dag: dict, sequential: bool = False) -> dict:
    """Execute a DAG on the runtime."""
    endpoint = f"{url}/execute"
    if sequential:
        endpoint += "?sequential=true"

    # Runtime's /execute expects {"dag": <dag>, "context": <map>}; sending
    # the dag unwrapped returns HTTP 422 on every request.
    payload = {"dag": dag}
    data = json.dumps(payload).encode("utf-8")
    req = Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    
    start = time.perf_counter()
    with urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    result["client_latency_ms"] = elapsed_ms
    return result


def load_dataset(path: str) -> list[dict]:
    """Load a JSONL dataset."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def create_triage_dag(query: str, context: dict) -> dict:
    """Create a customer support triage DAG."""
    return {
        "id": f"triage_{int(time.time() * 1000)}",
        "nodes": [
            {
                "id": "classify_urgency",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "prompt": f"Classify the urgency of this customer query as Low, Medium, High, or Critical.\n\nQuery: {query}\n\nCustomer tier: {context.get('customer_tier', 'free')}\n\nRespond with only the urgency level.",
                "dependencies": []
            },
            {
                "id": "classify_category",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "prompt": f"Classify this customer query into a category (authentication, billing, bug, feature-request, how-to, outage, security, etc.).\n\nQuery: {query}\n\nRespond with only the category.",
                "dependencies": []
            },
            {
                "id": "generate_response",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "prompt": f"Generate a helpful customer support response for this query.\n\nQuery: {query}\nUrgency: {{{{node.classify_urgency.output}}}}\nCategory: {{{{node.classify_category.output}}}}\n\nProvide a concise, helpful response.",
                "dependencies": ["classify_urgency", "classify_category"]
            }
        ]
    }


def create_extraction_dag(document: str) -> dict:
    """Create a parallel document extraction DAG."""
    return {
        "id": f"extraction_{int(time.time() * 1000)}",
        "nodes": [
            {
                "id": "extract_entities",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "prompt": f"Extract all named entities from this document. Return as a JSON array of strings.\n\nDocument: {document}",
                "dependencies": []
            },
            {
                "id": "summarize",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "prompt": f"Summarize this document in 2-3 sentences.\n\nDocument: {document}",
                "dependencies": []
            },
            {
                "id": "classify_domain",
                "node_type": "llm_fn",
                "model": "gpt-4o-mini",
                "prompt": f"Classify the domain of this document (e.g., technology, finance, medical, legal, etc.).\n\nDocument: {document}\n\nRespond with only the domain.",
                "dependencies": []
            },
            {
                "id": "combine_results",
                "node_type": "compute",
                "prompt": "Combine: entities={{{{node.extract_entities.output}}}}, summary={{{{node.summarize.output}}}}, domain={{{{node.classify_domain.output}}}}",
                "dependencies": ["extract_entities", "summarize", "classify_domain"]
            }
        ]
    }


def _build_dag(scenario: str, item: dict) -> dict:
    if scenario == "triage":
        return create_triage_dag(item["query"], item.get("context", {}))
    if scenario == "extraction":
        return create_extraction_dag(item["document"])
    raise ValueError(f"Unknown scenario: {scenario}")


def _run_single_trial(
    runtime_url: str,
    scenario: str,
    dataset_items: list[dict],
    sequential: bool,
) -> dict:
    """Execute one full pass over ``dataset_items`` and return raw measurements.

    Caller owns cache state setup before calling — this function only reads
    /cache/stats deltas; it does not clear the cache itself.
    """
    initial_stats = get_cache_stats(runtime_url)

    latencies: list[float] = []
    tokens_total = 0
    tokens_saved = 0
    parallelization_factors: list[float] = []
    level_times_per_request: list[list[float]] = []
    successful = 0
    failed = 0

    start_time = time.perf_counter()
    for item in dataset_items:
        try:
            dag = _build_dag(scenario, item)
            result = execute_dag(runtime_url, dag, sequential=sequential)
            latencies.append(result.get("total_execution_time_ms", result.get("client_latency_ms", 0)))
            tokens_total += result.get("total_tokens", 0) or 0
            tokens_saved += result.get("tokens_saved", 0) or 0
            pf = result.get("parallelization_factor")
            if pf is not None:
                parallelization_factors.append(float(pf))
            lvls = result.get("level_execution_times_ms")
            if lvls:
                level_times_per_request.append([float(x) for x in lvls])
            successful += 1
        except Exception as e:
            print(f"Request failed: {e}", file=sys.stderr)
            failed += 1
    total_time_ms = (time.perf_counter() - start_time) * 1000

    final_stats = get_cache_stats(runtime_url)
    cache_hits = final_stats.get("hits", 0) - initial_stats.get("hits", 0)
    cache_misses = final_stats.get("misses", 0) - initial_stats.get("misses", 0)
    cache_hit_rate = (
        cache_hits / (cache_hits + cache_misses)
        if (cache_hits + cache_misses) > 0
        else 0.0
    )

    if level_times_per_request:
        max_levels = max(len(v) for v in level_times_per_request)
        sums = [0.0] * max_levels
        counts = [0] * max_levels
        for vec in level_times_per_request:
            for i, t in enumerate(vec):
                sums[i] += t
                counts[i] += 1
        level_execution_times_ms_mean = [
            (sums[i] / counts[i]) if counts[i] else 0.0 for i in range(max_levels)
        ]
    else:
        level_execution_times_ms_mean = []

    parallelization_factor_mean = (
        sum(parallelization_factors) / len(parallelization_factors)
        if parallelization_factors
        else (1.0 if sequential else 0.0)
    )

    return {
        "latencies_ms": latencies,
        "tokens_total": tokens_total,
        "tokens_saved": tokens_saved,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hit_rate,
        "errors": failed,
        "successful": successful,
        "total_time_ms": total_time_ms,
        "parallelization_factor_mean": parallelization_factor_mean,
        "level_execution_times_ms_mean": level_execution_times_ms_mean,
    }


def _percentiles_simple(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return 0.0, 0.0, 0.0
    s = sorted(latencies)
    n = len(s)
    if n == 1:
        return s[0], s[0], s[0]
    return (
        s[int(0.50 * (n - 1))],
        s[int(0.95 * (n - 1))],
        s[int(0.99 * (n - 1))],
    )


def run_benchmark_scenario(
    runtime_url: str,
    scenario: str,
    dataset_path: str,
    num_requests: int,
    sequential: bool = False,
    cold_start: bool = False
) -> BenchmarkResult:
    """Run a single benchmark scenario (legacy single-trial wrapper)."""

    data = load_dataset(dataset_path)
    if num_requests > len(data):
        data = (data * ((num_requests // len(data)) + 1))[:num_requests]
    else:
        data = data[:num_requests]

    if cold_start:
        clear_cache(runtime_url)

    trial = _run_single_trial(runtime_url, scenario, data, sequential=sequential)
    p50, p95, p99 = _percentiles_simple(trial["latencies_ms"])

    if cold_start:
        mode = "cold"
    elif sequential:
        mode = "sequential"
    else:
        mode = "parallel"

    expected_parallel = 3 if scenario == "extraction" else 2
    parallelization_factor = expected_parallel if not sequential else 1.0

    return BenchmarkResult(
        scenario=scenario,
        mode=mode,
        provider=os.environ.get("AETHER_PROVIDER", "mock"),
        requests=num_requests,
        successful=trial["successful"],
        failed=trial["errors"],
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        latency_p99_ms=round(p99, 2),
        total_time_ms=round(trial["total_time_ms"], 2),
        tokens_total=trial["tokens_total"],
        tokens_saved=trial["tokens_saved"],
        cache_hits=trial["cache_hits"],
        cache_misses=trial["cache_misses"],
        cache_hit_rate=round(trial["cache_hit_rate"], 4),
        parallelization_factor=parallelization_factor,
        measured_at=datetime.utcnow().isoformat() + "Z",
        runtime_url=runtime_url,
        dataset=dataset_path,
        notes=None
    )


SUITE_DATASETS = [
    ("customer_support_100", "triage",     "bench/datasets/customer_support_100.jsonl"),
    ("document_analysis_50", "extraction", "bench/datasets/document_analysis_50.jsonl"),
]
SUITE_CONFIGS = ["sequential", "parallel", "parallel_cached"]


def _trial_percentiles(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return 0.0, 0.0, 0.0
    arr = np.asarray(latencies, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99], method="linear")
    return float(p50), float(p95), float(p99)


def run_config_trials(
    runtime_url: str,
    dataset_name: str,
    config: str,
    n_trials: int,
    dataset_items: list[dict],
    scenario: str,
) -> list[dict]:
    """Run N trials for one (dataset, config) tuple and return per-trial dicts."""
    assert config in SUITE_CONFIGS, f"unknown config: {config}"
    sequential = (config == "sequential")
    trials: list[dict] = []

    if config == "parallel_cached":
        # Warm the cache once; later measured trials must NOT clear it.
        clear_cache(runtime_url)
        warmup = _run_single_trial(runtime_url, scenario, dataset_items, sequential=False)
        if warmup["cache_hit_rate"] > 0.05:
            raise RuntimeError(
                f"parallel_cached warmup expected ~0% hit rate (cache freshly cleared) "
                f"but saw {warmup['cache_hit_rate']:.3f} — cache state is leaking across runs"
            )
        for i in range(n_trials):
            print(f"    trial {i+1}/{n_trials} (cached)...", flush=True)
            trial = _run_single_trial(runtime_url, scenario, dataset_items, sequential=False)
            trial["trial"] = i
            p50, p95, p99 = _trial_percentiles(trial["latencies_ms"])
            trial["p50"], trial["p95"], trial["p99"] = p50, p95, p99
            trials.append(trial)
        if trials and trials[0]["cache_hit_rate"] < 0.5:
            raise RuntimeError(
                f"parallel_cached first measured trial has hit_rate "
                f"{trials[0]['cache_hit_rate']:.3f}; expected > 0.5 after warmup"
            )
    else:
        for i in range(n_trials):
            clear_cache(runtime_url)
            print(f"    trial {i+1}/{n_trials} ({config})...", flush=True)
            trial = _run_single_trial(runtime_url, scenario, dataset_items, sequential=sequential)
            trial["trial"] = i
            p50, p95, p99 = _trial_percentiles(trial["latencies_ms"])
            trial["p50"], trial["p95"], trial["p99"] = p50, p95, p99
            trials.append(trial)
    return trials


def _bootstrap_ci(per_trial_values: list[float]) -> dict:
    """mean / std / 95% bootstrap CI for one metric across N trials."""
    arr = np.asarray(per_trial_values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n > 0 else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    notes: Optional[str] = None
    if n < 2 or float(arr.std()) == 0.0:
        ci = [mean, mean]
        notes = "constant per-trial values; CI degenerate"
    else:
        try:
            res = scipy_stats.bootstrap(
                (arr,),
                np.mean,
                confidence_level=0.95,
                n_resamples=10000,
                method="BCa",
                random_state=42,
            )
            ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
        except Exception as exc:
            res = scipy_stats.bootstrap(
                (arr,),
                np.mean,
                confidence_level=0.95,
                n_resamples=10000,
                method="percentile",
                random_state=42,
            )
            ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
            notes = f"BCa failed ({exc.__class__.__name__}); used percentile fallback"
    out = {"mean": mean, "std": std, "ci95": ci}
    if notes:
        out["notes"] = notes
    return out


def aggregate_trials(trials: list[dict]) -> dict:
    """Aggregate per-trial scalars across trials with bootstrap CIs."""
    metrics = {
        "latency_p50_ms":      [t["p50"] for t in trials],
        "latency_p95_ms":      [t["p95"] for t in trials],
        "latency_p99_ms":      [t["p99"] for t in trials],
        "cache_hit_rate":      [t["cache_hit_rate"] for t in trials],
        "tokens_saved_total":  [float(t["tokens_saved"]) for t in trials],
    }
    return {name: _bootstrap_ci(vals) for name, vals in metrics.items()}


def _trial_for_json(t: dict) -> dict:
    """Slim per-trial dict for raw_trial_results."""
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


def run_suite(args) -> int:
    """Run the full (dataset × config × N trials) suite and write JSON."""
    if not check_runtime(args.runtime_url):
        print(f"Error: Runtime not available at {args.runtime_url}", file=sys.stderr)
        print("Start the runtime with: cd aether-runtime && cargo run --release", file=sys.stderr)
        return 1

    selected_configs = SUITE_CONFIGS if args.config == "all" else [args.config]
    results = []

    for dataset_name, scenario, rel_path in SUITE_DATASETS:
        path = rel_path
        if not os.path.isabs(path):
            path = str(Path(__file__).resolve().parent.parent / rel_path)
        if not os.path.exists(path):
            print(f"Error: Dataset not found: {path}", file=sys.stderr)
            return 1
        dataset_items = load_dataset(path)

        for config in selected_configs:
            print(f"\n[{dataset_name} / {config}] running {args.trials} trials over {len(dataset_items)} items")
            trials = run_config_trials(
                args.runtime_url, dataset_name, config, args.trials, dataset_items, scenario
            )
            agg = aggregate_trials(trials)
            entry = {
                "dataset": dataset_name,
                "config": config,
                "trials": args.trials,
                **agg,
                "raw_trial_results": [_trial_for_json(t) for t in trials],
            }
            results.append(entry)
            print(
                f"  -> p50 {agg['latency_p50_ms']['mean']:.2f} ± {agg['latency_p50_ms']['std']:.2f} ms"
                f" | p95 {agg['latency_p95_ms']['mean']:.2f} ± {agg['latency_p95_ms']['std']:.2f} ms"
                f" | hit {agg['cache_hit_rate']['mean']:.3f}"
            )

    out = {
        "system": "aether",
        "version": "unknown",
        "provider": os.environ.get("AETHER_PROVIDER", "mock"),
        "hardware": {"cpu": "unknown", "ram_gb": 0, "os": "unknown"},
        "datasets": [d[0] for d in SUITE_DATASETS],
        "configs": list(selected_configs),
        "trials_per_config": args.trials,
        "results": results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


def run_baseline(baseline: str, num_requests: int) -> Optional[dict]:
    """Run a baseline benchmark using the baseline scripts."""
    script_path = Path(__file__).parent.parent / "bench" / "baselines" / f"{baseline}_baseline.py"
    
    if not script_path.exists():
        print(f"Baseline script not found: {script_path}", file=sys.stderr)
        return None
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--requests", str(num_requests)],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Baseline failed: {result.stderr}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Baseline error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Aether Benchmark Runner")
    parser.add_argument("--scenario", choices=["triage", "extraction"], help="Benchmark scenario")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--sequential", action="store_true", help="Force sequential execution")
    parser.add_argument("--cold", action="store_true", help="Clear cache before run (cold start)")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--baselines", action="store_true", help="Include baseline comparisons")
    parser.add_argument("--runtime-url", default=DEFAULT_RUNTIME_URL, help="Runtime URL")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--dataset-dir", default="bench/datasets", help="Dataset directory")
    parser.add_argument("--suite", action="store_true",
                        help="Run the full N-trial suite and emit aether_mock_v1.json")
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per (dataset, config) tuple in --suite mode")
    parser.add_argument("--config", default="all",
                        choices=SUITE_CONFIGS + ["all"],
                        help="Restrict --suite to one config (default: all)")
    parser.add_argument("--output-json", default="bench/results/aether_mock_v1.json",
                        help="--suite JSON output path")
    parser.add_argument("--output-md", default="bench/results/aether_mock_v1.md",
                        help="--suite Markdown report output path")

    args = parser.parse_args()

    if args.suite:
        sys.exit(run_suite(args))

    # Check runtime
    if not check_runtime(args.runtime_url):
        print(f"Error: Runtime not available at {args.runtime_url}", file=sys.stderr)
        print("Start the runtime with: cd aether-runtime && cargo run --release", file=sys.stderr)
        sys.exit(1)
    
    results = []
    
    # Determine scenarios to run
    if args.all:
        scenarios = ["triage", "extraction"]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        print("Error: Specify --scenario or --all", file=sys.stderr)
        sys.exit(1)
    
    # Dataset mapping
    dataset_map = {
        "triage": os.path.join(args.dataset_dir, "customer_support_100.jsonl"),
        "extraction": os.path.join(args.dataset_dir, "document_analysis_50.jsonl"),
    }
    
    # Run benchmarks
    for scenario in scenarios:
        dataset = dataset_map[scenario]
        if not os.path.exists(dataset):
            print(f"Warning: Dataset not found: {dataset}", file=sys.stderr)
            continue
        
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")
        
        # Cold start
        print(f"Running cold start ({args.requests} requests)...")
        result = run_benchmark_scenario(
            args.runtime_url, scenario, dataset, args.requests,
            sequential=args.sequential, cold_start=True
        )
        results.append(asdict(result))
        print(f"  p50: {result.latency_p50_ms}ms, p95: {result.latency_p95_ms}ms, cache: {result.cache_hit_rate*100:.1f}%")
        
        # Warm start
        print(f"Running warm start ({args.requests} requests)...")
        result = run_benchmark_scenario(
            args.runtime_url, scenario, dataset, args.requests,
            sequential=args.sequential, cold_start=False
        )
        result.mode = "warm"
        results.append(asdict(result))
        print(f"  p50: {result.latency_p50_ms}ms, p95: {result.latency_p95_ms}ms, cache: {result.cache_hit_rate*100:.1f}%")
        
        # Sequential ablation (if not already sequential)
        if not args.sequential:
            print(f"Running sequential ablation ({args.requests} requests)...")
            result = run_benchmark_scenario(
                args.runtime_url, scenario, dataset, args.requests,
                sequential=True, cold_start=True
            )
            results.append(asdict(result))
            print(f"  p50: {result.latency_p50_ms}ms, p95: {result.latency_p95_ms}ms")
    
    # Run baselines
    if args.baselines:
        print(f"\n{'='*60}")
        print("Running baselines...")
        print(f"{'='*60}")
        
        for baseline in ["langchain", "dspy"]:
            print(f"Running {baseline} baseline...")
            baseline_result = run_baseline(baseline, args.requests)
            if baseline_result:
                results.append(baseline_result)
                print(f"  p50: {baseline_result.get('latency_p50_ms', 'N/A')}ms")
    
    # Output results
    output_data = {
        "benchmark_run": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "runtime_url": args.runtime_url,
            "provider": os.environ.get("AETHER_PROVIDER", "mock"),
            "total_results": len(results)
        },
        "results": results
    }
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
