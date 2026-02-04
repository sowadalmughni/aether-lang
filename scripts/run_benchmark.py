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
    
    data = json.dumps(dag).encode("utf-8")
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
                "node_type": "LlmFn",
                "model": "gpt-4o-mini",
                "prompt": f"Classify the urgency of this customer query as Low, Medium, High, or Critical.\n\nQuery: {query}\n\nCustomer tier: {context.get('customer_tier', 'free')}\n\nRespond with only the urgency level.",
                "dependencies": []
            },
            {
                "id": "classify_category",
                "node_type": "LlmFn",
                "model": "gpt-4o-mini",
                "prompt": f"Classify this customer query into a category (authentication, billing, bug, feature-request, how-to, outage, security, etc.).\n\nQuery: {query}\n\nRespond with only the category.",
                "dependencies": []
            },
            {
                "id": "generate_response",
                "node_type": "LlmFn",
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
                "node_type": "LlmFn",
                "model": "gpt-4o-mini",
                "prompt": f"Extract all named entities from this document. Return as a JSON array of strings.\n\nDocument: {document}",
                "dependencies": []
            },
            {
                "id": "summarize",
                "node_type": "LlmFn",
                "model": "gpt-4o-mini",
                "prompt": f"Summarize this document in 2-3 sentences.\n\nDocument: {document}",
                "dependencies": []
            },
            {
                "id": "classify_domain",
                "node_type": "LlmFn",
                "model": "gpt-4o-mini",
                "prompt": f"Classify the domain of this document (e.g., technology, finance, medical, legal, etc.).\n\nDocument: {document}\n\nRespond with only the domain.",
                "dependencies": []
            },
            {
                "id": "combine_results",
                "node_type": "Compute",
                "prompt": "Combine: entities={{{{node.extract_entities.output}}}}, summary={{{{node.summarize.output}}}}, domain={{{{node.classify_domain.output}}}}",
                "dependencies": ["extract_entities", "summarize", "classify_domain"]
            }
        ]
    }


def run_benchmark_scenario(
    runtime_url: str,
    scenario: str,
    dataset_path: str,
    num_requests: int,
    sequential: bool = False,
    cold_start: bool = False
) -> BenchmarkResult:
    """Run a single benchmark scenario."""
    
    # Load dataset
    data = load_dataset(dataset_path)
    if num_requests > len(data):
        # Cycle through dataset if needed
        data = (data * ((num_requests // len(data)) + 1))[:num_requests]
    else:
        data = data[:num_requests]
    
    # Clear cache for cold start
    if cold_start:
        clear_cache(runtime_url)
    
    # Get initial cache stats
    initial_stats = get_cache_stats(runtime_url)
    
    # Execute requests
    latencies = []
    tokens_total = 0
    tokens_saved = 0
    successful = 0
    failed = 0
    
    start_time = time.perf_counter()
    
    for item in data:
        try:
            if scenario == "triage":
                dag = create_triage_dag(item["query"], item.get("context", {}))
            elif scenario == "extraction":
                dag = create_extraction_dag(item["document"])
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
            
            result = execute_dag(runtime_url, dag, sequential=sequential)
            
            latencies.append(result.get("total_execution_time_ms", result.get("client_latency_ms", 0)))
            tokens_total += result.get("total_tokens", 0)
            tokens_saved += result.get("tokens_saved", 0)
            successful += 1
            
        except Exception as e:
            print(f"Request failed: {e}", file=sys.stderr)
            failed += 1
    
    total_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Get final cache stats
    final_stats = get_cache_stats(runtime_url)
    cache_hits = final_stats.get("hits", 0) - initial_stats.get("hits", 0)
    cache_misses = final_stats.get("misses", 0) - initial_stats.get("misses", 0)
    cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
    
    # Compute percentiles
    if latencies:
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p50 = latencies_sorted[int(0.50 * (n - 1))] if n > 1 else latencies_sorted[0]
        p95 = latencies_sorted[int(0.95 * (n - 1))] if n > 1 else latencies_sorted[0]
        p99 = latencies_sorted[int(0.99 * (n - 1))] if n > 1 else latencies_sorted[0]
    else:
        p50 = p95 = p99 = 0.0
    
    # Determine mode
    if cold_start:
        mode = "cold"
    elif sequential:
        mode = "sequential"
    else:
        mode = "parallel"
    
    # Parallelization factor (3 parallel nodes in extraction, 2 in triage)
    expected_parallel = 3 if scenario == "extraction" else 2
    parallelization_factor = expected_parallel if not sequential else 1.0
    
    return BenchmarkResult(
        scenario=scenario,
        mode=mode,
        provider=os.environ.get("AETHER_PROVIDER", "mock"),
        requests=num_requests,
        successful=successful,
        failed=failed,
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        latency_p99_ms=round(p99, 2),
        total_time_ms=round(total_time_ms, 2),
        tokens_total=tokens_total,
        tokens_saved=tokens_saved,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_hit_rate=round(cache_hit_rate, 4),
        parallelization_factor=parallelization_factor,
        measured_at=datetime.utcnow().isoformat() + "Z",
        runtime_url=runtime_url,
        dataset=dataset_path,
        notes=None
    )


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
    
    args = parser.parse_args()
    
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
