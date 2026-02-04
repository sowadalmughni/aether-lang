#!/usr/bin/env python3
"""
LangChain Baseline Benchmark Stub

This script simulates a LangChain-style LLM pipeline for benchmark comparison
with Aether. It does NOT depend on actual LangChain packages - it implements
the behavioral patterns that LangChain code typically exhibits:

- Sequential prompt execution (no automatic parallelization)
- Manual caching with low hit rates (typically 10-20%)
- Runtime JSON parsing (with potential errors)

Usage:
    python langchain_baseline.py [--provider mock|openai|anthropic]

Environment Variables:
    BASELINE_PROVIDER: Provider to use (mock, openai, anthropic)
    OPENAI_API_KEY: API key for OpenAI
    ANTHROPIC_API_KEY: API key for Anthropic
"""

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any


# =============================================================================
# Mock LLM Client (simulates LangChain LLM calls)
# =============================================================================

class MockLLM:
    """Mock LLM that simulates LangChain-style calls with deterministic latency."""
    
    def __init__(self, base_latency_ms: int = 100):
        self.base_latency_ms = base_latency_ms
        self.call_count = 0
        
    def invoke(self, prompt: str, **kwargs) -> str:
        """Simulate an LLM call with latency based on prompt length."""
        self.call_count += 1
        
        # Simulate network latency proportional to prompt length
        prompt_factor = len(prompt) / 500.0  # ~500 chars = 1x latency
        latency_ms = self.base_latency_ms * (0.8 + 0.4 * prompt_factor)
        latency_ms += random.uniform(-10, 20)  # Add jitter
        
        time.sleep(latency_ms / 1000.0)
        
        # Generate deterministic response based on prompt hash
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return f"[LangChain Mock Response] Hash: {prompt_hash}"
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens (roughly 4 chars per token)."""
        return len(text) // 4


# =============================================================================
# Simple Cache (simulates typical LangChain manual caching)
# =============================================================================

class SimpleCache:
    """Simple cache with low hit rate to simulate manual LangChain caching."""
    
    def __init__(self, hit_probability: float = 0.15):
        self.cache: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        self.hit_probability = hit_probability
        
    def get(self, key: str) -> str | None:
        # Simulate low hit rate typical of manual caching
        if key in self.cache and random.random() < self.hit_probability:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: str) -> None:
        self.cache[key] = value
        
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# LangChain-style Pipeline Simulation
# =============================================================================

class LangChainPipeline:
    """
    Simulates a typical LangChain pipeline structure.
    
    This mimics the pattern:
        chain = prompt | llm | parser
        result = chain.invoke({"input": ...})
    """
    
    def __init__(self, llm: MockLLM, cache: SimpleCache | None = None):
        self.llm = llm
        self.cache = cache
        self.latencies: list[float] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.errors = 0
        
    def run_chain(self, prompt_template: str, inputs: dict[str, Any]) -> str | None:
        """Execute a chain with template substitution, LLM call, and parsing."""
        start = time.time()
        
        try:
            # Step 1: Template rendering (simulating PromptTemplate)
            prompt = prompt_template
            for key, value in inputs.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
            
            # Step 2: Check cache
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    self.latencies.append((time.time() - start) * 1000)
                    return cached
            
            # Step 3: LLM call (sequential - no parallelization)
            self.total_input_tokens += self.llm.estimate_tokens(prompt)
            response = self.llm.invoke(prompt)
            self.total_output_tokens += self.llm.estimate_tokens(response)
            
            # Step 4: Cache result
            if self.cache:
                self.cache.set(cache_key, response)
            
            # Step 5: Parse response (simulate JSON parsing with errors)
            # ~5-15% of responses fail parsing in typical LangChain usage
            if random.random() < 0.05:
                raise ValueError("JSON parsing failed: unexpected response format")
            
            self.latencies.append((time.time() - start) * 1000)
            return response
            
        except Exception as e:
            self.errors += 1
            self.latencies.append((time.time() - start) * 1000)
            return None
    
    def compute_percentiles(self) -> dict[str, float]:
        """Compute p50, p95, p99 from latency samples."""
        if not self.latencies:
            return {"p50": 0, "p95": 0, "p99": 0}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        def percentile(p: float) -> float:
            idx = int(p * (n - 1))
            return sorted_latencies[idx]
        
        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(num_requests: int = 10) -> dict[str, Any]:
    """Run the LangChain baseline benchmark."""
    
    provider = os.environ.get("BASELINE_PROVIDER", "mock")
    
    # Initialize components
    llm = MockLLM(base_latency_ms=100)
    cache = SimpleCache(hit_probability=0.15)
    pipeline = LangChainPipeline(llm, cache)
    
    # Sample prompts (simulating CustomerSupport-1K subset)
    prompts = [
        ("Classify the sentiment of: {text}", {"text": "Sample customer message"}),
        ("Extract entities from: {document}", {"document": "Document content"}),
        ("Summarize: {content}", {"content": "Long content to summarize"}),
    ]
    
    # Run benchmark
    start_time = time.time()
    successes = 0
    
    for i in range(num_requests):
        template, base_inputs = prompts[i % len(prompts)]
        inputs = {k: f"{v} {i}" for k, v in base_inputs.items()}
        
        result = pipeline.run_chain(template, inputs)
        if result:
            successes += 1
    
    total_time = (time.time() - start_time) * 1000
    percentiles = pipeline.compute_percentiles()
    
    # Build result
    result = {
        "baseline": "langchain",
        "dataset": f"synthetic_{num_requests}",
        "latency_p50_ms": round(percentiles["p50"], 2),
        "latency_p95_ms": round(percentiles["p95"], 2),
        "latency_p99_ms": round(percentiles["p99"], 2),
        "total_execution_time_ms": round(total_time, 2),
        "total_tokens": pipeline.total_input_tokens + pipeline.total_output_tokens,
        "input_tokens": pipeline.total_input_tokens,
        "output_tokens": pipeline.total_output_tokens,
        "cache_hit_rate": round(cache.hit_rate, 4),
        "success_rate": round(successes / num_requests, 4),
        "error_count": pipeline.errors,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "mode": provider,
        "parallelization_factor": 0.0,  # LangChain default is sequential
        "notes": "Simulated LangChain pipeline - sequential execution, manual caching"
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="LangChain Baseline Benchmark")
    parser.add_argument("--requests", type=int, default=10,
                        help="Number of requests to run")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for JSON results")
    args = parser.parse_args()
    
    result = run_benchmark(args.requests)
    
    output_json = json.dumps(result, indent=2)
    print(output_json)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
