# Baseline Benchmark Stubs

This directory contains baseline benchmark scripts for comparing Aether runtime
performance against traditional LLM orchestration approaches.

## Overview

These scripts are **scaffolding baselines** - they implement the same benchmark
report schema as Aether benchmarks to enable direct comparison. In mock mode,
they use deterministic simulated responses instead of real LLM API calls.

## Scripts

### langchain_baseline.py

Simulates a LangChain-style pipeline with:
- Sequential prompt execution
- Manual caching (opt-in, low hit rate)
- JSON parsing at runtime (error-prone)

### dspy_baseline.py

Simulates a DSPy-style pipeline with:
- Program-based composition
- Automatic prompt optimization (simulated)
- Module-based structure

## Usage

### Mock Mode (default - no API keys required)

```bash
# Run LangChain baseline
python bench/baselines/langchain_baseline.py

# Run DSPy baseline  
python bench/baselines/dspy_baseline.py
```

### Real Provider Mode (requires API keys)

Set environment variables to use real LLM providers:

```bash
# For OpenAI
export BASELINE_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# For Anthropic
export BASELINE_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Then run
python bench/baselines/langchain_baseline.py
python bench/baselines/dspy_baseline.py
```

## Output Schema

Both scripts output JSON matching the Aether benchmark report format:

```json
{
  "baseline": "langchain",
  "dataset": "synthetic_10",
  "latency_p50_ms": 150,
  "latency_p95_ms": 280,
  "latency_p99_ms": 350,
  "total_tokens": 5000,
  "cache_hit_rate": 0.15,
  "success_rate": 0.95,
  "measured_at": "2026-02-04T12:00:00Z",
  "mode": "mock"
}
```

## Comparison with Aether

These baselines represent the **projected baseline performance** from Section 8
of the Aether whitepaper. Key differences from Aether:

| Aspect | LangChain Baseline | DSPy Baseline | Aether |
|--------|-------------------|---------------|--------|
| Execution | Sequential | Sequential | Parallel DAG |
| Caching | Manual, low hit rate | None | Automatic, multi-level |
| Type Safety | Runtime JSON errors | Runtime errors | Compile-time checks |
| Parallelization | None | Limited | Automatic from DAG |

## Notes

- These are **stub implementations** for benchmark comparison purposes
- They do not depend on actual LangChain or DSPy packages to minimize dependencies
- The structure mimics typical usage patterns of those frameworks
- Latency and token costs in mock mode are simulated based on prompt length
