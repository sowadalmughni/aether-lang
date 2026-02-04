# Benchmark Metrics: Normalized Results

**Date**: February 4, 2026  
**Environment**: Mock LLM providers, 100ms simulated latency

---

## A. Baseline Results (Measured)

| Metric | LangChain | DSPy (Avg of 2 runs) |
|--------|-----------|----------------------|
| **latency_p50_ms** | 91.4 | 68.4 |
| **latency_p95_ms** | 103.01 | 77.8 |
| **latency_p99_ms** | 103.85 | 79.1 |
| **total_execution_time_ms** | 9027.13 | 6872.3 |
| **total_tokens** | 2269 | 5926 |
| **cache_hit_rate** | 0% | 0% |
| **success_rate** | 95% | 100% |
| **error_count** | 5 | 0 |
| **parallelization_factor** | 0.0 | 0.0 |
| **mode** | mock | mock |
| **requests** | 100 | 100 |

---

## B. Aether Results (Projected)

Based on runtime design (3 parallel LLM calls, 60% cache hit rate):

| Metric | Sequential | Parallel | Parallel + Cache |
|--------|------------|----------|------------------|
| **latency_p50_ms** | 274 | 103 | 58 |
| **latency_p95_ms** | 298 | 118 | 72 |
| **latency_p99_ms** | 312 | 125 | 84 |
| **speedup** | 1.0x | 2.7x | 4.7x |
| **cache_hit_rate** | N/A | N/A | 60% |

**Note**: Aether runtime requires MSVC toolchain to build. Projections based on design specifications.

---

## C. Comparison Summary

| System | p50 (ms) | Cache | Parallel | Success |
|--------|----------|-------|----------|---------|
| LangChain | 91.4 | 0% | No | 95% |
| DSPy | 68.4 | 0% | No | 100% |
| Aether (parallel) | 103* | 60%* | Yes | 100%* |

*Projected values
