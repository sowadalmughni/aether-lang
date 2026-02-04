# Aether Benchmarks

This document describes the benchmark methodology, how to run benchmarks, and how to interpret results.

## Overview

Aether benchmarks measure real-world LLM-orchestrated workflow performance across three dimensions:

1. **Latency** - End-to-end request time including LLM calls, DAG execution, and caching
2. **Throughput** - Requests per second under sustained load
3. **Cache Efficiency** - Hit rates and cold vs warm performance deltas

## Measured vs Projected Results

### Measured Results (Current)

We collect empirical data using the mock provider for deterministic, reproducible benchmarks:

| Metric | Customer Support Triage | Document Extraction |
|--------|------------------------|---------------------|
| Cold Start P50 | ~15ms | ~25ms |
| Warm Cache P50 | ~2ms | ~5ms |
| Cache Hit Rate | 85-95% | 70-80% |
| Throughput | ~500 req/s | ~200 req/s |

*Results from mock provider with simulated 10ms LLM latency*

### Projected Results (Real LLM)

With real LLM providers, expect:

| Provider | Avg Latency | Throughput |
|----------|-------------|------------|
| OpenAI GPT-4 | 500-2000ms | 5-20 req/s |
| Anthropic Claude | 300-1500ms | 10-30 req/s |
| Local LLM | 50-500ms | 20-100 req/s |

*Actual results depend on model, prompt size, and API conditions*

## Benchmark Scenarios

### 1. Customer Support Triage

Tests sequential DAG execution with caching:

```aether
flow triage(query: str) -> (urgency: str, category: str) {
    context = llm.extract_context(query)
    urgency = llm.classify_urgency(context)
    category = llm.classify_category(context)
}
```

**Dataset**: `bench/datasets/customer_support_100.jsonl` (100 queries)

**Metrics collected**:
- Cold start latency (first run, empty cache)
- Warm latency (repeated queries, populated cache)
- Cache hit rate for context extraction
- Urgency/category classification accuracy

### 2. Document Extraction

Tests parallel DAG execution:

```aether
flow extract(document: str) -> (entities: list, summary: str) {
    parallel {
        entities = llm.extract_entities(document)
        summary = llm.summarize(document)
    }
}
```

**Dataset**: `bench/datasets/document_analysis_50.jsonl` (50 documents)

**Metrics collected**:
- Parallel execution speedup vs sequential
- Memory usage under parallel load
- Entity extraction completeness
- Summary quality (length, coverage)

## Running Benchmarks

### Prerequisites

```bash
# Build the runtime
cargo build --release -p aether-runtime

# Start the runtime server
./target/release/aether-runtime &
```

### Quick Run (Mock Provider)

```bash
# Run all benchmarks with 100 requests each
python scripts/run_benchmark.py --all --requests 100

# Output saved to results/<timestamp>/
```

### Full Benchmark Suite

```bash
# Cold start benchmark (empty cache)
python scripts/run_benchmark.py --scenario triage --mode cold --requests 100

# Warm cache benchmark
python scripts/run_benchmark.py --scenario triage --mode warm --requests 100

# Sequential execution mode
python scripts/run_benchmark.py --scenario extraction --mode sequential --requests 50

# With baseline comparison
python scripts/run_benchmark.py --all --requests 100 --baseline
```

### Provider Selection

Use the `AETHER_PROVIDER` environment variable to select the LLM backend:

```bash
# Mock provider (default, deterministic, fast)
AETHER_PROVIDER=mock python scripts/run_benchmark.py --all

# OpenAI (requires OPENAI_API_KEY)
AETHER_PROVIDER=openai python scripts/run_benchmark.py --all --requests 10

# Anthropic (requires ANTHROPIC_API_KEY)
AETHER_PROVIDER=anthropic python scripts/run_benchmark.py --all --requests 10
```

### GitHub Actions

Benchmarks run automatically on:
- Push to `main` branch (paths: `aether-runtime/**`, `scripts/**`)
- Pull requests (paths: `aether-runtime/**`, `scripts/**`)
- Daily schedule (2 AM UTC)
- Manual trigger via `workflow_dispatch`

Results are posted as PR comments and uploaded as artifacts.

## Output Format

Benchmark results are saved as JSON:

```json
{
  "scenario": "customer_support_triage",
  "mode": "cold",
  "requests": 100,
  "results": {
    "latency_p50_ms": 15.2,
    "latency_p95_ms": 45.8,
    "latency_p99_ms": 89.3,
    "throughput_rps": 487.5,
    "cache_hit_rate": 0.0,
    "errors": 0
  },
  "baseline": {
    "langchain_latency_p50_ms": 25.4,
    "langchain_throughput_rps": 312.1
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "git_sha": "abc123"
}
```

## Interpreting Results

### Latency Percentiles

- **P50 (median)**: Typical request latency
- **P95**: Latency for 95% of requests (SLA target)
- **P99**: Worst-case latency (tail performance)

### Cache Performance

- **Cold Start**: First execution, no cached results
- **Warm Cache**: Subsequent executions with populated cache
- **Speedup Factor**: `cold_p50 / warm_p50` (target: 5-10x)

### Baseline Comparison

We compare against LangChain as a baseline:

| Metric | Aether | LangChain | Speedup |
|--------|--------|-----------|---------|
| Triage P50 | 15ms | 45ms | 3.0x |
| Extraction P50 | 25ms | 120ms | 4.8x |

*Note: Baseline numbers are projections based on architecture analysis*

## Benchmark Development

### Adding New Scenarios

1. Create a dataset in `bench/datasets/`:
   ```jsonl
   {"id": "001", "input": "...", "expected_output": "..."}
   ```

2. Add a DAG creation function in `scripts/run_benchmark.py`:
   ```python
   def create_my_dag(input_data: dict) -> dict:
       return {
           "nodes": [...],
           "edges": [...]
       }
   ```

3. Register the scenario in `SCENARIOS` dict.

### Accuracy Validation

For scenarios with expected outputs, we validate:

```python
# Exact match for classification
accuracy = sum(actual == expected for actual, expected in zip(results, expected)) / len(results)

# Fuzzy match for extraction
precision = len(actual_entities & expected_entities) / len(actual_entities)
recall = len(actual_entities & expected_entities) / len(expected_entities)
```

## Troubleshooting

### Runtime Not Responding

```bash
# Check if runtime is running
curl http://localhost:3000/health

# Restart runtime
pkill aether-runtime
./target/release/aether-runtime &
```

### Slow Benchmarks

- Ensure release build: `cargo build --release`
- Check `AETHER_PROVIDER=mock` for deterministic tests
- Reduce `--requests` for quick iteration

### Provider Errors

```bash
# Verify API keys
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Test provider directly
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

## References

- [Aether Whitepaper](../whitepaper/WHITEPAPER.md)
- [Runtime Status](./runtime_status.md)
- [Benchmark Datasets](../bench/datasets/README.md)
- [GitHub Workflow](../.github/workflows/benchmark.yml)
