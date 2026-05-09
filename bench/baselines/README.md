# Baseline Benchmarks (LangChain & DSPy)

These are the LangChain and DSPy baselines that produce the comparison
numbers in [`bench/results/langchain_v1.json`](../results/langchain_v1.json),
[`bench/results/dspy_v1.json`](../results/dspy_v1.json), and their real-API
counterparts. They use real `langchain` 0.3.28 (`langchain-core` 0.3.84)
and real `dspy` 2.6.27 — both pinned in [`bench/requirements.txt`](../requirements.txt).
They are not simulations.

## Scripts

### langchain_baseline.py

LCEL chain (`Runnable | Runnable | …`) with optional `RunnableParallel`
for explicit parallel branches and `langchain.cache.InMemoryCache`
(LRU exact-match) for caching.

### dspy_baseline.py

DSPy `Module` composition using real `Signature`, `Predict`,
`ChainOfThought`, and `ChatAdapter` symbols. No automatic caching.

## Usage

### Mock Mode (default — no API keys required)

```bash
python bench/baselines/langchain_baseline.py
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

| Aspect | LangChain Baseline | DSPy Baseline | Aether |
|--------|-------------------|---------------|--------|
| Execution | LCEL chains, optionally `RunnableParallel` | DSPy `Module` + `Predict` (sequential) | Level-parallel DAG |
| Caching | `langchain.cache.InMemoryCache` (LRU exact-match) | None | L1 exact-match (measured 0.7/1.0 hit rate, [ablation_cache_v1.json](../results/ablation_cache_v1.json)) |
| Type Safety | Runtime parse errors (17/30 caught, [ablation_typesafety_v1.json](../results/ablation_typesafety_v1.json)) | Runtime parse errors (17/30 caught, same source) | Compile-time (30/30 caught, same source) |
| Parallelization | Where authored via `RunnableParallel` | None in baseline | Automatic from DAG (1.4778×/2.5841×, [ablation_parallel_v1.json](../results/ablation_parallel_v1.json)) |

## Result files

- [`bench/results/langchain_v1.json`](../results/langchain_v1.json) — mock-mode LangChain run
- [`bench/results/dspy_v1.json`](../results/dspy_v1.json) — mock-mode DSPy run
- [`bench/results/langchain_real_api_v1.json`](../results/langchain_real_api_v1.json) — real-API counterpart
- [`bench/results/dspy_real_api_v1.json`](../results/dspy_real_api_v1.json) — real-API counterpart
