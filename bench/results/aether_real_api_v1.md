# Aether benchmark results — aether

- **Provider:** openai
- **Version (git SHA):** `9c8001ce2699`
- **CPU:** Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
- **RAM:** 7.71 GiB
- **OS:** Ubuntu 24.04.4 LTS
- **Trials per config:** 3
- **Measured at (UTC):** 2026-05-08T06:29:58.125032Z

## Dataset: `customer_support_100`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | Cache hit rate | Tokens saved (total) |
| --- | ---: | --- | --- | --- | --- | --- |
| `sequential` | 3 | 6821.00 ± 636.32 [6121.50, 7235.67] | 11609.82 ± 2528.15 [8859.45, 13267.48] | 15420.54 ± 5168.38 [9793.46, 18807.97] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel` | 3 | 5395.17 ± 69.21 [5354.17, 5475.00] | 8148.40 ± 648.04 [7427.35, 8566.68] | 9881.67 ± 2167.48 [8460.80, 12240.65] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel_cached` | 3 | 37.33 ± 2.08 [35.00, 38.67] | 41.00 ± 0.00 [41.00, 41.00] | 41.37 ± 0.58 [41.02, 42.04] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] |

## Dataset: `document_analysis_50`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | Cache hit rate | Tokens saved (total) |
| --- | ---: | --- | --- | --- | --- | --- |
| `sequential` | 3 | 6814.00 ± 876.29 [6259.50, 7805.00] | 9274.10 ± 1195.59 [8505.95, 10610.55] | 12492.28 ± 3221.01 [10411.56, 16072.98] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel` | 3 | 2794.50 ± 335.47 [2589.67, 3179.50] | 5234.43 ± 172.59 [5053.75, 5349.05] | 9012.41 ± 4864.05 [6197.54, 14628.89] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel_cached` | 3 | 31.67 ± 0.58 [31.00, 32.00] | 38.00 ± 1.00 [37.00, 39.00] | 39.01 ± 0.49 [38.53, 39.51] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] |

## Methodology

Each (dataset, config) tuple was measured over 3 independent trials of the full dataset. `sequential` and `parallel` clear the runtime cache before every trial; `parallel_cached` warms the cache once with a discarded warmup pass and then runs the measured trials without further clears. Per-request latencies feed per-trial p50/p95/p99 (`numpy.percentile`, linear method); cell entries are `mean ± std [95% CI]` aggregated across the per-trial scalars via `scipy.stats.bootstrap` (BCa, 10 000 resamples, seed=42; percentile fallback when BCa errors on degenerate variance). The mock provider is deterministic, so trial-to-trial variance reflects scheduling and HTTP-loopback jitter only.
