# Aether benchmark results — aether

- **Provider:** mock
- **Version (git SHA):** `4d16ec5cd5cb`
- **CPU:** Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
- **RAM:** 7.71 GiB
- **OS:** Ubuntu 24.04.4 LTS
- **Trials per config:** 5
- **Measured at (UTC):** 2026-05-03T09:49:57.976386Z

## Dataset: `customer_support_100`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | Cache hit rate | Tokens saved (total) |
| --- | ---: | --- | --- | --- | --- | --- |
| `sequential` | 5 | 209.20 ± 1.92 [207.80, 211.00] | 217.65 ± 0.91 [217.03, 218.46] | 248.62 ± 22.61 [232.32, 267.82] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel` | 5 | 145.80 ± 0.84 [145.20, 146.60] | 150.04 ± 0.71 [149.45, 150.64] | 152.63 ± 1.50 [151.66, 154.03] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel_cached` | 5 | 33.00 ± 0.71 [32.40, 33.60] | 39.02 ± 1.24 [37.60, 39.65] | 40.65 ± 1.60 [39.61, 42.33] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] |

## Dataset: `document_analysis_50`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | Cache hit rate | Tokens saved (total) |
| --- | ---: | --- | --- | --- | --- | --- |
| `sequential` | 5 | 224.40 ± 1.19 [223.50, 225.40] | 230.15 ± 0.55 [229.75, 230.55] | 233.25 ± 2.27 [231.71, 235.29] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel` | 5 | 88.50 ± 1.22 [87.50, 89.40] | 92.91 ± 0.56 [92.40, 93.22] | 94.31 ± 2.19 [93.00, 97.05] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel_cached` | 5 | 31.50 ± 0.71 [31.00, 32.10] | 38.71 ± 0.44 [38.20, 39.00] | 39.92 ± 1.05 [39.18, 40.73] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] |

## Methodology

Each (dataset, config) tuple was measured over 5 independent trials of the full dataset. `sequential` and `parallel` clear the runtime cache before every trial; `parallel_cached` warms the cache once with a discarded warmup pass and then runs the measured trials without further clears. Per-request latencies feed per-trial p50/p95/p99 (`numpy.percentile`, linear method); cell entries are `mean ± std [95% CI]` aggregated across the per-trial scalars via `scipy.stats.bootstrap` (BCa, 10 000 resamples, seed=42; percentile fallback when BCa errors on degenerate variance). The mock provider is deterministic, so trial-to-trial variance reflects scheduling and HTTP-loopback jitter only.
