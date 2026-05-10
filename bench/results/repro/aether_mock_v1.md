# Aether benchmark results — aether

- **Provider:** mock
- **Version (git SHA):** `unknown`
- **CPU:** Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
- **RAM:** 7.71 GiB
- **OS:** Debian GNU/Linux 13 (trixie)
- **Trials per config:** 5
- **Measured at (UTC):** 2026-05-10T05:07:45.714612Z

## Dataset: `customer_support_100`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | Cache hit rate | Tokens saved (total) |
| --- | ---: | --- | --- | --- | --- | --- |
| `sequential` | 5 | 190.40 ± 0.55 [190.00, 190.80] | 192.81 ± 0.42 [192.24, 193.00] | 193.61 ± 0.89 [193.01, 194.41] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel` | 5 | 127.20 ± 0.45 [127.00, 127.80] | 129.00 ± 0.71 [128.40, 129.60] | 129.21 ± 0.83 [128.60, 129.81] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel_cached` | 5 | 24.00 ± 0.00 [24.00, 24.00] | 24.81 ± 0.45 [24.20, 25.02] | 25.60 ± 0.54 [25.20, 26.00] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] |

## Dataset: `document_analysis_50`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | Cache hit rate | Tokens saved (total) |
| --- | ---: | --- | --- | --- | --- | --- |
| `sequential` | 5 | 201.60 ± 0.89 [201.00, 202.60] | 203.51 ± 1.24 [202.60, 204.51] | 204.01 ± 1.22 [203.11, 205.01] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel` | 5 | 75.60 ± 0.55 [75.20, 76.00] | 76.91 ± 0.20 [76.64, 77.00] | 77.10 ± 0.23 [77.00, 77.41] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] |
| `parallel_cached` | 5 | 23.20 ± 0.45 [23.00, 23.80] | 24.40 ± 0.55 [24.00, 24.80] | 24.60 ± 0.42 [24.20, 24.90] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] |

## Methodology

Each (dataset, config) tuple was measured over 5 independent trials of the full dataset. `sequential` and `parallel` clear the runtime cache before every trial; `parallel_cached` warms the cache once with a discarded warmup pass and then runs the measured trials without further clears. Per-request latencies feed per-trial p50/p95/p99 (`numpy.percentile`, linear method); cell entries are `mean ± std [95% CI]` aggregated across the per-trial scalars via `scipy.stats.bootstrap` (BCa, 10 000 resamples, seed=42; percentile fallback when BCa errors on degenerate variance). The mock provider is deterministic, so trial-to-trial variance reflects scheduling and HTTP-loopback jitter only.
