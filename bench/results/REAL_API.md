# Real-API benchmark — Aether vs LangChain vs DSPy on gpt-4o-mini

This document describes the real-OpenAI benchmark run that produced
[`bench/results/real_api_v1.json`](real_api_v1.json). Every number here is
copied verbatim from that JSON or from the run log
[`bench/results/real_api_run.log`](real_api_run.log); nothing is fabricated
or extrapolated.

The run was driven end-to-end by [`scripts/run_real_api_benchmark.sh`](../../scripts/run_real_api_benchmark.sh),
which (1) verified `OPENAI_API_KEY` was set, (2) computed an upfront cost
estimate per the spec formula and aborted unless within the `--budget`
gate, (3) built `aether-runtime` with the `llm-api` cargo feature, (4)
spawned the runtime with `AETHER_PROVIDER=openai`, (5) ran the Aether
suite + LangChain baseline + DSPy baseline against `BASELINE_PROVIDER=openai`
with `--trials 3`, and (6) merged the three per-system JSONs into
`real_api_v1.json` via [`scripts/merge_real_api_results.py`](../../scripts/merge_real_api_results.py).

## Run metadata

| Field | Value |
| --- | --- |
| Schema | `aether-real-api-v1` |
| Run produced (UTC) | `2026-05-08T09:26:39.889631Z` |
| Wall-clock duration | 16,728 s (≈ 4 h 38 m) |
| Model | `gpt-4o-mini` (resolved to `gpt-4o-mini-2024-07-18` by OpenAI's router) |
| Trials per (dataset, config) | 3 |
| Datasets | `customer_support_100` (100 items, triage) + `document_analysis_50` (50 items, extraction) |
| Configs per dataset | `sequential`, `parallel`, `parallel_cached` |
| Pricing source | OpenAI public list price for gpt-4o-mini |
| Pricing — input | $0.15 / 1 M tokens |
| Pricing — output | $0.60 / 1 M tokens |
| Budget gate (`--budget`) | $10.00 |
| Upfront cost estimate (formula) | $1.4040 |
| Actual cost (measured) | **$0.478349** |
| Actual under budget | **YES** ($0.478 ≤ $10.00, used 4.78 % of budget) |
| All three systems hit identical datasets | **YES** |

The "actual cost" is computed in [`scripts/merge_real_api_results.py`](../../scripts/merge_real_api_results.py)
from token counts that each system recorded **straight from the OpenAI API
response payloads**:

* **Aether** — the runtime's `DagExecutionResponse.results[]` carries
  per-node `input_tokens` / `output_tokens` populated from the OpenAI
  response's `usage` block (see `aether-runtime/src/llm.rs:520-532`).
  [`scripts/run_benchmark.py`](../../scripts/run_benchmark.py) sums these
  per trial.
* **LangChain** — a `BaseCallbackHandler` (`UsageTracker` in
  [`bench/baselines/langchain_baseline.py`](../baselines/langchain_baseline.py))
  attached directly to the `ChatOpenAI` instance reads
  `llm_output["token_usage"]` on `on_llm_end`. We snapshot/diff per trial.
* **DSPy** — at the end of each trial we read
  `dspy.settings.lm.history[start_idx:].usage` (DSPy 2.6 stores the OpenAI
  `usage` field on every history entry) and sum
  `prompt_tokens` / `completion_tokens`.

## Per-system totals

| System | Version | Trial input toks | Trial output toks | Warmup input toks | Warmup output toks | Total input | Total output | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aether`    | `9c8001c…` (commit before this run) | 185,580 | 137,856 | 30,930 | 22,829 | 216,510 | 160,685 | $0.128887 |
| `langchain` | `0.3.28`  | 180,720 | 135,669 | 30,120 | 22,451 | 210,840 | 158,120 | $0.126498 |
| `dspy`      | `2.6.27`  | 566,133 | 176,942 | 94,353 | 29,542 | 660,486 | 206,484 | $0.222963 |
| **TOTAL**   |           | 932,433 | 450,467 | 155,403 | 74,822 | **1,087,836** | **525,289** | **$0.478349** |

"Warmup" tokens are tokens spent during the `parallel_cached` warmup pass.
The warmup IS a real API hit (cache freshly cleared) so its tokens are real
cost, even though the warmup itself is discarded from latency aggregates.
The merge script counts them in the actual cost.

## Per-(system, dataset, config) latency

p50 / p95 are mean ± std across 3 trials, in milliseconds. Hit-rate is the
runtime cache hit rate (Aether) or the explicit prompt-keyed cache hit
rate (baselines). Sequential and parallel configs clear the cache before
every trial; `parallel_cached` warms the cache once, discards the warmup,
and runs 3 measured trials over the populated cache.

### customer_support_100 (triage, 100 items, 3 LLM calls per item)

| System | Config | p50 (ms) | p95 (ms) | Hit rate | Trial in/out toks | Warmup in/out toks |
| --- | --- | --- | --- | ---: | --- | --- |
| aether    | sequential      | 6821.00 ± 636.32 | 11609.82 ± 2528.15 | 0.000 | 46,437 / 48,826 | 0 / 0 |
| aether    | parallel        | 5395.17 ± 69.21  | 8148.40 ± 648.04   | 0.000 | 46,437 / 48,409 | 0 / 0 |
| aether    | parallel_cached | 37.33 ± 2.08     | 41.00 ± 0.00       | 1.000 | 0 / 0           | 15,479 / 16,200 |
| langchain | sequential      | 5306.09 ± 355.27 | 9102.86 ± 1080.57  | 0.000 | 42,933 / 47,477 | 0 / 0 |
| langchain | parallel        | 4754.53 ± 263.57 | 8215.10 ± 548.18   | 0.000 | 42,933 / 47,574 | 0 / 0 |
| langchain | parallel_cached | 3.39 ± 0.18      | 4.06 ± 0.26        | 1.000 | 0 / 0           | 14,311 / 15,708 |
| dspy      | sequential      | 5926.07 ± 409.95 | 9120.28 ± 1011.91  | 0.000 | 175,860 / 46,562 | 0 / 0 |
| dspy      | parallel        | 5276.27 ± 193.83 | 8389.91 ± 369.83   | 0.000 | 175,857 / 46,219 | 0 / 0 |
| dspy      | parallel_cached | 0.72 ± 0.06      | 1.17 ± 0.51        | 1.000 | 0 / 0           | 58,617 / 15,439 |

### document_analysis_50 (extraction, 50 items, 3 LLM calls per item)

| System | Config | p50 (ms) | p95 (ms) | Hit rate | Trial in/out toks | Warmup in/out toks |
| --- | --- | --- | --- | ---: | --- | --- |
| aether    | sequential      | 6814.00 ± 876.29 | 9274.10 ± 1195.59 | 0.000 | 46,353 / 20,274 | 0 / 0 |
| aether    | parallel        | 2794.50 ± 335.47 | 5234.43 ± 172.59  | 0.000 | 46,353 / 20,347 | 0 / 0 |
| aether    | parallel_cached | 31.67 ± 0.58     | 38.00 ± 1.00      | 1.000 | 0 / 0           | 15,451 / 6,629  |
| langchain | sequential      | 4563.88 ± 427.52 | 8826.45 ± 2684.57 | 0.000 | 47,427 / 20,355 | 0 / 0 |
| langchain | parallel        | 2238.17 ± 117.99 | 4514.95 ± 267.35  | 0.000 | 47,427 / 20,263 | 0 / 0 |
| langchain | parallel_cached | 2.49 ± 0.04      | 2.69 ± 0.04       | 1.000 | 0 / 0           | 15,809 / 6,743  |
| dspy      | sequential      | 7806.07 ± 1693.17 | 10328.14 ± 2467.17 | 0.000 | 107,208 / 42,153 | 0 / 0 |
| dspy      | parallel        | 3257.48 ± 257.92  | 6376.47 ± 2092.93  | 0.000 | 107,208 / 42,008 | 0 / 0 |
| dspy      | parallel_cached | 1.10 ± 0.06       | 1.74 ± 0.76        | 1.000 | 0 / 0            | 35,736 / 14,103 |

## Anomalies and notes

* **Aether `parallel_cached` p50 (~30–40 ms) is two orders of magnitude
  slower than the LangChain / DSPy `parallel_cached` p50 (~1–4 ms).** This
  is faithful to the deployed shape, not a measurement bug: Aether's
  cache lives inside the runtime, so a cached request still pays one HTTP
  loopback round-trip from the bench client to `127.0.0.1:3000` per item.
  LangChain's `ExplicitCache` and DSPy's `ExplicitCache` are in-process
  Python dictionaries with zero network. If we cared to compare like
  shapes, the relevant Aether number would be the runtime's internal
  cache hit time, not the wall-clock from the bench client.

* **DSPy uses ~3× the input tokens of LangChain or Aether** (660 k vs
  ~210 k). Cause: `dspy.ChainOfThought` and `dspy.Predict` serialize the
  full signature definition (input/output field names + descriptions +
  ChatAdapter formatting markers) into every prompt. This is a property
  of DSPy's design — it's how the framework binds prompts to typed
  signatures. We did not modify it; the comparison reflects the systems
  as deployed.

* **Aether sequential is consistently slower than LangChain sequential**
  (e.g. 6821 ms vs 5306 ms on customer_support_100). Aether pays per-item
  HTTP overhead from the bench client to the runtime in addition to the
  OpenAI call. With three calls per item, that's three round-trips of
  loopback overhead. LangChain/DSPy run the LLM call directly from the
  bench process. As above, this is the deployment-shape difference, not
  a comparator artefact.

* **Real cost was 34 % of upfront estimate** ($0.478 vs $1.404). The
  estimate assumed 200 output tokens per call uniformly; in practice
  classification steps (urgency, category, domain) emit ≤ 30 tokens, and
  only the response-generation step emits ~100+ tokens. The estimate
  also assumed extraction has 5 calls per doc; the actual scenario uses
  3 LLM nodes plus a non-LLM combine.

* **No item failures.** `errors=0` for every (system, dataset, config,
  trial). No `429` rate-limit responses observed in
  `bench/results/runtime_real_api.log` or in either baseline's stderr.
  This is consistent with the run pacing — peak concurrent requests
  is 3 for the parallel configs, well below OpenAI's per-minute limits.

* **All three systems hit the same items.** The merge step verifies this
  and records `all_systems_same_datasets: true` in the combined JSON.

* **Runtime log is 114 MB.** `bench/results/runtime_real_api.log` is verbose
  (one tracing line per request, with the full DAG payload inlined). It
  is *not* committed; only the structured JSON and this Markdown are.

## Verbatim merge-step terminal output

The wrapper script's final stage (`[6/6]`) was
[`scripts/merge_real_api_results.py`](../../scripts/merge_real_api_results.py).
This is the exact stdout it printed at the end of the run (copied verbatim
from `bench/results/real_api_run.log`, which is gitignored as a `*.log`):

```
================================================================
Real-API benchmark — actual cost (from response usage)
================================================================
  model:           gpt-4o-mini
  rates:           $0.15/1M in, $0.6/1M out
  [aether   ] tokens_in=   216510  tokens_out=   160685  cost=$0.128887
  [langchain] tokens_in=   210840  tokens_out=   158120  cost=$0.126498
  [dspy     ] tokens_in=   660486  tokens_out=   206484  cost=$0.222963
        TOTAL  tokens_in=  1087836  tokens_out=   525289  cost=$0.478349
  budget: $10.00, estimated upfront: $1.4040, actual: $0.478349
  datasets identical across systems: YES (['customer_support_100', 'document_analysis_50'])
  wrote: bench/results/real_api_v1.json

Done in 16728 s. Combined real-API benchmark at bench/results/real_api_v1.json
```

## How to verify

```bash
# Re-derive the cost from the committed JSON without trusting the merge script.
python3 - <<'PY'
import json
d = json.load(open('bench/results/real_api_v1.json'))
ip = d['pricing']['input_per_million_usd']
op = d['pricing']['output_per_million_usd']
cost = d['tokens_input_total']*ip/1e6 + d['tokens_output_total']*op/1e6
print(f"recomputed cost = ${cost:.6f}, recorded = ${d['actual_cost_usd']}")
PY
# Expected output:
#   recomputed cost = $0.478349, recorded = $0.478349
```

## OpenAI dashboard

The user-facing OpenAI usage dashboard for this account/project (covering
2026-05-08) is the source-of-truth billing record. Per-call response
`usage` blocks (which we summed into the JSON) are what populate it. We
do not attach a screenshot here; the JSON's per-trial `tokens_input` /
`tokens_output` numbers, plus the merge-step terminal output above, are
self-contained and reproducible.
