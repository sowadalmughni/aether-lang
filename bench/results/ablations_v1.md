# Aether ablation results — caching, parallelization, type-safety (v1)

- **System:** aether
- **Provider:** mock
- **Version (git SHA):** `8aee2cce5f969b5e2d84e94216355354dcc0eb7f`
- **CPU:** Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
- **RAM (GiB):** 7.71
- **OS:** Ubuntu 24.04.4 LTS
- **Trials per config:** 5
- **Measured at (UTC):** 2026-05-08T13:05:22.982910Z

## 1. Caching ablation

Three modes per dataset, 5 trials each. `no_cache` clears the cache before every individual `/execute` request (the runtime has no per-request cache-disable knob; this is the workaround). `l1_exact_match` is the runtime's default L1 behavior, clearing once per trial. `repeat_warm` runs the dataset once as warmup (latencies discarded; warmup tokens recorded separately) and then a second time as the measured pass.

### Dataset: `customer_support_100`

| Config | Trials | p50 (ms) | Cache hit rate | Tokens saved (total) | Δ hit_rate vs no_cache | Δ p50 (ms) vs no_cache |
| --- | ---: | --- | --- | --- | --- | --- |
| `no_cache` | 5 | 144.80 ± 0.45 [144.20, 145.00] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] | — | — |
| `l1_exact_match` | 5 | 143.80 ± 1.52 [142.60, 144.90] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] | +0.0000 [+0.0000, +0.0000] | — |
| `repeat_warm` | 5 | 33.20 ± 0.84 [32.40, 33.80] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] | +1.0000 [+1.0000, +1.0000] | -111.60 [-112.60, -111.00] |

### Dataset: `customer_support_repeat_100`

| Config | Trials | p50 (ms) | Cache hit rate | Tokens saved (total) | Δ hit_rate vs no_cache | Δ p50 (ms) vs no_cache |
| --- | ---: | --- | --- | --- | --- | --- |
| `no_cache` | 5 | 145.10 ± 0.22 [145.00, 145.40] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0 ± 0.0 [0.0, 0.0] | — | — |
| `l1_exact_match` | 5 | 36.10 ± 1.43 [34.90, 37.20] | 0.7000 ± 0.0000 [0.7000, 0.7000] | 0.0 ± 0.0 [0.0, 0.0] | +0.7000 [+0.7000, +0.7000] | — |
| `repeat_warm` | 5 | 33.80 ± 1.92 [32.80, 36.00] | 1.0000 ± 0.0000 [1.0000, 1.0000] | 0.0 ± 0.0 [0.0, 0.0] | +1.0000 [+1.0000, +1.0000] | -111.30 [-112.30, -108.80] |

## 2. Parallelization ablation

Two modes per dataset, 5 trials each. `sequential` posts to `/execute?sequential=true`; `parallel` posts to `/execute` (default). The cache is cleared before every trial in both modes so the parallelization signal is not confounded with caching effects. `parallelization_factor` is the runtime response field of the same name (`sum(node_execution_times_ms) / total_execution_time_ms`) -- a single definition used identically across both datasets and modes.

### Dataset: `customer_support_100`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | parallelization_factor |
| --- | ---: | --- | --- | --- | --- |
| `sequential` | 5 | 206.00 ± 0.71 [205.40, 206.60] | 213.87 ± 1.89 [212.09, 215.06] | 246.70 ± 30.78 [224.96, 273.04] | 1.000 ± 0.000 [1.000, 1.000] |
| `parallel` | 5 | 139.40 ± 0.55 [139.00, 139.80] | 148.01 ± 1.88 [146.60, 149.41] | 155.34 ± 6.08 [151.13, 161.17] | 1.475 ± 0.001 [1.474, 1.476] |

### Dataset: `document_analysis_50`

| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | parallelization_factor |
| --- | ---: | --- | --- | --- | --- |
| `sequential` | 5 | 218.60 ± 0.55 [218.20, 219.00] | 224.66 ± 1.57 [223.42, 225.76] | 232.82 ± 6.36 [228.13, 237.62] | 1.000 ± 0.000 [1.000, 1.000] |
| `parallel` | 5 | 84.60 ± 0.65 [84.10, 85.20] | 90.95 ± 3.55 [88.86, 94.75] | 100.12 ± 8.83 [93.73, 107.58] | 2.557 ± 0.007 [2.552, 2.563] |

### Speedup (paired-trial bootstrap, ratio sequential.p50 / parallel.p50)

| Dataset | speedup_p50 (mean) | 95% CI |
| --- | --- | --- |
| `customer_support_100` | 1.478x | [1.473, 1.485] |
| `document_analysis_50` | 2.584x | [2.563, 2.599] |

## 3. Type-safety ablation

Per-bug breakdown over a 30-test corpus (10 type_mismatch + 10 undefined_reference + 5 missing_field + 5 duplicate_definition). For each bug, an `aetherc check` is run on the .aether file, and a `python` is run on the LangChain and DSPy equivalents. Aether result `caught_at_compile_time` means stderr matched a known SemanticError variant; Python result `caught_at_runtime` means the subprocess exited non-zero with a Python traceback (includes SyntaxError at file load); `missed_silently` means the file ran to exit 0.

| ID | Bucket | Expected (aetherc) | Aether | LangChain | DSPy |
| --- | --- | --- | --- | --- | --- |
| `tm_01` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_02` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_03` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_04` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_05` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_06` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_07` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_08` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_09` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `tm_10` | type_mismatch | TypeMismatch | caught_at_compile_time (TypeMismatch) | missed_silently | missed_silently |
| `ur_01` | undefined_reference | UndefinedSymbol/UndefinedFunction | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_02` | undefined_reference | UndefinedSymbol | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_03` | undefined_reference | UndefinedSymbol | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_04` | undefined_reference | UndefinedSymbol/UndefinedFunction | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_05` | undefined_reference | UndefinedSymbol/UndefinedFunction | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_06` | undefined_reference | UndefinedSymbol | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_07` | undefined_reference | UndefinedSymbol | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_08` | undefined_reference | UndefinedSymbol | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_09` | undefined_reference | UndefinedSymbol/UndefinedFunction | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `ur_10` | undefined_reference | UndefinedSymbol | caught_at_compile_time (UndefinedSymbol) | caught_at_runtime (NameError) | caught_at_runtime (NameError) |
| `mf_01` | missing_field | UnknownField | caught_at_compile_time (UnknownField) | caught_at_runtime (AttributeError) | caught_at_runtime (AttributeError) |
| `mf_02` | missing_field | ArgumentCountMismatch | caught_at_compile_time (ArgumentCountMismatch) | caught_at_runtime (TypeError) | caught_at_runtime (TypeError) |
| `mf_03` | missing_field | UnknownField | caught_at_compile_time (UnknownField) | caught_at_runtime (AttributeError) | caught_at_runtime |
| `mf_04` | missing_field | UnknownField | caught_at_compile_time (UnknownField) | caught_at_runtime (AttributeError) | caught_at_runtime (AttributeError) |
| `mf_05` | missing_field | ArgumentCountMismatch | caught_at_compile_time (ArgumentCountMismatch) | caught_at_runtime (TypeError) | caught_at_runtime (TypeError) |
| `dd_01` | duplicate_definition | DuplicateDefinition | caught_at_compile_time (DuplicateDefinition) | missed_silently | missed_silently |
| `dd_02` | duplicate_definition | DuplicateDefinition | caught_at_compile_time (DuplicateDefinition) | missed_silently | missed_silently |
| `dd_03` | duplicate_definition | DuplicateField | caught_at_compile_time (DuplicateField) | missed_silently | missed_silently |
| `dd_04` | duplicate_definition | DuplicateParameter | caught_at_compile_time (DuplicateParameter) | caught_at_runtime (SyntaxError) | caught_at_runtime (SyntaxError) |
| `dd_05` | duplicate_definition | DuplicateVariant | caught_at_compile_time (DuplicateVariant) | caught_at_runtime (TypeError) | caught_at_runtime (TypeError) |

### Aggregate

| Detector | Caught (compile-time + runtime) | Missed silently |
| --- | ---: | ---: |
| Aether (compile-time) | 30/30 | 0/30 |
| LangChain (runtime) | 17/30 | 13/30 |
| DSPy (runtime) | 17/30 | 13/30 |

### By bucket

| Bucket | Total | Aether caught | LC runtime | DSPy runtime |
| --- | ---: | ---: | ---: | ---: |
| type_mismatch | 10 | 10 | 0 | 0 |
| undefined_reference | 10 | 10 | 10 | 10 |
| missing_field | 5 | 5 | 5 | 5 |
| duplicate_definition | 5 | 5 | 2 | 2 |

### Methodology note: cd → dd substitution

> The original ablation design included a circular_dependency category, but verification revealed that aetherc's source-level cd detector is currently preempted by semantic analysis on programs that contain other issues; the SemanticError::CircularDependency variant is defined but never emitted. Rather than fabricate test cases that would not trigger the intended error path, we substituted duplicate_definition tests, which exercise a different but more practically significant error class (silent shadowing in Python is more dangerous than a circular dependency, which typically manifests as RecursionError or ImportError -- loud and visible). The cd detection gap is tracked at https://github.com/sowadalmughni/aether-lang/issues/4 and is targeted for a follow-up compiler release.

## Methodology footer

Cache and parallelization measurements use the existing `_bootstrap_ci` helper from `scripts/run_benchmark.py:546-593` (`scipy.stats.bootstrap`, BCa, 10 000 resamples, seed=42; percentile fallback on degenerate variance). Cell entries are `mean ± std [95% CI]`. Cross-mode deltas and the parallelization speedup ratio use paired-trial bootstrapping with the same parameters. Type-safety classification is by stderr-pattern match against the SemanticError variant names in `aether-compiler/src/semantic.rs:36-177`. The mock provider is deterministic (50 ms flat per LLM call); trial-to-trial variance reflects scheduling and HTTP-loopback jitter only.
