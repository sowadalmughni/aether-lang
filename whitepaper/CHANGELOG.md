# Aether Whitepaper Changelog

**Current Version**: 3.0.2  
**Last Updated**: May 10, 2026  
**Status**: Prototype - Runtime, real-API, security suite all measured; reproduce.sh end-to-end clean; whitepaper consolidated to single canonical source via pandoc/docx

---

## Version History

| Version | Date | Status | Summary |
|---------|------|--------|---------|
| 3.0.2 | May 10, 2026 | Single canonical paper, pandoc/docx pipeline | LaTeX/PDF pipeline removed (archived to `whitepaper/archive/latex/`); `WHITEPAPER_ACADEMIC.md` is the sole markdown source of truth; figure set reduced from 10 to 5 PNGs (300 DPI) at `whitepaper/figures/`; 7 ` ```aether ` fences retagged ` ```rust ` for skylighting; `whitepaper/aether.docx` is the canonical build output (49 pages, 490866 bytes via pandoc 3.9.0.2). No measurement-data delta from 3.0.1. |
| 3.0.1 | May 10, 2026 | Reproducibility patches | `bench/requirements.txt` openai/litellm pins reconciled; `bench/run_all_mock.py` orchestrator paths fixed for the docker image; `aether_mock_v1.json` and `dspy_v1.json` references refreshed for per-trial `tokens_total` (added post-cc49f3d); `reproduce.sh` now runs end-to-end with verifier verdict "numeric drift within informational tolerance (exit 0)" |
| 3.0 | May 8, 2026 | Measured-data revision | Runtime executed; every numeric claim now traces to a JSON in `bench/results/`; "(projected)" labels removed; Reproducibility callout, Statistical Methodology, HotpotQA + Security results, hardware-variance threat added |
| 2.7 | Feb 5, 2026 | Telemetry & Benchmarks | OTLP tracing re-enabled, criterion benchmarks, OpenTelemetry 0.21.0 |
| 2.6 | Feb 4, 2026 | Full Benchmark Suite | Synthetic datasets, benchmark runner, provider switching, CI integration |
| 2.5 | Feb 4, 2026 | Benchmark Infrastructure | Latency percentiles, sequential mode, baseline stubs |
| 2.4 | Feb 4, 2026 | Runtime MVP | Parallel execution, caching, template engine, observability |
| 2.3 | Feb 2026 | End-to-End Demo | CLI `run` command, DAG Visualizer enhancements |
| 2.2 | Feb 2026 | Type System MVP | 5-pass semantic analyzer, comprehensive type checking |
| 2.1 | Feb 2026 | Phase 1 Complete | Parser, semantic analysis, code generator, CLI |
| 2.0 | Feb 2026 | Major Revision | Research update, restructured whitepaper |
| 1.0 | Jul 2025 | Initial Draft | Original whitepaper |

---

## [3.0.2] - May 10, 2026

### Summary
LaTeX/PDF pipeline replaced with pandoc/docx. Whitepaper consolidated to a single canonical source (`WHITEPAPER_ACADEMIC.md`); figure set reduced from 10 PDFs to 5 PNGs (300 DPI). No measurement-data delta from 3.0.1; every numeric claim still traces to the same JSONs in `bench/results/`.

### Changed — pipeline
- `whitepaper/Makefile`: rewritten as a single `docx` target driving `pandoc -> docx` with `--reference-doc=reference.docx`, `--syntax-highlighting=tango`, `--metadata-file=metadata.yml`, `--toc --toc-depth=2`, `--resource-path=.:figures`. Tectonic and the LaTeX preamble/header.tex/preprocess_md.py are no longer used.
- `bench/figures/generate_figures.py`: emits PNG (300 DPI, white background, tight bbox) to `whitepaper/figures/`. Was: PDF to `whitepaper/latex/figures/`. Function `fig3_cache_hit_rate` removed; the cache numbers (0.7000 L1 hit rate, 1.0000 warm, −75% / −77% latency reduction vs the 144.8 ms baseline — all from `bench/results/ablation_cache_v1.json`) fold into a one-line summary beneath the §4.2 parallel_speedup figure caption. No new measurements introduced.
- `bench/figures/regenerate.sh`: header comment updated to reflect new path/format.

### Changed — markdown source
- `WHITEPAPER_ACADEMIC.md`: 7 ` ```aether ` code fences retagged ` ```rust ` (lines previously at 278, 302, 332, 364, 373, 399, 792) so pandoc's skylighting renders Rust-grammar syntax highlighting in the docx, which colorises `fn`/`let`/`match`/`enum`/`struct`/`if`/`else` correctly for the Aether code samples.
- `WHITEPAPER_ACADEMIC.md`: 5 inline figure references inserted at sections 4.2, 6.1, 9.2, 9.5, 11.2 — `parallel_speedup.png`, `compiler_pipeline.png`, `cross_system_latency.png`, `type_safety_corpus.png`, `security_outcome.png`. The cache one-liner sits beneath the §4.2 figure.

### Added
- `whitepaper/figures/` — 5 PNGs (300 DPI) + 4 JSON sidecars + `compiler_pipeline.mmd` source.
  - `cross_system_latency.png` (83 KB) + sidecar — sourced from `aether_mock_v1.json`, `langchain_v1.json`, `dspy_v1.json`
  - `parallel_speedup.png` (76 KB) + sidecar — sourced from `ablation_parallel_v1.json`
  - `type_safety_corpus.png` (99 KB) + sidecar — sourced from `ablation_typesafety_v1.json`
  - `security_outcome.png` (123 KB) + sidecar — sourced from `security_v1.json`
  - `compiler_pipeline.png` (101 KB) — rendered from `compiler_pipeline.mmd` via mermaid-cli (mmdc 10.9.1 -e png -s 2 -b white -t default); not data-driven, no sidecar.
- `whitepaper/metadata.yml` — title, author, date, abstract (verbatim 3-paragraph copy from WHITEPAPER_ACADEMIC.md), 5 keywords.
- `whitepaper/reference.docx` — pandoc default reference docx (`pandoc -o reference.docx --print-default-data-file reference.docx`); 11 KB. Editable later for typography.
- `whitepaper/aether.docx` — build output (490866 bytes; 49 pages via libreoffice headless conversion). Committed as a final artefact rather than ignored.

### Removed (archived to `whitepaper/archive/`, not deleted)
- `whitepaper/WHITEPAPER.md` (engineering reference; superseded by the single canonical paper).
- `whitepaper/aether-whitepaper-academic-changes.md` (diff log between the two papers).
- `whitepaper/audit_mapping.md`, `whitepaper/benchmark_metrics.md`, `whitepaper/TODO.md` (scratch notes).
- `whitepaper/latex/` (entire directory: aether.tex, aether.pdf, aether.log, aether.aux, aether.preprocessed.md, header.tex, references.bib, preprocess_md.py, arxiv.sty, SUBMISSION.md, figures/PDFs).
- `whitepaper/diagrams/` (entire directory: 5 .mmd sources + render.sh; only compiler_pipeline.mmd is rerendered as PNG and lives at `whitepaper/figures/compiler_pipeline.mmd`).

### Acceptance verification
```
$ ls whitepaper/
CHANGELOG.md  Makefile  WHITEPAPER_ACADEMIC.md  aether.docx  archive  figures  metadata.yml  reference.docx

$ ls whitepaper/figures/
compiler_pipeline.mmd  compiler_pipeline.png  cross_system_latency.json
cross_system_latency.png  parallel_speedup.json  parallel_speedup.png
security_outcome.json  security_outcome.png  type_safety_corpus.json
type_safety_corpus.png

$ ls whitepaper/archive/
TODO.md  WHITEPAPER.md  aether-whitepaper-academic-changes.md  audit_mapping.md
benchmark_metrics.md  diagrams  latex

$ test ! -e "Aether Programming Language.md" && echo absent
absent

$ grep -nE '^```aether' whitepaper/WHITEPAPER_ACADEMIC.md ; echo "exit=$?"
exit=1   # no hits

$ grep -c '!\[Figure' whitepaper/WHITEPAPER_ACADEMIC.md
5

$ pandoc whitepaper/WHITEPAPER_ACADEMIC.md \
    --from=markdown+pipe_tables+grid_tables+raw_html --to=docx \
    --reference-doc=whitepaper/reference.docx --resource-path=.:whitepaper:whitepaper/figures \
    --syntax-highlighting=tango --metadata-file=whitepaper/metadata.yml \
    --toc --toc-depth=2 -o whitepaper/aether.docx
# (no stdout, no warnings, exit 0)

$ wc -c whitepaper/aether.docx
490866 whitepaper/aether.docx

$ "/c/Program Files/LibreOffice/program/soffice.exe" --headless \
    --convert-to pdf whitepaper/aether.docx --outdir /tmp/
convert E:\Project\aether-lang\whitepaper\aether.docx as a Writer document
  -> /tmp/aether.pdf using filter : writer_pdf_Export
# exit 0; PDF size 1437428 bytes

$ python -c "import pypdf; r=pypdf.PdfReader('/tmp/aether.pdf'); print(len(r.pages))"
49
```

The PDF was inspected programmatically (pypdf): page 1 contains the title, author, date, full abstract, and "Table of Contents" heading; all 5 figure captions are present at pages 11 (Fig 1), 18 (Fig 2), 26 (Fig 3), 30 (Fig 4), and 39 (Fig 5). The PDF is the headless sanity-check artefact and is **not** committed; users open `whitepaper/aether.docx` in Word or LibreOffice GUI for final visual review of code-block colouring, table layout, and figure placement.

---

## [3.0.1] - May 10, 2026

### Summary
Artifact-evaluation reproducibility patch. No paper-level numbers change; the v3.0 measured-data claims continue to hold. This revision fixes the issues a third party would hit running `reproduce.sh` from a fresh clone, and refreshes the two reference JSONs whose per-trial `tokens_total` field was committed before the token-tracking commit (cc49f3d) and therefore failed the strict `exact_required` diff.

### Fixed — bench/requirements.txt
- `openai==1.55.0` was unresolvable: `langchain-openai==0.2.14` requires `openai>=1.58.1,<2.0.0`. Bumped to `openai==1.58.1` (commit `296819c`).
- `openai==1.58.1` was *also* unresolvable once `litellm` (transitive via `dspy==2.6.27`'s `litellm>=1.60.3`) tightened its lower bound on openai through 2025-2026 (1.61 → 1.66.1 → 1.68.2 → 2.8.0 at 1.80+). Bumped to `openai==1.68.2` and pinned `litellm==1.69.2` explicitly so pip cannot silently drift onto a litellm version that requires `openai>=2.8.0` and breaks `langchain-openai`'s `<2.0.0` cap (commit `912670f`).

### Fixed — bench/run_all_mock.py orchestrator
- `bench/runners/ablation_typesafety.py` defaults to `target/debug/aetherc` and `/home/deamers_academy/aether-bench-venv/bin/python` — both maintainer-host artifacts that do not exist in the docker image (Dockerfile builds release-only; the bench venv lives at `/opt/venv`). The runner exposes `--aetherc` and `--python` overrides; the orchestrator now passes them explicitly with the correct in-container paths (commit `a838fb5`).
- `bench/baselines/aether_hotpot.py` does not have a `--mode` flag (mock vs real-API is a property of the runtime it talks to). The orchestrator was passing `--mode mock` and aborting the suite with `argparse: unrecognized arguments`. The argument is now omitted; the runtime container's `AETHER_PROVIDER=mock` continues to drive mock mode (commit `2f50a4a`).

### Refreshed — committed reference JSONs
- `bench/results/aether_mock_v1.json` and `bench/results/dspy_v1.json` were committed at 852be6e and 7d2be1c respectively (2026-05-03), *before* commit cc49f3d ("feat(bench): record real OpenAI token usage per trial") added per-trial token tracking. Their `results[*].raw_trial_results[*].tokens_total` field was uniformly `0`. The current bench code correctly emits real counts (deterministic in mock mode: 34817 for `customer_support_100`, 25885 for `document_analysis_50`, 88686 / 52259 for the dspy variants). Both reference JSONs are refreshed to match current code; `bench/results/aether_mock_v1.md` is regenerated from the same run.

### Acceptance verification
```
$ docker compose -p aether-repro up -d runtime  # healthy
$ docker exec aether-bench-run python bench/run_all_mock.py
[aether_mock_v1] exit=0
[ablation_cache_v1] exit=0
[ablation_parallel_v1] exit=0
[ablation_typesafety_v1] exit=0
[langchain_v1] exit=0
[dspy_v1] exit=0
[hotpotqa_aether_v1] exit=0
[hotpotqa_langchain_v1] exit=0
[hotpotqa_dspy_v1] exit=0

$ python bench/verify_reproduction.py --reference bench/results --produced bench/results/repro \
      --tolerance bench/verify_tolerance.toml --report bench/results/repro/DIFF_REPORT.md
ablation_cache_v1.json: numeric drift (within informational tolerance)
ablation_parallel_v1.json: numeric drift (within informational tolerance)
ablation_typesafety_v1.json: ok
aether_mock_v1.json: ok
aether_real_api_v1.json: n/a (requires OPENAI_API_KEY)
dspy_v1.json: ok
hotpotqa_aether_v1.json: numeric drift (within informational tolerance)
hotpotqa_dspy_v1.json: numeric drift (within informational tolerance)
hotpotqa_langchain_v1.json: numeric drift (within informational tolerance)
langchain_v1.json: numeric drift (within informational tolerance)
real_api_v1.json: n/a (requires OPENAI_API_KEY)
security_v1.json: n/a (requires OPENAI_API_KEY)
VERDICT: numeric drift within informational tolerance (exit 0)
```

The "numeric drift" rows are latency-only (different host hardware than the reference Linux i5-8250U; the reference percentile means lie within ±25–50% bands per `bench/verify_tolerance.toml`). All `exact_required` structural fields match.

---

## [3.0] - May 8, 2026

### Summary
Major measured-data revision applied to **both** `WHITEPAPER.md` (formerly v2.7) and `WHITEPAPER_ACADEMIC.md` (formerly v3.1-academic, now v3.2-academic). Every "(projected)" label that was previously gated by a missing-runtime caveat is now replaced with a measurement traceable to a JSON file under `bench/results/`. The remaining "we did not measure X" disclaimers are explicit and labeled.

### Added — bench/results/ artifacts cited by both papers
- `bench/results/aether_mock_v1.json` — Aether mock-mode runs (5 trials × 3 configs × 2 datasets), Linux i5-8250U, git `4d16ec5cd5cb0957d7dc6408b5df25ba7befbe9b`, measured 2026-05-03
- `bench/results/langchain_v1.json` — LangChain mock-mode runs (5 trials × 3 configs × 2 datasets), Windows i5 stepping, langchain 0.3.28, git `9591852f9ebb34822dc628197d47cb643c0ac381`, measured 2026-05-04
- `bench/results/dspy_v1.json` — DSPy mock-mode runs (5 trials × 3 configs × 2 datasets), Windows i5 stepping, dspy 2.6.27, git `558df452e1e9c36d35bdbc474369862a631c458c`, measured 2026-05-08
- `bench/results/aether_real_api_v1.json` — Aether real-OpenAI runs (3 trials × 3 configs × 2 datasets, gpt-4o-mini), Linux i5-8250U, git `9c8001ce269920c98fa733a82b3c69ea7352e37e`, cost $0.128887, measured 2026-05-08
- `bench/results/langchain_real_api_v1.json` — LangChain real-OpenAI runs, langchain 0.3.28, cost $0.126498, measured 2026-05-08
- `bench/results/dspy_real_api_v1.json` — DSPy real-OpenAI runs, dspy 2.6.27, cost $0.222963, measured 2026-05-08
- `bench/results/real_api_v1.json` — merged real-API result with cost reconciliation, total cost **$0.478349** under $10.00 budget gate, schema `aether-real-api-v1`
- `bench/results/REAL_API.md` — companion Markdown summary of the real-API run with verbatim merge-step output
- `bench/results/ablation_cache_v1.json` — cache ablation: `no_cache` / `l1_exact_match` / `repeat_warm` × `customer_support_100` + `customer_support_repeat_100`, 5 trials each, git `8aee2cce5f969b5e2d84e94216355354dcc0eb7f`, measured 2026-05-08; produces 0.7000 hit rate (l1) and 1.0000 (warm) on the 70%-repeat workload, with paired BCa Δ p50 = −111.3 ms [−112.3, −108.8]
- `bench/results/ablation_parallel_v1.json` — parallel ablation: `sequential` vs `parallel` × `customer_support_100` + `document_analysis_50`, 5 trials each, git `8aee2cce5f969b5e2d84e94216355354dcc0eb7f`, measured 2026-05-08; produces paired-trial speedup_p50 of 1.4778× [1.4728, 1.4849] and 2.5841× [2.5635, 2.5986]
- `bench/results/ablation_typesafety_v1.json` — type-safety corpus: 30 cases × 4 buckets (10 type_mismatch + 10 undefined_reference + 5 missing_field + 5 duplicate_definition), git `8aee2cce5f969b5e2d84e94216355354dcc0eb7f`, measured 2026-05-08; produces Aether 30/30 caught at compile time, LangChain 17/30 caught at runtime + 13 missed silently, DSPy 17/30 caught at runtime + 13 missed silently
- `bench/results/ablations_v1.md` — companion Markdown summary for the three ablations
- `bench/results/security_v1.json` — InjecAgent-adapted prompt-injection corpus, 20 attack + 20 benign × 3 trials × 3 configs against real `gpt-4o-mini`, git `2759f1e7f5e26eb435f3c98951ea1fcf193f2b5e`, measured 2026-05-08, run cost $0.036912900000000005; produces compile_time_catch_rate = 1.0 (CI degenerate) for `aether_taint_on`, 0.0 for `aether_taint_off` and `langchain_baseline`, with attack_success_rate = 0.0 across all configs (model itself rejected the attacks)
- `bench/results/hotpotqa_aether_v1.json` — Aether HotpotQA dev_500 latency (mock LLM, 3 trials), git `b581653a0611fdc3ddfbaf8af9553593a61cb585`, p50 mean 143.83 ms ± 0.29
- `bench/results/hotpotqa_langchain_v1.json` — LangChain HotpotQA dev_500 latency (mock LLM, 3 trials), git `b581653a0611fdc3ddfbaf8af9553593a61cb585`, p50 mean 108.50 ms ± 0.85
- `bench/results/hotpotqa_dspy_v1.json` — DSPy HotpotQA dev_500 latency (mock LLM, 3 trials), git `b581653a0611fdc3ddfbaf8af9553593a61cb585`, p50 mean 103.75 ms ± 0.18

### Added — paper structure
- `WHITEPAPER_ACADEMIC.md`: new Reproducibility callout (after Abstract), new Section 8.5 (Statistical Methodology), new Section 9.7 (HotpotQA Latency, mock-only with EM/F1 disclaimer), new Section 9.8 (Compile-Time Taint Tracking results), new Section 9.9.4 (Hardware variance threat to validity)
- `WHITEPAPER.md`: new Reproducibility callout (after frontmatter), Section 8.4 fully replaced with measured-results table

### Changed
- `WHITEPAPER_ACADEMIC.md`:
  - Abstract: `(projects 2.7x latency reduction)` → measured speedups 1.4778× and 2.5841× with paired BCa CIs and JSON cite; `(60% cache hit rate improvement)` → 0.7000 / 1.0000; LangChain 91.4 ms / DSPy 68.4 ms claims removed (no JSON source); contribution count updated 4 → 5 with explicit taint-tracking item
  - Section 9.1 summary table: H1/H2/H3 cells replaced with measured outcomes plus a new H4 row for taint tracking; per-row JSON-field source column added
  - Section 9.2 latency analysis: 274/103/58 ms table replaced with two per-dataset tables sourced from `ablation_parallel_v1.json` and the parallel_cached row from `aether_mock_v1.json`
  - Section 9.3 caching performance: 60% / 18,240 / $0.91 row dropped (no JSON source); replaced with two per-dataset tables sourced from `ablation_cache_v1.json` plus an explicit "tokens_saved_total = 0.0 across every cache config" disclaimer
  - Section 9.4 code complexity: stripped (no JSON source for the 253/287/283/312/78/111 cells; matched `.aether` sources for the case-study programs are not committed in the repo); replaced with an explicit "we did not measure" disclaimer
  - Section 9.5 type safety: 50-case (15+12+10+13) table replaced with the actual 30-case corpus from `ablation_typesafety_v1.json`; added the cd→dd substitution methodology note verbatim
  - Section 9.6.5 performance results: replaced with the real-OpenAI run from `aether_real_api_v1.json` and `real_api_v1.json`
  - Section 9.7 (Threats to Validity) renumbered to 9.9; old 9.7.4 (MSVC toolchain) removed; new 9.9.4 documents hardware variance across measurement environments; 9.9.1 (Mock provider bias) updated to point at the real-API data as mitigation
  - Section 11.3 (Security current status): "designed, not yet implemented" line replaced with the measured `compile_time_catch_rate=1.0` result and pointer to Section 9.8
  - Appendix B.1 / B.2 / B.4: Taint tracking marked Implemented with cite; OpenAI Client moved to Implemented & benchmarked with `real_api_v1.json` cite; Benchmark datasets bumped 2 → 5 with each dataset paired with its JSON; CircularDependency variant flagged as defined-but-not-emitted with link to issue #4
  - Section 1.1 contributions: contribution #4 augmented to mention real-API run; new contribution #5 for compile-time taint tracking with `security_v1.json` cite
  - Frontmatter: version `3.1-academic` → `3.2-academic`, date `February 2026` → `May 2026`, status updated
- `WHITEPAPER.md`:
  - Section 8.1 closing line "No empirical results exist yet" replaced with pointer to `bench/results/` and Section 8.4
  - Section 8.4 "Projected Results" → "Measured Results (3.0)" with full JSON-field map
  - Section 1 closing line "All performance projections in this paper are theoretical" replaced with the post-3.0 truth
  - Section 13.1 limitation "Performance claims are projections, not measurements" → updated to reflect 3.0 status
  - Section 15.1 (Document History) "Performance claims remain theoretical projections" → updated to reflect 3.0 status
  - Section 15.3 (Quarterly Review Checklist) "Update projections" → "re-run scripts and refresh JSONs"
  - Frontmatter: version `2.7` → `3.0`, date `February 2026` → `May 2026`, status updated

### Removed
- All `(projected)` labels in `WHITEPAPER_ACADEMIC.md` Sections 9.1, 9.2, 9.3, and the Abstract
- Section 9.7.4 (Aether runtime not executed / MSVC toolchain) — runtime did execute; threat is moot
- Section 8.4 "Projected Results" caveat block in `WHITEPAPER.md` — superseded by measured table
- Section 9.4 LOC table cells (253/287/283/312/78/111) — no JSON source; explicit disclaimer added in their place

### Honest disclaimers (cells we did not measure, called out explicitly in both papers)
- **Behavioral attack-success rate (ASR) reduction.** ASR is 0.0 in every config including the LangChain baseline because `gpt-4o-mini` itself rejected every attempted injection on this 60-case corpus. The differentiating measurement is therefore static `compile_time_catch_rate`, not behavioral ASR delta.
- **HotpotQA accuracy (EM, F1).** Reported as 0.0 / 0.0 verbatim from the JSONs because the run uses a mock LLM. The latency numbers are meaningful; the accuracy numbers are not.
- **Equivalent-functionality LOC for case-study programs.** The earlier 253/287/283/312/78/111 figures had no JSON source and matched `.aether` sources for the case studies are not committed; restoring this measurement is on the follow-up roadmap.
- **Mock-mode token-savings dollar figures.** `tokens_saved_total = 0.0` across every cache config because the mock LLM does not populate `tokens_saved`. Real-API cost is reported in `real_api_v1.json`.

### Acceptance verification
```
$ rg -n "projected" whitepaper/WHITEPAPER_ACADEMIC.md
(no hits — all replaced or stripped)
$ rg -n "projected" whitepaper/WHITEPAPER.md
864:- We did not measure equivalent-functionality LOC. The earlier "60–70% / 40–50%" cells of the projected table came from an extrapolation; …
(only the explicit "we did not measure" disclaimer remains)
$ rg -n "\[MEASURED\]" whitepaper/WHITEPAPER_ACADEMIC.md whitepaper/WHITEPAPER.md
(no hits)
```

---

## [2.7] - February 5, 2026

### Summary
Telemetry layer fully re-enabled with OTLP support. Added criterion-based performance benchmarks for DAG execution. Updated OpenTelemetry dependencies.

### Added
- **Telemetry Re-enablement**:
  - `tracing_opentelemetry::layer()` now fully integrated and working
  - OTLP export support for Jaeger, Zipkin, and other backends
  - Replaced deprecated `opentelemetry-jaeger` with `opentelemetry-otlp`
- **Criterion Performance Benchmarks** (`aether-runtime/benches/runtime_benchmarks.rs`):
  - `execute_simple_dag_sequential`: Sequential A→B→C execution
  - `execute_simple_dag_parallel`: Parallel mode (structure-constrained)
  - `execute_parallel_dag_10_nodes`: 10 independent nodes benchmark
- **Cargo.toml Updates**:
  - Added `criterion = { version = "0.5", features = ["async_tokio"] }` as dev-dependency
  - Added `[[bench]]` configuration for `runtime_benchmarks`

### Changed
- **OpenTelemetry Dependencies**:
  - `tracing-opentelemetry`: 0.22.0
  - `opentelemetry`: 0.21.0 
  - `opentelemetry_sdk`: 0.21.0
  - `opentelemetry-otlp`: 0.14.0
- Section 1 (Executive Summary): Updated status to "Approaching Beta Milestone"
- Telemetry module (`telemetry.rs`): Refactored for OTLP compatibility

### Technical Details
The telemetry initialization now uses `opentelemetry_otlp::new_exporter().tonic()` for OTLP gRPC export, with fallback to `http://localhost:4317` default endpoint. The `TelemetryConfig` maintains backward-compatible `jaeger_agent_endpoint` and `jaeger_collector_endpoint` fields.

---

## [2.6] - February 4, 2026

### Summary
Full benchmark suite with synthetic datasets, Python benchmark runner, provider switching, and CI integration for reproducible performance validation.

### Added
- **Synthetic Datasets** (`bench/datasets/`):
  - `customer_support_100.jsonl`: 100 customer support queries with urgency (Low/Medium/High/Critical) and category (18 categories), including context fields (customer_tier, previous_tickets)
  - `document_analysis_50.jsonl`: 50 documents across 25+ domains for parallel entity extraction and summarization
  - `README.md`: Dataset schema documentation and usage guide
- **Benchmark Runner Script** (`scripts/run_benchmark.py`):
  - Cold/warm/sequential execution modes
  - DAG creation for triage and extraction scenarios
  - Latency percentile computation (p50/p95/p99)
  - Cache statistics and throughput metrics
  - JSON output with baseline integration
  - CLI: `python scripts/run_benchmark.py --all --requests 100 --output results/`
- **Provider Switching**:
  - `AETHER_PROVIDER` environment variable (`mock|openai|anthropic`)
  - `forced_provider: Option<LlmProvider>` field in `LlmConfig`
  - Default impl parses env var at startup
  - `create_client()` respects forced provider over model-based detection
  - Unit tests for provider switching
- **CI Benchmark Workflow** (`.github/workflows/benchmark.yml`):
  - Push/PR triggers with path filtering (`aether-runtime/**`, `scripts/**`)
  - Scheduled runs (2 AM UTC daily)
  - Manual trigger via `workflow_dispatch` with configurable requests and provider
  - PR commenting with benchmark results table
  - Artifact upload with 90-day retention
- **Benchmark Documentation** (`docs/benchmarks.md`):
  - Measured vs projected methodology
  - Benchmark scenarios (customer support triage, document extraction)
  - How to run benchmarks locally and in CI
  - Output format and interpretation guide
  - Troubleshooting section

### Changed
- Section 1 (Executive Summary): Updated current status to include full benchmark suite
- Section 4.2 (Efficiency Through Static Optimization): Expanded with complete benchmark infrastructure details
- Section 7.7 (LLM Provider Interface): Added AETHER_PROVIDER documentation
- Section 8 (Evaluation and Benchmarking): Complete rewrite with implemented infrastructure
- Section 8.2.1 (Datasets): Updated with implemented datasets and schemas
- Section 12.1 (Current Status): Added all new benchmark components
- Section 15.1 (Document History): Added v2.6 entry

### Consolidated
- Removed individual versioned whitepaper files (v2.0-v2.4)
- Single canonical `WHITEPAPER.md` replaces `aether-whitepaper-v2.5-canonical.md`
- Single `CHANGELOG.md` replaces individual `aether-changelog-vX.X.md` files

---

## [2.5] - February 4, 2026

### Summary
Benchmark infrastructure for reproducible validation of performance claims (Section 8 / SC-4, SC-5, SC-6).

### Added
- **Latency Percentile Computation**: Server-side p50/p95/p99 for node and level execution times
  - New fields: `node_latency_p50_ms`, `node_latency_p95_ms`, `node_latency_p99_ms`
  - New fields: `level_latency_p50_ms`, `level_latency_p95_ms`, `level_latency_p99_ms`
  - Floor-based index method on sorted samples
- **Sequential Execution Mode**: `POST /execute?sequential=true` forces `max_concurrency=1`
- **Baseline Comparison Stubs** (`bench/baselines/`):
  - `langchain_baseline.py`: Sequential execution, 15% cache hit simulation
  - `dspy_baseline.py`: Module-based composition, no caching
  - `README.md`: Documentation for running baselines
- **Unit Tests**: 7 new tests for percentile computation

### Changed
- Section 8.1: Changed from "Disclaimer" to "Implementation Status"
- Section 8.3: Updated with baseline stub information
- Section 8.5: Changed from "Planned" to partial implementation

---

## [2.4] - February 4, 2026

### Summary
Runtime MVP implementation with parallel execution, caching, and observability.

### Added
- **Execution Engine** (`main.rs`): Topological sort, level-based parallel execution via JoinSet
- **Context Management** (`context.rs`): ContextStore trait, InMemoryContextStore, ExecutionContext
- **Template Engine** (`template.rs`): `{{context.KEY}}` and `{{node.ID.output}}` substitution
- **Caching** (`cache.rs`): Exact-match LRU cache, CacheKey::from_dag_node, tokens_saved tracking
- **LLM Providers** (`llm.rs`): LlmClient trait, Mock/OpenAI/Anthropic implementations
- **Security** (`security.rs`): Prompt injection detection, InputSanitizer trait
- **Observability** (`telemetry.rs`): OpenTelemetry + Jaeger integration, Prometheus metrics
- **New Types** in `aether-core`:
  - `NodeState` enum: Pending, Running, Succeeded, Failed, Skipped
  - `NodeStatus` struct: state, attempts, error, skip_reason
  - `ErrorPolicy` enum: Fail, Skip, Retry
- **Response Fields**: level_execution_times_ms, max_concurrency_used, node_status, tokens_saved
- **Test Fixtures**: test_dag_context.json, test_dag_cache.json, test_dag_parallel.json, malicious_dag.json

### Changed
- Section 7: Runtime Architecture changed from "Planned" to "Implemented"
- Section 6.1: Runtime row updated from "Partial" to "Implemented"

---

## [2.3] - February 2026

### Summary
End-to-End Demo Loop enabling compile → execute → visualize workflow.

### Added
- **CLI `run` Command**:
  ```bash
  aetherc run <file>                      # Compile and execute
  aetherc run <file> --runtime-url <URL>  # Custom runtime URL
  aetherc run <file> --flow <name>        # Run specific flow
  aetherc run <file> --context '{...}'    # Pass context as JSON
  ```
- **Runtime URL Configuration**: `--runtime-url` > `AETHER_RUNTIME_URL` > default
- **DAG Visualizer Enhancements**:
  - Hierarchical layout using dagre.js
  - File loading via drag-and-drop or file picker
  - Color-coded execution status (green=cached, blue=executed, red=failed, gray=skipped)
  - Node details panel with error information
- **Sample Files**: sample_dag.json, sample_execution_result.json, sample_mixed_status_result.json
- **Dependencies**: reqwest (HTTP client), url (URL parsing), @dagrejs/dagre

### Changed
- Section 11.1: Added `run` command documentation
- Section 11.3: New DAG Visualizer section

---

## [2.2] - February 2026

### Summary
Type System MVP with comprehensive semantic analysis.

### Added
- **5-Pass Semantic Analyzer** (~2,100 lines):
  1. Symbol Collection: Type definitions and function signatures
  2. Type Internals Validation: Duplicate field/variant detection
  3. LLM Function Validation: model/prompt presence, template refs
  4. Flow Analysis: Forward-only type inference, call validation
  5. Function Analysis: Regular function body validation
- **Type System Features**:
  - Forward-only type inference for all expressions
  - Template validation: `{{variable}}`, `{{context.KEY}}`, `{{node.ID.output}}`
  - Duplicate detection for all symbol types
  - Call validation with argument count and type checking
  - Return type verification
- **Error Infrastructure**:
  - Error accumulation (up to 10 errors)
  - Source locations (line, column)
  - "Did you mean?" suggestions (Levenshtein distance)
  - 15+ distinct error types
- **Cycle Detection**: Kahn's algorithm topological sort
- **Unit Tests**: 25+ tests covering semantic rules

### Changed
- Section 6.2.3: Completely rewritten with 5-pass analyzer details
- Section 6.2.4: Added cycle detection to IR description

---

## [2.1] - February 2026

### Summary
Phase 1 of compiler complete with all core components implemented.

### Added
- **Parser**: Full recursive descent parser (~1900 lines)
  - All language constructs: `llm fn`, `flow`, `struct`, `enum` with associated data
  - String templates with `{{variable}}` interpolation
- **Semantic Analysis**: Symbol table, type collection, validation
- **Code Generator**: AST to DAG JSON with template_refs metadata
- **CLI**: `aetherc compile`, `check`, `parse` commands (clap)
- **Shared Types** (`aether-core`): Dag, DagNode, TemplateRef, ExecutionHints, RenderPolicy

### Changed
- Section 6.1: Implementation status table updated
- Section 12.2: Phase 1 marked as complete

---

## [2.0] - February 2026

### Summary
Major revision with research update and restructured whitepaper.

### Added
- **Section 3: Related Work** with 2024-2026 research:
  - LangChain/LangGraph 1.0, DSPy (ICLR 2024), BAML, CrewAI
  - LangSmith, DeepEval v3.0, RAGAS evaluation frameworks
  - OWASP LLM Top 10 2025, StruQ/SecAlign security research
  - MCP and A2A protocol specifications
- **Section 10: Security Model** with threat model, taint tracking design
- **Section 13: Limitations** with open research questions
- **Measurable Success Criteria**: SC-1 through SC-12
- **30 authoritative citations** (up from 25)

### Changed
- **Section 1**: Rewritten with explicit "Current Status" disclaimer
- **Section 2**: Restructured around 5 engineering failures with "Measurable problem" statements
- **Section 4**: Renamed to "Design Goals and Measurable Success Criteria"
- **Section 8**: Rewritten as "Evaluation Methodology" with benchmark suite design
- **Section 14**: Rewritten to be honest about prototype status

### Removed
- Section 13 (old): Formal Operational Semantics (incomplete)
- Unsupported performance claims
- Marketing language

### Fixed
- UTF-8 encoding corruption (em-dashes, smart quotes)
- Section numbering (was 1,2,4-9,11-13,16-18 → now sequential 1-17)
- Unclosed Figure 4 caption

---

## [1.0] - July 2025

### Summary
Initial whitepaper draft: "Designing a Programming Language for Efficient LLM Integration"

---

## Maintenance

### Versioning Scheme
- **Major (X.0)**: Fundamental design changes, significant restructuring
- **Minor (X.Y)**: New sections, expanded features, implementation updates
- **Patch (X.Y.Z)**: Typo fixes, clarifications, broken link fixes

### Quarterly Review Checklist
- [ ] Check for LangChain/LangGraph releases
- [ ] Check for DSPy updates
- [ ] Check for new BAML features
- [ ] Update benchmark results
- [ ] Review security advisories (OWASP LLM Top 10)
- [ ] Verify all external links
