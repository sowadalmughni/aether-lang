# Aether Whitepaper Changelog

**Current Version**: 2.7  
**Last Updated**: February 5, 2026  
**Status**: Prototype - Phase 1-3 Complete, Approaching Beta Milestone

---

## Version History

| Version | Date | Status | Summary |
|---------|------|--------|---------|
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
