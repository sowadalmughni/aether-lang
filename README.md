# Aether Programming Language

[![CI](https://github.com/aether-lang/aether/actions/workflows/ci.yml/badge.svg)](https://github.com/aether-lang/aether/actions/workflows/ci.yml)
[![Benchmarks](https://github.com/aether-lang/aether/actions/workflows/benchmark.yml/badge.svg)](https://github.com/aether-lang/aether/actions/workflows/benchmark.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducible Artifacts](https://img.shields.io/badge/Artifacts-Reproducible-green)](Dockerfile)

A programming language designed for efficient, reliable, and scalable Large Language Model (LLM) integration.

## 🚀 Overview

Aether transforms LLM integration from fragile scripting into robust, engineered systems. By providing first-class abstractions for LLM orchestration, intelligent caching, and type-safe prompt management, Aether enables developers to build production-grade AI applications with confidence.

**Current Status**: v3.0.2 — single canonical paper, pandoc/docx pipeline. Every benchmark claim previously labelled "projected" still resolves to a JSON file under [`bench/results/`](bench/results/) (no measurement-data delta from v3.0). Whitepaper: [`whitepaper/aether.docx`](whitepaper/aether.docx) (canonical), markdown source at [`whitepaper/WHITEPAPER_ACADEMIC.md`](whitepaper/WHITEPAPER_ACADEMIC.md). The prior LaTeX/PDF pipeline is preserved at [`whitepaper/archive/latex/`](whitepaper/archive/latex/) for reference.

### Key Features

- 🔧 **Type-Safe LLM Functions**: 30/30 intentionally malformed programs caught at compile time vs 17/30 by LangChain and DSPy — [`bench/results/ablation_typesafety_v1.json`](bench/results/ablation_typesafety_v1.json)
- 🌊 **Declarative Flows**: DAG-based execution with level-parallel scheduling
- ⚡ **Intelligent Caching**: L1 hit rate 0.7000 on 70%-repeat workloads, 1.0000 warm-cache — [`bench/results/ablation_cache_v1.json`](bench/results/ablation_cache_v1.json)
- 🛡️ **Security**: 100% compile-time catch rate on 60-case InjecAgent-adapted corpus — [`bench/results/security_v1.json`](bench/results/security_v1.json)
- 🚀 **Parallel DAG Execution**: 1.4778× and 2.5841× measured speedup vs sequential on synthetic benchmarks — [`bench/results/ablation_parallel_v1.json`](bench/results/ablation_parallel_v1.json)
- 📊 **Observability**: Prometheus metrics, OpenTelemetry tracing, latency percentiles (p50/p95/p99)
- 🔄 **Reproducible**: Single-command `bash reproduce.sh` + Dockerized build — see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

## 📖 Quick Start

The recommended path is Docker — single command, no host toolchain setup, mirrors CI.

### Prerequisites

- Docker 24+ (Engine or Desktop)

### Build, run the runtime, and run the benchmark

```bash
git clone https://github.com/aether-lang/aether.git
cd aether
docker build -t aether .
docker compose up --build runtime          # runtime on :3000
docker compose run --rm bench              # mock-mode benchmark, JSON to bench/results/
docker compose down
```

For native Linux instructions and trade-offs see [`docs/BUILD.md`](docs/BUILD.md). Native Windows (MSVC) is not supported — use WSL2 or Docker Desktop.

### Example: Customer Support Agent

```aether
llm fn triage_agent(query: string) -> UrgencyLevel {
    model: "gpt-4o",
    prompt: "Classify the urgency of this query as Low, Medium, High, or Critical: {{query}}"
}

llm fn knowledge_agent(query: string) -> string {
    model: "gpt-4o", 
    prompt: "Answer this question from our knowledge base: {{query}}"
}

flow handle_support_query(query: string) -> string {
    let urgency = triage_agent(query: query);
    
    if urgency == UrgencyLevel.Critical {
        return escalate_to_human(query: query);
    } else {
        return knowledge_agent(query: query);
    }
}
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Aether Code   │───▶│ Aether Compiler │───▶│    DAG JSON     │
│   (.aether)     │    │   (aetherc)     │    │  (optimized)   │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Type Checking  │    │ Aether Runtime │
                       │  & Validation   │    │  (Axum + Tokio)│
                       └─────────────────┘    └────────┬────────┘
                                                │
                       ┌─────────────────────────▼─────────────┐
                       │  • Level-based parallel execution           │
                       │  • Exact-match LRU caching                   │
                       │  • Template substitution (context/node)      │
                       │  • Prometheus metrics + OpenTelemetry        │
                       │  • Latency percentiles (p50/p95/p99)         │
                       └─────────────────────────┬─────────────┘
                                                │
                                                ▼
                       ┌────────────────────────────────────────┐
                       │ LLM Providers (Mock/OpenAI/Anthropic)      │
                       └────────────────────────────────────────┘
```

## 🛠️ Components

### Compiler (`aether-compiler/`)
- **Status**: ✅ Phase 1 Complete
- **Features**: Lexer, parser, semantic analysis, type checking, DAG code generation
- **CLI**: `aetherc compile`, `check`, `parse`, `run` commands
- **Output**: DAG JSON with dependency computation and template metadata

### Runtime (`aether-runtime/`)
- **Status**: ✅ MVP Complete
- **Features**: Level-based parallel execution, exact-match caching, template substitution, error policies
- **Observability**: Prometheus metrics, OpenTelemetry tracing, latency percentiles (p50/p95/p99)
- **Endpoints**: `/execute`, `/cache/stats`, `/cache/clear`, `/metrics`, `/health`

### Core Types (`aether-core/`)
- **Status**: ✅ Implemented
- **Shared**: `Dag`, `DagNode`, `TemplateRef`, `ExecutionHints`, `NodeState`, `NodeStatus`, `ErrorPolicy`

### DAG Visualizer (`aether-dag-visualizer/`)
- **Status**: ✅ Implemented
- **Features**: Hierarchical layout (dagre.js), drag-and-drop file loading, execution status colors

### Benchmark Suite (`bench/`)
- **Status**: ✅ Infrastructure Implemented
- **Baselines**: Real LangChain (0.3.28) and DSPy (2.6.27) baseline scripts; mock and real-provider modes (see [`bench/baselines/README.md`](bench/baselines/README.md))

## 📊 Performance

### Measured Results

Every row below has a JSON source file in [`bench/results/`](bench/results/). Reproduce with `bash reproduce.sh` (mock mode, ~95–110 min on the reference hardware in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).

| Metric | Aether | Baseline | Source |
|--------|--------|----------|--------|
| **Type-safety catch rate** | 30/30 (100%) compile time | LangChain 17/30 runtime, DSPy 17/30 runtime | [ablation_typesafety_v1.json](bench/results/ablation_typesafety_v1.json) |
| **L1 cache hit rate** (70%-repeat workload) | 0.7000 | n/a | [ablation_cache_v1.json](bench/results/ablation_cache_v1.json) |
| **L1 cache hit rate** (warm) | 1.0000 | n/a | [ablation_cache_v1.json](bench/results/ablation_cache_v1.json) |
| **Compile-time security catch** (60 InjecAgent-adapted cases) | 100% | n/a | [security_v1.json](bench/results/security_v1.json) |
| **Parallel speedup**, customer_support_100 (p50) | 1.4778× | 1.0× sequential | [ablation_parallel_v1.json](bench/results/ablation_parallel_v1.json) |
| **Parallel speedup**, document_analysis_50 (p50) | 2.5841× | 1.0× sequential | [ablation_parallel_v1.json](bench/results/ablation_parallel_v1.json) |

### Benchmark Infrastructure

The runtime provides server-side measurement for reproducible benchmarking:

| Feature | Endpoint | Description |
|---------|----------|-------------|
| **Latency Percentiles** | `/execute` | p50/p95/p99 for node and level execution |
| **Sequential Mode** | `/execute?sequential=true` | Forces serial execution for ablation studies |
| **Cache Statistics** | `/cache/stats` | Hit rate, tokens saved, entry count |
| **Prometheus Metrics** | `/metrics` | Counters, histograms, gauges |

### Baseline Comparisons

Baselines in [`bench/baselines/`](bench/baselines/) use real, pinned packages — not simulations:

- **LangChain pattern**: real `langchain` 0.3.28, LCEL chains + `InMemoryCache` (LRU exact-match)
- **DSPy pattern**: real `dspy` 2.6.27, `Module` + `Predict` + `ChainOfThought`

```bash
# Run baseline benchmarks (mock mode, no API keys needed)
python bench/baselines/langchain_baseline.py --requests 100
python bench/baselines/dspy_baseline.py --requests 100
```

## 🔒 Security Features

**Implemented:**
- **Compile-time taint tracking**: untrusted-input isolation enforced by the compiler (100% catch rate on the 60-case InjecAgent-adapted corpus, [`security_v1.json`](bench/results/security_v1.json))
- **Prompt Injection Detection**: pattern-based detection of jailbreak attempts (DAN mode, ignore instructions, etc.)
- **InputSanitizer Trait**: pluggable sanitization strategies
- **Blacklist Patterns**: configurable pattern matching for malicious prompts

**Planned:**
- Integration with profanity filtering APIs
- Declarative guardrail annotations (`@input_guard`, `@output_guard`)
- Tool access control with explicit permission grants

## 🧪 Testing & Quality Assurance

```bash
# Run all tests
cargo test --workspace

# Run compiler tests (includes snapshot tests)
cd aether-compiler && cargo test

# Run runtime tests (includes integration tests)
cd aether-runtime && cargo test

# Run baseline benchmarks (mock mode)
python bench/baselines/langchain_baseline.py --requests 100
python bench/baselines/dspy_baseline.py --requests 100

# Run with sequential mode for ablation
curl -X POST "http://localhost:3000/execute?sequential=true" -d @dag.json

# Build documentation
cargo doc --workspace --no-deps
```

## 🔁 Reproducing the paper's numbers

```bash
bash reproduce.sh
```

Mock-mode reproduction takes ~95–110 minutes on the reference hardware (Intel i5-8250U, 8 GB RAM, Ubuntu 24.04). Output JSONs are written to `bench/results/repro/` and diffed against the committed reference set; the diff report is `bench/results/repro/DIFF_REPORT.md`. Real-API JSONs (`security_v1.json`, `*_real_api_v1.json`) require `OPENAI_API_KEY` and incur cost; reproduction instructions for those are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §6.

## 📈 Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: Core Compiler** | ✅ Complete | Parser, semantic analysis, type checking, DAG generation |
| **Phase 2: Runtime MVP** | ✅ Complete | Parallel execution, caching, template engine, observability |
| **Phase 2.5: Benchmarks** | ✅ Complete | Latency percentiles, sequential mode, baseline harness, criterion benchmarks |
| **Phase 2.7: Telemetry** | ✅ Complete | OTLP tracing re-enabled, OpenTelemetry 0.21.0 integration |
| **Phase 3: Measurement & Whitepaper (v3.0 / v3.2-academic)** | ✅ Complete | End-to-end measured ablations, real-API runs, reproducibility pipeline |
| **Phase 4: Advanced Caching** | 🔄 In Progress | Semantic cache, provider prefix caching |
| **Phase 5: IDE Tooling** | 📋 Planned | VS Code extension, LSP support |
| **Phase 6: Production** | 📋 Planned | Multi-level caching, expanded guardrails, broader taint coverage |

See [roadmap.gantt.yml](roadmap.gantt.yml) and [`whitepaper/WHITEPAPER_ACADEMIC.md`](whitepaper/WHITEPAPER_ACADEMIC.md) for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](GOVERNANCE.md).

### Development Setup

```bash
# Using Docker (recommended)
docker build -t aether .
docker run -p 3000:3000 -p 5173:5173 aether
```

For native Linux setup (rustup, Node 20, pnpm 10.4.1, Python venv, build/test/lint commands) see [`docs/BUILD.md`](docs/BUILD.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [React Flow](https://reactflow.dev/) for visualization
- Security patterns inspired by [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- Security corpus adapted from [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (see `bench/security/data/` for the vendored subset)

## 📚 Resources

- **Paper (canonical)**: [`whitepaper/aether.docx`](whitepaper/aether.docx) — built via `make -C whitepaper docx`
- **Paper (markdown source)**: [`whitepaper/WHITEPAPER_ACADEMIC.md`](whitepaper/WHITEPAPER_ACADEMIC.md)
- **Figures**: [`whitepaper/figures/`](whitepaper/figures/) (5 PNGs at 300 DPI; data figures have JSON sidecars listing source field paths in `bench/results/`)
- **Reproducibility**: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and `bash reproduce.sh`
- **Build instructions**: [`docs/BUILD.md`](docs/BUILD.md)
- **Baseline benchmarks**: [`bench/baselines/README.md`](bench/baselines/README.md)
- **Changelog**: [`whitepaper/CHANGELOG.md`](whitepaper/CHANGELOG.md)

---

<div align="center">
  <strong>Transforming LLM Integration, One Flow at a Time</strong>
</div>
