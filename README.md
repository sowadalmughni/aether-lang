# Aether Programming Language

[![Build Status](https://github.com/aether-lang/aether/workflows/CI/badge.svg)](https://github.com/aether-lang/aether/actions)
[![Documentation](https://img.shields.io/badge/docs-rustdoc-blue)](https://aether-lang.github.io/aether/rustdoc/aether_runtime/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducible Artifacts](https://img.shields.io/badge/Artifacts-Reproducible-green)](https://github.com/aether-lang/aether/blob/main/Dockerfile)
[![DAG Visualizer](https://img.shields.io/badge/Demo-DAG_Visualizer-purple)](https://aether-lang.github.io/aether/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-Nightly-orange)](https://github.com/aether-lang/aether/actions/workflows/benchmark.yml)

A programming language designed for efficient, reliable, and scalable Large Language Model (LLM) integration.

## 🚀 Overview

Aether transforms LLM integration from fragile scripting into robust, engineered systems. By providing first-class abstractions for LLM orchestration, intelligent caching, and type-safe prompt management, Aether enables developers to build production-grade AI applications with confidence.

**Current Status**: Prototype - Phase 1 Complete, Runtime MVP with Benchmark Infrastructure (v2.5)

### Key Features

- **🔧 Type-Safe LLM Functions**: Define LLM interactions with structured inputs and outputs ✓
- **🌊 Declarative Flows**: Orchestrate complex LLM workflows with DAG-based execution ✓
- **⚡ Intelligent Caching**: Exact-match LRU cache with tokens_saved tracking ✓
- **🛡️ Security-First**: Prompt injection detection with pattern-based filtering ✓
- **📊 Observability**: Prometheus metrics, OpenTelemetry tracing, latency percentiles ✓
- **🔄 Reproducible**: Deterministic builds with benchmark infrastructure ✓

## 📖 Quick Start

### Prerequisites

- Rust 1.75.0 or later
- Node.js 20.x or later (for DAG Visualizer)
- Python 3.9+ (for baseline benchmarks)

### Installation

```bash
# Clone the repository
git clone https://github.com/aether-lang/aether.git
cd aether

# Build the compiler and runtime
cargo build --release

# Start the runtime server (default: http://127.0.0.1:3000)
cd aether-runtime && cargo run --release

# In another terminal, compile and run an Aether program
cd aether-compiler && cargo run --release -- run ../examples/hello.aether
```

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
- **Demo**: [Live Demo](https://aether-lang.github.io/aether/)

### Benchmark Suite (`bench/`)
- **Status**: ✅ Infrastructure Implemented
- **Baselines**: LangChain and DSPy simulation stubs with mock mode
- **Harness**: lm-evaluation-harness integration

## 📊 Performance

### Benchmark Infrastructure (Implemented)

The runtime provides server-side measurement for reproducible benchmarking:

| Feature | Endpoint | Description |
|---------|----------|-------------|
| **Latency Percentiles** | `/execute` | p50/p95/p99 for node and level execution |
| **Sequential Mode** | `/execute?sequential=true` | Forces serial execution for ablation studies |
| **Cache Statistics** | `/cache/stats` | Hit rate, tokens saved, entry count |
| **Prometheus Metrics** | `/metrics` | Counters, histograms, gauges |

### Baseline Comparisons

Baseline stubs in `bench/baselines/` allow direct comparison with:
- **LangChain pattern**: Sequential execution, 15% cache hit simulation
- **DSPy pattern**: Module-based composition, no caching

```bash
# Run baseline benchmarks (mock mode, no API keys needed)
python bench/baselines/langchain_baseline.py --requests 100
python bench/baselines/dspy_baseline.py --requests 100
```

### Projected Results

| Metric | Python+LangChain | Aether (Projected) | Basis |
|--------|------------------|-------------------|-------|
| **Latency (parallel)** | 100% (baseline) | 60-70% | Static parallelization |
| **Cache hit rate** | 10-20% (manual) | 40-60% | Compiler-assisted |
| **Schema errors** | 5-15% runtime | <1% | Compile-time typing |

*Note: These are projected benchmarks. Empirical validation infrastructure is implemented.*

## 🔒 Security Features

**Implemented:**
- **Prompt Injection Detection**: Pattern-based detection of jailbreak attempts (DAN mode, ignore instructions, etc.)
- **InputSanitizer Trait**: Pluggable sanitization strategies
- **Blacklist Patterns**: Configurable pattern matching for malicious prompts

**Planned:**
- Compile-time taint tracking for untrusted input isolation
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

## 📈 Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: Core Compiler** | ✅ Complete | Parser, semantic analysis, type checking, DAG generation |
| **Phase 2: Runtime MVP** | ✅ Complete | Parallel execution, caching, template engine, observability |
| **Phase 2.5: Benchmarks** | ✅ Complete | Latency percentiles, sequential mode, baseline stubs |
| **Phase 3: Advanced Caching** | 🔄 In Progress | Semantic cache, provider prefix caching |
| **Phase 4: IDE Tooling** | 📋 Planned | VS Code extension, LSP support |
| **Phase 5: Production** | 📋 Planned | Multi-level caching, guardrails, taint tracking |

See [roadmap.gantt.yml](roadmap.gantt.yml) and [whitepaper](whitepaper/WHITEPAPER.md) for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](GOVERNANCE.md).

### Development Setup

```bash
# Using Docker (recommended)
docker build -t aether .
docker run -p 3000:3000 -p 5173:5173 aether

# Or local development
./scripts/setup-dev.sh
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [React Flow](https://reactflow.dev/) for visualization
- Benchmarking powered by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Security patterns inspired by [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

## 📚 Resources

- **Whitepaper**: [Aether: A Domain-Specific Language for Type-Safe LLM Orchestration](whitepaper/WHITEPAPER.md)
- **Changelog**: [Version History](whitepaper/CHANGELOG.md)
- **API Documentation**: [Rustdoc](https://aether-lang.github.io/aether/rustdoc/)
- **Live Demo**: [DAG Visualizer](https://aether-lang.github.io/aether/)
- **Baseline Benchmarks**: [bench/baselines/README.md](bench/baselines/README.md)

---

<div align="center">
  <strong>Transforming LLM Integration, One Flow at a Time</strong>
</div>

