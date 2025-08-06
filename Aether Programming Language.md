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

### Key Features

- **🔧 Type-Safe LLM Functions**: Define LLM interactions with structured inputs and outputs
- **🌊 Declarative Flows**: Orchestrate complex LLM workflows with automatic optimization
- **⚡ Intelligent Caching**: Multi-layered caching system (exact-match, semantic, predictive)
- **🛡️ Security-First**: Built-in prompt injection protection and content filtering
- **📊 Observability**: Comprehensive metrics, tracing, and debugging tools
- **🔄 Reproducible**: Deterministic builds and artifact evaluation compliance

## 📖 Quick Start

### Prerequisites

- Rust 1.75.0 or later
- Node.js 20.x or later
- Python 3.9+ (for benchmarking)

### Installation

```bash
# Clone the repository
git clone https://github.com/aether-lang/aether.git
cd aether

# Build the runtime
cd aether-runtime
cargo build --release

# Start the runtime
cargo run --release
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
│   Aether Code   │───▶│  Aether Compiler │───▶│   Optimized     │
│   (.aether)     │    │                 │    │   DAG Runtime   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Type Checking  │    │  LLM Providers  │
                       │  & Validation   │    │  (OpenAI, etc.) │
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ Components

### Runtime & Execution Engine
- **Location**: `aether-runtime/`
- **Description**: Async runtime with DAG execution, caching, and observability
- **Documentation**: [Rustdoc](https://aether-lang.github.io/aether/rustdoc/aether_runtime/)

### DAG Visualizer
- **Location**: `aether-dag-visualizer/`
- **Description**: Interactive React-based visualization of execution graphs
- **Demo**: [Live Demo](https://aether-lang.github.io/aether/)

### Benchmark Suite
- **Location**: `bench/`
- **Description**: Performance benchmarking using lm-evaluation-harness
- **Results**: [Nightly Benchmarks](https://github.com/aether-lang/aether/actions/workflows/benchmark.yml)

## 📊 Performance

Based on our customer support case study:

| Metric | Baseline Python | Aether | Improvement |
|--------|----------------|--------|-------------|
| **Latency (Cache Miss)** | 2500ms | 1500ms | 40% faster |
| **Latency (Cache Hit)** | 2500ms | 50ms | 98% faster |
| **Cost (30% Cache Hit)** | $15.00/1k queries | $10.50/1k queries | 30% savings |
| **Cost (60% Cache Hit)** | $15.00/1k queries | $6.00/1k queries | 60% savings |

*Note: These are projected benchmarks. Empirical validation is ongoing.*

## 🔒 Security Features

- **Prompt Injection Protection**: Automatic detection of malicious prompt patterns
- **Content Filtering**: Integration with profanity and safety filters
- **Input Sanitization**: Configurable sanitization pipeline
- **Audit Trails**: Comprehensive logging of all LLM interactions

## 🧪 Testing & Quality Assurance

```bash
# Run unit tests
cd aether-runtime && cargo test

# Run benchmarks
cd bench/lm-evaluation-harness
python -m lm_eval --model aether_stub --tasks hellaswag --limit 10

# Build documentation
cargo doc --no-deps
```

## 📈 Roadmap

- **Q3 2025**: Complete parser and semantic analysis
- **Q4 2025**: Full runtime with LLM integration
- **Q1 2026**: Advanced caching and optimization
- **Q2 2026**: Production tooling and IDE support

See [roadmap.gantt.yml](roadmap.gantt.yml) for detailed timeline.

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

- **Whitepaper**: [Designing a Programming Language for Efficient LLM Integration](docs/whitepaper.pdf)
- **API Documentation**: [Rustdoc](https://aether-lang.github.io/aether/rustdoc/)
- **Live Demo**: [DAG Visualizer](https://aether-lang.github.io/aether/)
- **Benchmarks**: [Performance Results](https://github.com/aether-lang/aether/actions/workflows/benchmark.yml)

---

<div align="center">
  <strong>Transforming LLM Integration, One Flow at a Time</strong>
</div>

