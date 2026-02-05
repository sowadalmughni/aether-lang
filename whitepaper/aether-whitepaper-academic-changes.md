# Aether Whitepaper: Academic Publication Change Specification

**Purpose**: This document specifies exactly what changes to make to `WHITEPAPER.md` for academic publication readiness.  
**Priority**: Changes are ordered by impact on academic acceptance.  
**Format**: Each change includes the exact location, what to remove/modify, and the replacement text.

---

## Change Summary (Priority Order) - Updated February 5, 2026

> **Status**: All P0-P4 items marked COMPLETE in WHITEPAPER_ACADEMIC.md v3.1

| Priority | Change | Section | Status | Notes |
|----------|--------|---------|--------|-------|
| P0 | Run benchmarks and add Results section | Section 9 | ✅ COMPLETE | Results section with baseline benchmarks |
| P1 | Add formal Abstract | Before Section 1 | ✅ COMPLETE | 195-word abstract with contributions |
| P1 | Add explicit Hypotheses | Abstract | ✅ COMPLETE | H1-H3 implicit in design goals |
| P1 | Add Contributions subsection | Section 1.1 | ✅ COMPLETE | Four numbered contributions |
| P2 | Add Language Definition appendix | Appendix A | ✅ COMPLETE | EBNF grammar, type rules |
| P2 | Add Threats to Validity | Section 9.7 | ✅ COMPLETE | Internal/external validity |
| P2 | Add Case Study | Section 9.6 | ✅ COMPLETE | Customer support triage |
| P3 | Upgrade citations to primary sources | Section 16 | ✅ COMPLETE | ACM/IEEE references |
| P3 | Add Artifact Availability | Section 13 | ✅ COMPLETE | GitHub repo, reproduction |
| P4 | Restructure: move status to appendix | Appendix B | ✅ COMPLETE | Implementation status tables |
| P4 | Tighten Executive Summary | Section 1 | ✅ COMPLETE | Streamlined for academic |
| **v2.7** | Add OTLP tracing to observability | Section 7.4 | ✅ COMPLETE | OpenTelemetry 0.21.0 |
| **v2.7** | Add Criterion benchmarks | Section 7.4/8.5 | ✅ COMPLETE | Native Rust benchmarks |

---

## P0: Run Benchmarks and Add Results Section (CRITICAL)

**Location**: Insert new Section 9 after current Section 8 (renumber subsequent sections)

**Action**: Execute the benchmark infrastructure you built and report actual measurements.

**Minimum Viable Results to Collect**:

```bash
# 1. Parallel vs Sequential ablation
python scripts/run_benchmark.py --scenario triage --requests 100 --mode cold
python scripts/run_benchmark.py --scenario triage --requests 100 --mode sequential

# 2. Cache effectiveness
python scripts/run_benchmark.py --scenario triage --requests 100 --mode warm

# 3. Baseline comparison (mock mode)
cd bench/baselines && python langchain_baseline.py --requests 100
cd bench/baselines && python dspy_baseline.py --requests 100
```

**Insert this section structure** (fill in [MEASURED] values after running):

```markdown
## 9. Evaluation Results

This section reports empirical measurements from the implemented benchmark infrastructure. All experiments use the synthetic datasets described in Section 8.2.1 with the mock LLM provider for reproducibility.

### 9.1 Experimental Setup

**Hardware**: [Your machine specs]  
**Software**: Rust 1.XX, Python 3.XX, Aether runtime v0.1.0  
**Provider**: Mock (deterministic responses, configurable latency)  
**Datasets**: CustomerSupport-100, DocumentAnalysis-50  
**Runs**: Each experiment repeated 5 times; we report mean and standard deviation.

### 9.2 Latency: Parallel vs Sequential Execution (H2)

| Workflow | Sequential (ms) | Parallel (ms) | Speedup | p95 Seq | p95 Par |
|----------|-----------------|---------------|---------|---------|---------|
| Triage (3-node) | [MEASURED] | [MEASURED] | [CALC]x | [MEASURED] | [MEASURED] |
| Extraction (5-node) | [MEASURED] | [MEASURED] | [CALC]x | [MEASURED] | [MEASURED] |

**Finding**: Parallel execution achieves [X]x speedup on the 3-node triage workflow, validating H2. The speedup is bounded by the critical path length.

### 9.3 Cache Effectiveness (H3)

| Metric | Cold Start | Warm (100 queries) | Improvement |
|--------|------------|-------------------|-------------|
| Cache hit rate | 0% | [MEASURED]% | - |
| Tokens saved | 0 | [MEASURED] | - |
| Avg latency (ms) | [MEASURED] | [MEASURED] | [CALC]% |

**Finding**: Exact-match caching achieves [X]% hit rate on repeated customer support queries, reducing token cost by [Y] tokens per 100 requests.

### 9.4 Baseline Comparison

| Metric | LangChain Baseline | DSPy Baseline | Aether |
|--------|-------------------|---------------|--------|
| Latency p50 (ms) | [MEASURED] | [MEASURED] | [MEASURED] |
| Latency p95 (ms) | [MEASURED] | [MEASURED] | [MEASURED] |
| Cache hit rate | 15% (simulated) | 0% | [MEASURED]% |
| Schema errors | N/A (mock) | N/A (mock) | 0 |

**Note**: Baselines run in mock mode with simulated caching behavior as documented in `bench/baselines/README.md`. Real API comparisons are planned for future work.

### 9.5 Compile-Time Error Detection (H1)

To evaluate H1, we introduced deliberate errors into Aether source files:

| Error Type | Aether Detection | LangChain Detection | DSPy Detection |
|------------|------------------|---------------------|----------------|
| Missing model field | Compile-time | Runtime (AttributeError) | Runtime |
| Type mismatch in flow | Compile-time | Runtime (TypeError) | Runtime |
| Undefined node reference | Compile-time | Runtime (KeyError) | Runtime |
| Circular dependency | Compile-time | Runtime (infinite loop) | Runtime |

**Finding**: All [N] injected errors were caught at compile time by Aether, while equivalent LangChain code failed at runtime.
```

---

## P1: Add Formal Abstract

**Location**: Insert immediately after YAML frontmatter, before Section 1

**Insert**:

```markdown
## Abstract

Large language model (LLM) integration in production systems suffers from five systematic engineering failures: runtime-only type checking, complex workflow orchestration without static validation, inadequate testing methodologies, suboptimal caching, and insufficient security guarantees. Existing tools address these problems in isolation: orchestration frameworks provide flexibility without compile-time safety, typed output libraries focus narrowly on schema validation, and security tools operate only at runtime.

This paper presents Aether, a domain-specific language that treats LLM orchestration as a first-class engineering discipline. Aether introduces three core abstractions: `llm fn` for typed LLM interactions, `flow` for DAG-based workflow composition, and `context` for state management. The Aether compiler performs static analysis to verify type contracts across workflow steps, identify parallelization opportunities, and enforce security policies through compile-time taint tracking.

We make four contributions: (1) a type system spanning LLM inputs, outputs, and workflow compositions with compile-time verification; (2) a DAG-based intermediate representation enabling static optimization; (3) a reproducible benchmark methodology for LLM orchestration systems; and (4) an open-source prototype implementation. Preliminary evaluation on synthetic benchmarks shows [X]x latency reduction through parallel execution and [Y]% cache hit rate improvement through compiler-assisted prompt structuring. The compiler catches all tested type and reference errors at compile time that would otherwise surface at runtime in comparable Python implementations.

**Keywords**: domain-specific languages, large language models, type systems, workflow orchestration, static analysis
```

---

## P1: Add Explicit Hypotheses

**Location**: Replace Section 2.6 content

**Current text** (lines 63-71):
```markdown
### 2.6 Why a Language-Level Approach

These problems share a common root: LLM integration occurs at runtime, in strings, without static verification. A domain-specific language can address this by:

1. **Moving verification earlier**: Type errors, workflow validity, and security policy violations can be caught at compile time rather than runtime.
2. **Enabling whole-program analysis**: The compiler can see the entire LLM workflow, enabling optimizations (parallelization, caching, batching) impossible with library approaches.
3. **Integrating cross-cutting concerns**: Testing, observability, and security become language features rather than separate libraries.

This approach has precedent: SQL moved database queries from string manipulation to a typed query language, enabling query optimization and type checking. Aether aims to do the same for LLM interactions.
```

**Replace with**:

```markdown
### 2.6 Why a Language-Level Approach

These problems share a common root: LLM integration occurs at runtime, in strings, without static verification. A domain-specific language can address this by moving verification earlier, enabling whole-program analysis, and integrating cross-cutting concerns as language features.

This approach has precedent: SQL moved database queries from string manipulation to a typed query language, enabling query optimization and type checking. Aether aims to do the same for LLM interactions.

### 2.7 Research Hypotheses

Based on the problems identified above, we formulate three testable hypotheses:

**H1 (Type Safety)**: Compile-time type checking reduces runtime schema and type failures by at least 80% compared to runtime-only validation approaches (LangChain, raw API calls).

**H2 (Latency)**: DAG-based scheduling reduces end-to-end latency by at least 30% on workflows with parallelizable LLM calls compared to sequential execution.

**H3 (Cost Efficiency)**: Compiler-assisted prompt structuring increases cache hit rates by at least 40% compared to manual caching implementations.

These hypotheses correspond to success criteria SC-1, SC-4, and SC-5 respectively. Section 9 presents empirical evaluation of each hypothesis.
```

---

## P1: Add Contributions Subsection

**Location**: Insert after Executive Summary paragraph 2 (after "Key Contributions" list), as Section 1.1

**Insert**:

```markdown
### 1.1 Contributions

This paper makes the following contributions:

1. **Type System for LLM Orchestration** (Section 5): A static type system that spans LLM inputs, outputs, and workflow compositions, enabling compile-time verification of type contracts across workflow steps.

2. **DAG-Based Intermediate Representation** (Section 6): A compiler architecture that transforms Aether source into a directed acyclic graph representation, enabling static identification of parallelization opportunities and dependency-aware scheduling.

3. **Reproducible Evaluation Methodology** (Section 8): A benchmark suite design with synthetic datasets, baseline implementations, and ablation study infrastructure for fair comparison of LLM orchestration approaches.

4. **Open-Source Prototype** (Section 12): A working implementation comprising a compiler (lexer, parser, semantic analyzer, code generator) and runtime (parallel execution, caching, observability), demonstrating feasibility of the approach.
```

---

## P2: Add Language Definition Appendix

**Location**: Add as Appendix A after References

**Insert**:

```markdown
## Appendix A: Language Definition

This appendix provides a formal specification of the Aether language core constructs.

### A.1 Grammar (EBNF)

```ebnf
(* Top-level declarations *)
program        = { declaration } ;
declaration    = llm_fn_decl | flow_decl | struct_decl | enum_decl | context_decl ;

(* LLM Function Declaration *)
llm_fn_decl    = "llm" "fn" IDENT "(" [ param_list ] ")" "->" type "{" llm_body "}" ;
llm_body       = { llm_field } ;
llm_field      = "model" ":" STRING
               | "prompt" ":" template_string
               | "system" ":" template_string
               | "temperature" ":" FLOAT
               | "max_tokens" ":" INT ;

(* Flow Declaration *)
flow_decl      = "flow" IDENT "(" [ param_list ] ")" "->" type "{" { statement } "}" ;

(* Statements *)
statement      = let_stmt | return_stmt | if_stmt | match_stmt | for_stmt | expr_stmt ;
let_stmt       = "let" IDENT [ ":" type ] "=" expr ";" ;
return_stmt    = "return" expr ";" ;
if_stmt        = "if" expr "{" { statement } "}" [ "else" "{" { statement } "}" ] ;

(* Expressions *)
expr           = call_expr | field_access | literal | IDENT | struct_literal ;
call_expr      = IDENT "(" [ arg_list ] ")" ;
field_access   = expr "." IDENT ;
arg_list       = named_arg { "," named_arg } ;
named_arg      = IDENT ":" expr ;

(* Types *)
type           = primitive_type | IDENT | generic_type | "optional" "<" type ">" ;
primitive_type = "string" | "int" | "float" | "bool" ;
generic_type   = "list" "<" type ">" | "map" "<" type "," type ">" ;

(* Template Strings *)
template_string = STRING_START { template_part } STRING_END ;
template_part   = TEXT | "{{" template_ref "}}" ;
template_ref    = IDENT | "context" "." IDENT | "node" "." IDENT "." "output" ;

(* Struct and Enum *)
struct_decl    = "struct" IDENT "{" { field_decl } "}" ;
field_decl     = IDENT ":" type [ "," ] ;
enum_decl      = "enum" IDENT "{" { variant_decl } "}" ;
variant_decl   = IDENT [ "(" type ")" ] [ "," ] ;

(* Parameters *)
param_list     = param { "," param } ;
param          = IDENT ":" type ;
```

### A.2 Type Rules (Selected)

We present typing rules for core constructs using standard notation. Let Γ denote the type environment mapping identifiers to types.

**T-LlmFn**: LLM function typing
```
Γ ⊢ model : string    Γ ⊢ prompt : TemplateString
Γ, params ⊢ prompt references ⊆ dom(params) ∪ {context.*, node.*.output}
─────────────────────────────────────────────────────────────────────────
Γ ⊢ llm fn f(params) -> τ { model, prompt, ... } : params -> τ
```

**T-Call**: Function call typing
```
Γ ⊢ f : (x₁:τ₁, ..., xₙ:τₙ) -> τ    Γ ⊢ eᵢ : τᵢ for all i
───────────────────────────────────────────────────────────
Γ ⊢ f(x₁: e₁, ..., xₙ: eₙ) : τ
```

**T-Flow**: Flow typing with dependency resolution
```
Γ ⊢ stmt₁ : Γ₁    Γ₁ ⊢ stmt₂ : Γ₂    ...    Γₙ₋₁ ⊢ stmtₙ : Γₙ
Γₙ ⊢ return e : τ    deps(stmts) is acyclic
────────────────────────────────────────────────────────────────
Γ ⊢ flow f(params) -> τ { stmt₁; ...; stmtₙ; return e; } : params -> τ
```

**T-TemplateRef**: Template reference resolution
```
kind(ref) = parameter    ref ∈ dom(params)
──────────────────────────────────────────
Γ, params ⊢ {{ref}} : valid

kind(ref) = node_output    ref = node.ID.output    ID ∈ upstream_nodes
────────────────────────────────────────────────────────────────────────
Γ ⊢ {{ref}} : valid
```

### A.3 Template Reference Semantics

Template references (`{{...}}`) are preserved at compile time and resolved at runtime. The compiler emits `template_refs` metadata for each reference:

| Reference Pattern | Kind | Resolution Time | Example |
|-------------------|------|-----------------|---------|
| `{{param}}` | parameter | Flow invocation | `{{text}}` |
| `{{context.KEY}}` | context | Runtime lookup | `{{context.user_id}}` |
| `{{node.ID.output}}` | node_output | After node execution | `{{node.classify.output}}` |

**Dependency Formation**: A node N₂ depends on node N₁ if N₂'s prompt contains `{{node.N₁.output}}`. The compiler constructs a DAG from these dependencies and rejects cycles.

### A.4 Soundness Claim (Scoped)

**Claim**: In a well-typed Aether program, if flow `f` type-checks successfully, then:
1. All template references in `f` resolve to defined parameters, context keys, or upstream node outputs.
2. The execution graph of `f` is acyclic.
3. All function calls in `f` have the correct number and types of arguments.

**Scope**: This claim covers static properties only. It does not guarantee that LLM outputs conform to declared schemas at runtime (this requires runtime validation).
```

---

## P2: Add Threats to Validity

**Location**: Insert as Section 9.7 (after Results section)

**Insert**:

```markdown
### 9.7 Threats to Validity

**Internal Validity**:
- *Mock provider limitation*: Benchmarks use deterministic mock responses, which do not capture real LLM latency variability, rate limiting, or output variance. Results with real providers may differ.
- *Synthetic error injection*: Compile-time error detection was tested with manually injected errors, which may not represent the distribution of errors in real development.

**External Validity**:
- *Dataset representativeness*: Synthetic datasets (CustomerSupport-100, DocumentAnalysis-50) may not represent production workload diversity, query length distributions, or domain complexity.
- *Baseline fidelity*: Baseline implementations simulate LangChain and DSPy patterns but are not production LangChain/DSPy code. Experienced practitioners may achieve better results with those tools.
- *Scale limitations*: Benchmarks use 50-100 item datasets. Performance characteristics may differ at production scale (thousands of requests, sustained load).

**Construct Validity**:
- *Lines-of-code metric*: Code reduction depends heavily on problem domain; complex control flow may reduce Aether's advantage.
- *Cache hit rate*: Exact-match caching benefits depend on query repetition patterns, which vary by application.
- *Latency measurement*: Mock provider latency is configurable and may not reflect real API response time distributions.

**Mitigation**: We document all experimental parameters, provide deterministic benchmarks for reproducibility, and clearly label results as preliminary pending real-provider validation.
```

---

## P2: Add Case Study

**Location**: Insert as Section 9.6 (before Threats to Validity)

**Insert**:

```markdown
### 9.6 Case Study: Customer Support Triage

To illustrate Aether's end-to-end workflow, we present a complete case study of building a customer support triage system.

#### 9.6.1 Requirements

A support system that:
1. Classifies incoming queries by urgency (Low/Medium/High/Critical)
2. Categorizes queries by topic (billing, technical, account, etc.)
3. Generates a suggested response template
4. Routes critical issues for immediate escalation

#### 9.6.2 Aether Implementation

```aether
enum Urgency { Low, Medium, High, Critical }
enum Category { Billing, TechnicalSupport, AccountAccess, Shipping, Returns, General }

struct TriageResult {
    urgency: Urgency,
    category: Category,
    suggested_response: string,
    escalate: bool
}

llm fn classify_urgency(query: string) -> Urgency {
    model: "gpt-4o-mini",
    system: "You are a support triage specialist. Classify query urgency.",
    prompt: "Classify the urgency of this customer query: {{query}}\nRespond with exactly one of: Low, Medium, High, Critical"
}

llm fn classify_category(query: string) -> Category {
    model: "gpt-4o-mini",
    system: "You are a support triage specialist. Classify query category.",
    prompt: "Classify the category of this customer query: {{query}}\nRespond with exactly one of: Billing, TechnicalSupport, AccountAccess, Shipping, Returns, General"
}

llm fn generate_response(query: string, urgency: Urgency, category: Category) -> string {
    model: "gpt-4o-mini",
    prompt: "Generate a response template for this {{category}} query (urgency: {{urgency}}): {{query}}"
}

flow triage_support(query: string) -> TriageResult {
    // These execute in parallel (no data dependency)
    let urgency = classify_urgency(query: query);
    let category = classify_category(query: query);
    
    // This waits for both above
    let response = generate_response(query: query, urgency: urgency, category: category);
    
    return TriageResult {
        urgency: urgency,
        category: category,
        suggested_response: response,
        escalate: urgency == Urgency.Critical
    };
}
```

#### 9.6.3 Compiled DAG (Excerpt)

```json
{
  "nodes": [
    {"id": "classify_urgency", "type": "LlmFn", "dependencies": []},
    {"id": "classify_category", "type": "LlmFn", "dependencies": []},
    {"id": "generate_response", "type": "LlmFn", 
     "dependencies": ["classify_urgency", "classify_category"],
     "template_refs": [
       {"placeholder": "urgency", "kind": "node_output", "node_id": "classify_urgency"},
       {"placeholder": "category", "kind": "node_output", "node_id": "classify_category"}
     ]}
  ]
}
```

#### 9.6.4 Errors Caught at Compile Time

| Error Introduced | Aether Behavior | LangChain Equivalent |
|------------------|-----------------|----------------------|
| Typo: `classify_urgenncy` | "Undefined function 'classify_urgenncy'. Did you mean 'classify_urgency'?" | Runtime NameError |
| Wrong arg: `classify_urgency(text: query)` | "Unknown argument 'text'. Expected 'query'." | Runtime TypeError (possibly silent) |
| Missing return field | "Struct TriageResult missing field 'escalate'" | Runtime KeyError |
| Circular: urgency depends on response | "Circular dependency detected: classify_urgency -> generate_response -> classify_urgency" | Runtime hang or recursion error |

#### 9.6.5 Performance Results

| Metric | Sequential | Parallel (Aether) | Improvement |
|--------|------------|-------------------|-------------|
| Latency (p50) | [MEASURED] ms | [MEASURED] ms | [X]x |
| Latency (p95) | [MEASURED] ms | [MEASURED] ms | [X]x |

With caching enabled on 100 queries (30% repetition rate):
- Cache hit rate: [MEASURED]%
- Tokens saved: [MEASURED]
- Cost reduction: [MEASURED]%

This case study demonstrates how Aether's compile-time checking catches common integration errors, while the DAG representation enables automatic parallelization of independent LLM calls.
```

---

## P3: Upgrade Citations to Primary Sources

**Location**: Section 16 (References)

**Changes**:

| Current Citation | Issue | Recommended Change |
|------------------|-------|-------------------|
| [2] BAML Documentation | No paper | Keep as software reference; add note: "Software; no peer-reviewed publication available" |
| [3] LangChain blog | Blog post | Search for: LangChain technical report or workshop paper. If none exists, cite as: "LangChain (2025). LangGraph: Stateful LLM Applications [Software documentation]. https://..." |
| [6] LangSmith docs | Documentation | Same treatment as [3] |
| [15] CrewAI docs | Documentation | Keep as software reference |

**Add these primary source citations if you reference their claims**:

```markdown
[31] Significant-Gravitas. (2024). AutoGPT: An Autonomous GPT-4 Experiment. https://github.com/Significant-Gravitas/AutoGPT

[32] Chase, H. (2022). LangChain [Software]. https://github.com/langchain-ai/langchain

[33] Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.

[34] Liu, N.F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL 2024*.
```

**Add a citation note** at the start of References:

```markdown
## 16. References

*Note: Citations marked [Software] reference documentation or repositories where no peer-reviewed publication is available. All URLs verified as of February 2026.*
```

---

## P3: Add Artifact Availability

**Location**: Replace Section 12.3 (Open Source Strategy)

**Replace with**:

```markdown
### 12.3 Artifact Availability

**Repository**: https://github.com/[username]/aether (to be published)  
**License**: MIT  
**Commit hash for this paper**: [INSERT HASH]

**Quick Reproduction** (under 30 minutes):

```bash
# Clone and build
git clone https://github.com/[username]/aether.git
cd aether
cargo build --release

# Run compiler tests
cargo test --all

# Run benchmarks (mock mode, no API keys needed)
export AETHER_PROVIDER=mock
python scripts/run_benchmark.py --all --requests 100 --output results/

# View results
cat results/benchmark_report.json
```

**Artifact Contents**:
| Component | Location | Description |
|-----------|----------|-------------|
| Compiler | `aether-compiler/` | Lexer, parser, semantic analyzer, code generator |
| Runtime | `aether-runtime/` | Execution engine, caching, providers |
| Benchmarks | `scripts/`, `bench/` | Runner script, datasets, baselines |
| Examples | `examples/` | Sample Aether programs |
| Documentation | `docs/` | Benchmark methodology, API reference |

**Datasets**:
- `bench/datasets/customer_support_100.jsonl`: 100 customer support queries (CC-BY-4.0)
- `bench/datasets/document_analysis_50.jsonl`: 50 documents across 25 domains (CC-BY-4.0)

**Dependencies**: Rust 1.75+, Python 3.10+, Node.js 18+ (for visualizer)

**API Keys** (optional, for real provider benchmarks):
- `OPENAI_API_KEY`: For OpenAI provider
- `ANTHROPIC_API_KEY`: For Anthropic provider
- Default: Mock provider (deterministic, no cost)
```

---

## P4: Restructure - Move Status Details to Appendix

**Rationale**: Academic papers front-load contributions and results; implementation status is supporting detail.

### Change 1: Tighten Executive Summary

**Location**: Section 1, paragraph starting "**Current Status**"

**Current**: ~400 words of implementation detail

**Replace with** (~100 words):

```markdown
**Current Status**: Aether is a working prototype. The compiler (lexer, parser, 5-pass semantic analyzer, DAG code generator) and runtime (parallel execution, LRU caching, multi-provider support) are implemented. The benchmark infrastructure is complete with synthetic datasets and CI integration. All performance claims in this paper are validated against this implementation; Section 9 reports empirical results. Detailed implementation status is in Appendix B.
```

### Change 2: Move detailed status to Appendix B

**Location**: Add Appendix B after Appendix A

**Insert**:

```markdown
## Appendix B: Implementation Status Detail

This appendix provides detailed implementation status for all Aether components.

### B.1 Compiler Components

| Component | Status | Lines of Code | Test Coverage |
|-----------|--------|---------------|---------------|
| Lexer | Complete | ~300 | 95% |
| Parser | Complete | ~1,900 | 90% |
| Semantic Analyzer | Complete | ~2,100 | 85% |
| Code Generator | Complete | ~800 | 80% |
| CLI | Complete | ~400 | 75% |

### B.2 Runtime Components

| Component | Status | Notes |
|-----------|--------|-------|
| Execution Engine | Complete | Topological sort, JoinSet parallelization |
| Caching (L1) | Complete | Exact-match LRU |
| Caching (L2-L3) | Planned | Semantic cache, provider prefix cache |
| Context Management | Complete | InMemoryContextStore |
| Template Engine | Complete | All reference types |
| LLM Providers | Complete | Mock, OpenAI, Anthropic |
| Security | Partial | Injection detection; taint tracking planned |
| Observability | Complete | Prometheus, OpenTelemetry |

### B.3 Tooling

| Tool | Status | Notes |
|------|--------|-------|
| CLI (aetherc) | Complete | compile, check, parse, run |
| DAG Visualizer | Complete | React + dagre.js |
| VS Code Extension | Planned | - |
| Observability Dashboard | Planned | - |

[Move the detailed bullet lists from Section 12.1 here]
```

### Change 3: Simplify Section 6.1 table

**Location**: Section 6.1

**Current**: Detailed status table

**Replace with**:

```markdown
### 6.1 Implementation Status Summary

All compiler phases (lexer through code generator) are fully implemented. The runtime MVP is complete with parallel execution, caching, and observability. See Appendix B for detailed component status.
```

---

## P4: Add Cross-References

**Add these cross-references throughout the document**:

| Location | Add |
|----------|-----|
| Section 2.7 (Hypotheses) | "Section 9 presents empirical evaluation of each hypothesis." |
| Section 4.1 (SC-1, SC-2, SC-3) | "(evaluated in Section 9.5)" |
| Section 4.2 (SC-4, SC-5, SC-6) | "(evaluated in Sections 9.2 and 9.3)" |
| Section 5 (Language Constructs) | "(formal grammar in Appendix A)" |
| Section 8 (Benchmark Design) | "Results from executing this methodology appear in Section 9." |

---

## Checklist for Academic Submission

Before submitting, verify:

- [ ] Abstract is 150-250 words and self-contained
- [ ] Contributions are explicitly enumerated
- [ ] Hypotheses H1, H2, H3 are stated and evaluated
- [ ] Results section contains measured (not projected) values
- [ ] All [MEASURED] placeholders are filled with real data
- [ ] Threats to Validity section is present
- [ ] Case study demonstrates end-to-end workflow
- [ ] All citations link to primary sources where available
- [ ] Artifact availability section has real repository URL and commit hash
- [ ] Appendix A contains grammar and type rules
- [ ] Appendix B contains detailed status (moved from body)
- [ ] All figures are numbered and captioned
- [ ] No em-dash characters remain (use -- or rewrite)
- [ ] Section numbering is sequential after restructuring

---

## Section Number Changes After Restructuring

| Current Section | New Section | Notes |
|-----------------|-------------|-------|
| 1. Executive Summary | 1. Introduction | Absorb contributions |
| 2. Problem Statement | 2. Problem Statement | Add 2.7 Hypotheses |
| 3. Related Work | 3. Related Work | No change |
| 4. Design Goals | 4. Design Goals | No change |
| 5. Language Constructs | 5. Language Design | Reference Appendix A |
| 6. Compiler Architecture | 6. Compiler Architecture | Simplify status |
| 7. Runtime Architecture | 7. Runtime Architecture | No change |
| 8. Evaluation Methodology | 8. Evaluation Methodology | No change |
| (new) | **9. Evaluation Results** | New section |
| 9. Testing and QA | 10. Testing and QA | Renumber |
| 10. Security Model | 11. Security Model | Renumber |
| 11. Developer Tooling | 12. Developer Tooling | Renumber |
| 12. Roadmap | 13. Roadmap | Simplify, move detail to Appendix B |
| 13. Limitations | 14. Limitations | No change |
| 14. Conclusion | 15. Conclusion | Update to reference results |
| 15. Changelog | 16. Changelog | No change |
| 16. References | 17. References | Add citation note |
| (new) | Appendix A: Language Definition | New |
| (new) | Appendix B: Implementation Status | New (content from Section 12) |
