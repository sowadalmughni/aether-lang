# Aether Language: TODO Documentation

**Date**: February 5, 2026  
**Version**: 2.7  
**Status**: Approaching Beta Milestone (July 2026)

---

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer/Parser | ✅ Complete | Full Aether syntax support |
| Semantic Analyzer | ✅ Complete | Type checking, forward-only inference |
| Code Generator | ✅ Complete | DAG JSON output with template_refs |
| Runtime Execution | ✅ Complete | Level-based parallel, JoinSet scheduling |
| Caching (Level 1) | ✅ Complete | Exact-match LRU with tokens_saved |
| Template Engine | ✅ Complete | `{{context.}}`, `{{node..output}}` |
| Error Policies | 🔄 Partial | Fail/Skip done, Retry planned |
| Observability | ✅ Complete | Prometheus + OTLP tracing (v2.7) |
| Security | 🔄 Partial | Injection detection done, taint tracking planned |
| Benchmarks | ✅ Complete | Criterion + Python runner + CI |

---

## Implementation TODOs

### High Priority (Before Beta - July 2026)

#### Runtime
- [ ] **Retry Policy Execution**: Implement `RetryConfig` with exponential backoff
- [ ] **Streaming Responses**: SSE for long-running executions
- [ ] **Connection Pooling**: Reuse HTTP clients for LLM APIs
- [ ] **Rate Limiting**: Per-provider request throttling

#### Security
- [ ] **Compile-time Taint Tracking**: Track data flow through DAG nodes
- [ ] **Content Filtering**: Block harmful/sensitive content before LLM calls
- [ ] **Policy DSL**: Declarative security rules in `.aether` files

#### Caching
- [ ] **Level 2: Semantic Cache**: Vector similarity for near-match prompts (requires vector DB)
- [ ] **Level 3: Provider Prefix Cache**: Stable prompt prefix hints for provider optimization

### Medium Priority (Before v1.0 - December 2026)

#### IDE Tooling
- [ ] **Language Server Protocol (LSP)**: Real-time diagnostics and completion
- [ ] **VS Code Extension**: Syntax highlighting, error reporting, DAG preview
- [ ] **DAG Visualizer Improvements**: Real-time execution tracing

#### Testing
- [ ] **Property-Based Testing**: QuickCheck/proptest for parser edge cases
- [ ] **Fuzz Testing**: AFL-style input fuzzing for security
- [ ] **Coverage Reporting**: Line and branch coverage in CI

#### Context Management
- [ ] **RedisContextStore**: Persistent cross-request context
- [ ] **FileContextStore**: JSONL-backed context for debugging
- [ ] **Context TTL**: Automatic expiration for stale context

### Low Priority (Post v1.0)

- [ ] **Multi-language Bindings**: Python, JavaScript, Go SDKs
- [ ] **Container Images**: Docker/Podman distribution
- [ ] **Helm Charts**: Kubernetes deployment
- [ ] **Performance Profiling UI**: Web dashboard for execution analysis

---

## Documentation TODOs

### Whitepapers

#### WHITEPAPER.md (Technical)
- [x] Update Section 1 (Executive Summary) with v2.7 status ✅
- [x] Add Section 8.5 for criterion benchmarks ✅
- [ ] Update performance projections with actual benchmark data (when available)

#### WHITEPAPER_ACADEMIC.md
- [x] Update Abstract with latest empirical results ✅
- [x] Expand Appendix B with v2.7 feature matrix ✅
- [x] Add OTLP tracing to Section 7 (Runtime Architecture) ✅
- [ ] Update H1-H3 hypotheses with validation status (requires empirical results)

### Supporting Documents
- [x] CHANGELOG.md: Add v2.7 entry (DONE)
- [x] benchmark_metrics.md: Add criterion section (DONE)
- [x] audit_mapping.md: Update status (DONE)
- [x] aether-whitepaper-academic-changes.md: Mark all P0-P4 items as complete ✅

### External Documentation
- [ ] **API Reference**: OpenAPI/Swagger for runtime endpoints
- [ ] **User Guide**: Getting started, examples, troubleshooting
- [ ] **Architecture Guide**: Deep dive into compiler and runtime internals
- [ ] **Security Guide**: Best practices, threat model, mitigations

---

## Roadmap to Milestones

### Beta Release (July 2026)
1. Complete retry policy execution
2. Implement semantic caching (Level 2)
3. VS Code extension with basic LSP
4. Comprehensive user documentation
5. Docker image distribution

### Release Candidate (October 2026)
1. Complete taint tracking
2. Full LSP implementation
3. Performance optimization pass
4. Security audit and hardening

### Version 1.0 (December 2026)
1. Production-ready stability
2. Multi-cloud deployment guides
3. Community contribution guidelines
4. Long-term support policy

---

## Testing Matrix

| Test Type | Status | Coverage |
|-----------|--------|----------|
| Unit Tests (Rust) | ✅ | ~80% |
| Integration Tests | ✅ | Runtime + Compiler |
| Criterion Benchmarks | ✅ | DAG execution |
| Python Benchmarks | ✅ | Baseline comparisons |
| CI Pipeline | ✅ | GitHub Actions |
| Property Tests | 📋 | Planned |
| Fuzz Tests | 📋 | Planned |

---

## Dependencies to Watch

| Package | Current | Notes |
|---------|---------|-------|
| opentelemetry | 0.21.0 | Stable, OTLP support |
| tracing-opentelemetry | 0.22.0 | Compatible with opentelemetry 0.21 |
| tokio | 1.0 | Stable, full features |
| axum | 0.7 | Web framework |
| criterion | 0.5 | Benchmarking |

---

## Metrics to Track

### Success Criteria (from Whitepaper)

| ID | Target | Current | Status |
|----|--------|---------|--------|
| SC-4 | >30% latency reduction | Projected 2.7x (~63%) | ⏳ Validation needed |
| SC-5 | >40% cache hit rate | Projected 60% | ⏳ Validation needed |
| SC-6 | >25% cost reduction | Projected via cache | ⏳ Validation needed |

### Next Empirical Validation Steps
1. Run criterion benchmarks with MSVC toolchain
2. Execute Python benchmarks against real LLM APIs
3. Collect production-like workload traces
4. Update whitepapers with actual metrics
