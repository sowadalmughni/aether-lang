# arXiv submission metadata — Aether v3.2-academic

This file documents the metadata required to submit `aether-arxiv-v3.2.tar.gz`
to arXiv. Submission itself is a manual step the author performs after
producing this package; nothing in this repository auto-uploads.

## Title

Aether: A Domain-Specific Language for Type-Safe LLM Orchestration

## Authors

Md. Sowad Al-Mughni

## arXiv categories

| Role | Category |
|------|----------|
| Primary | `cs.PL` (Programming Languages) |
| Cross-list | `cs.SE` (Software Engineering) |
| Cross-list | `cs.LG` (Machine Learning) |

Rationale: the central contribution is a domain-specific language with a
static type system and DAG IR (PL); the work targets LLM-orchestration
engineering practice (SE); and the application domain is large language
model integration (LG).

## Comment field (arXiv "Comments" line)

```
36 pages, 10 figures, 23 references. Source, benchmark JSONs, and
reproducibility scripts at the repository linked under "Code"; see
REPRODUCIBILITY.md and reproduce.sh. Tag: paper-v3.2-academic.
```

## Abstract (form field, <= 1920 characters)

The PDF's full abstract (2369 chars) carries the complete numeric trace
required by the paper. arXiv's submission form caps the metadata abstract
at 1920 characters, so the version below is condensed for that field
while preserving every measured number that appears in the PDF abstract.

```
Large language model (LLM) integration in production suffers from systematic engineering failures: runtime-only type checking, complex orchestration without static validation, suboptimal caching, and inadequate security guarantees. Existing tools address these in isolation: orchestration frameworks lack compile-time safety, typed-output libraries focus narrowly on schemas, and security tools operate only at runtime.

This paper presents Aether, a domain-specific language that treats LLM orchestration as a first-class engineering discipline. Aether introduces three core abstractions -- llm fn for typed LLM interactions, flow for DAG-based composition, and context for state -- and a compiler that performs static type, parallelism, and security analysis.

We contribute (1) a type system spanning LLM inputs, outputs, and workflow compositions with compile-time verification; (2) a DAG-based IR enabling static optimization; (3) a reproducible benchmark methodology with full JSON artifacts under bench/results/; (4) an open-source prototype with OTLP tracing (OpenTelemetry 0.21.0) and Criterion benchmarks; and (5) compile-time taint tracking with measured 100% catch rate on a 60-case adapted InjecAgent corpus. Measured outcomes: parallel execution yields paired BCa speedups of 1.4778x on customer_support_100 and 2.5841x on document_analysis_50; the L1 exact-match cache reaches a hit rate of 0.7000 on a 70%-repeat workload and 1.0000 warm-cache; the compiler catches 30/30 intentionally malformed programs at compile time vs 17/30 caught at runtime by LangChain and DSPy (13 missed silently by each baseline). End-to-end real-OpenAI runs (gpt-4o-mini, 3 trials, both datasets, all three systems) cost $0.128887 for Aether of $0.478349 total. All numbers trace to JSON files in bench/results/.
```

## Tarball

| Field | Value |
|-------|-------|
| File | `aether-arxiv-v3.2.tar.gz` |
| Size | 232,569 bytes (227 KiB) |
| SHA-256 | `1e4621e0b81f4cb5824ecf48317bd097632d07577f8c7e4c4f479d078f908729` |

Root layout (18 files):

```
./aether.tex          % main document; bibliography rendered inline by pandoc citeproc
./header.tex          % preamble fragment
./arxiv.sty           % vendored arXiv preprint style (kourgeorge/arxiv-style 92051469, MIT)
./figures/architecture.pdf
./figures/caching_cascade.pdf
./figures/challenges.pdf
./figures/compiler_pipeline.pdf
./figures/roadmap.pdf
./figures/cache_hit_rate.pdf
./figures/cross_system_latency.pdf
./figures/parallel_speedup.pdf
./figures/security_outcome.pdf
./figures/type_safety_corpus.pdf
./figures/cache_hit_rate.json        % audit sidecar (figure -> JSON field-path trace)
./figures/cross_system_latency.json  % audit sidecar
./figures/parallel_speedup.json      % audit sidecar
./figures/security_outcome.json      % audit sidecar
./figures/type_safety_corpus.json    % audit sidecar
```

No `.bib` or `.bbl` is included: the Pandoc + citeproc pipeline used to
generate `aether.tex` resolves and typesets the bibliography directly into
the LaTeX source, so the document is self-contained.

## Build instructions (arXiv reviewer / reproducer)

The package builds with Tectonic (XeTeX backend) without auxiliary tools:

```sh
tar -xzf aether-arxiv-v3.2.tar.gz -C build/
cd build
tectonic aether.tex
```

Verified locally with Tectonic 0.16.9 from a freshly extracted tarball;
output is `aether.pdf` (36 pages, 368,535 bytes). The PDF byte-content
varies across builds because Tectonic embeds a build timestamp; the xdv
intermediate is byte-identical (1,103,612 bytes) and the rendered pages
match the in-repo `whitepaper/latex/aether.pdf`.

## Reproducibility

Every numeric claim in the paper is sourced from a JSON file under
`bench/results/`. End-to-end reproduction (mock-mode, ~110-125 minutes on
the reference Intel i5-8250U / 8 GB / Ubuntu 24.04 baseline) is via the
top-level driver script:

```sh
bash reproduce.sh
```

This builds the pinned Docker image (`Dockerfile`), brings up the runtime
service in mock mode, re-executes the full mock benchmark suite, and
diffs the produced JSONs against the committed reference set. The verdict
lands in `bench/results/repro/DIFF_REPORT.md`. Tolerance thresholds and
expected drift are documented in `bench/verify_tolerance.toml` and
`REPRODUCIBILITY.md`.

Real-API runs (`*_real_api_v1.json`, `security_v1.json`, `real_api_v1.json`)
are out of scope for `reproduce.sh` because they require an `OPENAI_API_KEY`
and incur ~$0.50 of OpenAI cost. They are independently reproducible per
the instructions in `bench/results/REAL_API.md`.

## Manual submission note

arXiv submission is performed manually by the author through the
arXiv.org web interface after this package is produced. No agent-driven
or scripted submission to arXiv is performed from this repository.
