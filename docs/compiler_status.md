# Aether Compiler Status

This document tracks the implementation status of the Aether language compiler.

## Overview

The Aether compiler transforms `.aether` source files into DAG (Directed Acyclic Graph) JSON that the runtime can execute. The compiler follows a traditional pipeline:

1. **Lexer** - Tokenizes source code
2. **Parser** - Builds Abstract Syntax Tree (AST)
3. **Semantic Analysis** - Symbol resolution and type checking
4. **Code Generation** - Emits DAG JSON

## CLI Commands

The `aetherc` CLI provides three commands:

```bash
# Compile to DAG JSON
aetherc compile path/to/file.aether -o output.json

# Parse and type-check only
aetherc check path/to/file.aether

# Parse and output AST as JSON
aetherc parse path/to/file.aether
```

---

## Implemented Features

### Lexer (Complete)

| Feature | Status | Notes |
|---------|--------|-------|
| Keywords | Done | `llm`, `fn`, `flow`, `struct`, `enum`, `context`, `type`, `let`, `if`, `else`, `for`, `while`, `match`, `return`, `test`, `import`, etc. |
| Operators | Done | Arithmetic, comparison, logical, assignment |
| Literals | Done | Strings (with escapes), integers, floats, booleans |
| Template markers | Done | `{{` and `}}` for interpolation |
| Comments | Done | Line (`//`) and block (`/* */`) |
| Built-in types | Done | `string`, `int`, `float`, `bool`, `list`, `map`, `optional` |

### Parser (Complete)

| Construct | Status | Notes |
|-----------|--------|-------|
| `llm fn` declarations | Done | Full body: model, temperature, max_tokens, system/user/prompt |
| `fn` declarations | Done | Parameters, optional return type, block body |
| `flow` definitions | Done | Parameters, return type, block body with statements |
| `struct` definitions | Done | Named fields with types |
| `enum` definitions | Done | Simple variants and variants with associated data `Variant(Type)` |
| `context` definitions | Done | Same structure as struct |
| `type` aliases | Done | With optional `where` constraint (parsing only) |
| `test` blocks | Done | Test name and body |
| `import` statements | Done | Braced list with aliases |
| Decorators | Done | `@name(args)` syntax |
| Expressions | Done | Literals, identifiers, binary/unary ops, calls, field access, indexing, match, struct literals, list/map literals |
| Statements | Done | let, return, if/else, for, while, try/catch, assert |
| String templates | Done | `{{variable}}` interpolation parsing |

### Semantic Analysis (MVP Complete)

| Feature | Status | Notes |
|---------|--------|-------|
| Symbol table | Done | Hierarchical scopes with push/pop for types, functions, and variables |
| Type collection | Done | Structs, enums, type aliases registered with full type info |
| Function signatures | Done | LLM fn, fn, flow signatures collected with parameter types |
| LLM function validation | Done | Checks for required model and prompt fields |
| Flow call analysis | Done | Extracts function calls, validates arguments, tracks dependencies |
| Undefined reference detection | Done | Reports undefined functions with "Did you mean?" suggestions |
| Duplicate definition detection | Done | Reports duplicate type/function/parameter/field names |
| Type checking | Done | Forward-only type inference for expressions |
| Template validation | Done | Validates `{{variable}}`, `{{context.KEY}}`, `{{node.ID.output}}` references |
| Struct/Enum validation | Done | Duplicate field/variant detection, field access validation |
| Return type validation | Done | Verifies return expressions match declared types |
| Error accumulation | Done | Collects up to 10 errors before aborting |
| Source locations | Done | Line and column numbers in all error messages |

### Code Generation (MVP Complete)

| Feature | Status | Notes |
|---------|--------|-------|
| Flow to DAG conversion | Done | One DagNode per LLM fn call |
| Dependency computation | Done | Derived from data flow (variable bindings) |
| Cycle detection | Done | Topological sort using Kahn's algorithm |
| Template placeholder preservation | Done | `{{...}}` kept in `prompt_template` |
| Template reference metadata | Done | Structured `template_refs` array with kind, path, node_id |
| Model/temperature/max_tokens | Done | Preserved in DagNode |
| DAG metadata | Done | flow_name, inputs, output_type, compiler_version |
| Source location tracking | Done | Optional, disabled for tests |

### DAG JSON Schema

The compiler emits DAG JSON with the following structure:

```json
{
  "schema_version": "1.0",
  "metadata": {
    "flow_name": "analyze",
    "source_file": "example.aether",
    "compiler_version": "0.1.0",
    "compiled_at": "2026-02-04T...",
    "inputs": [{ "name": "doc", "type": "string", "required": true }],
    "output_type": "AnalysisResult"
  },
  "nodes": [
    {
      "id": "summary",
      "node_type": "llm_fn",
      "name": "summarize",
      "prompt_template": "Summarize: {{text}}",
      "template_refs": [
        {
          "raw": "{{text}}",
          "kind": "parameter",
          "path": ["text"],
          "required": true,
          "sensitivity": "low"
        }
      ],
      "model": "gpt-4o",
      "temperature": 0.3,
      "dependencies": [],
      "return_type": "string"
    }
  ]
}
```

---

## Planned Features (Phase 2+)

### Parser Enhancements

| Feature | Priority | Notes |
|---------|----------|-------|
| `async`/`await` validation | Medium | Currently parsed but not semantically validated |
| `retry with` syntax | Medium | Error handling with backoff strategies |
| `fallback` clauses | Medium | Fallback LLM providers |
| `golden_dataset` | Low | Test data integration |
| Pipeline operators | Low | `|>` for function composition |

### Semantic Analysis Enhancements

| Feature | Priority | Notes |
|---------|----------|-------|
| Full type inference | ✅ Done | Forward-only type inference for let bindings and expressions |
| Type compatibility checking | ✅ Done | Validates assignments, returns, and argument types |
| Constrained type validation | Medium | `type Rating = int where 1 <= value <= 5` |
| Context usage tracking | Low | Analyze context variable usage |
| Import resolution | Medium | Load and merge imported modules |

### Code Generation Enhancements

| Feature | Priority | Notes |
|---------|----------|-------|
| Compile-time constant folding | Medium | Fold `{{const.NAME}}` when safe |
| Render policy generation | Medium | Security hints per node |
| Execution hints | Low | parallel_group, max_concurrency |
| Multiple output formats | Low | Rust code, Python, etc. |

### Runtime Integration

| Feature | Priority | Notes |
|---------|----------|-------|
| Template rendering | High | Substitute `template_refs` at runtime |
| Security policy enforcement | High | Validate against `render_policy` |
| Cache key computation | High | Use `template_refs` for accurate caching |

---

## Architecture

### Crate Structure

```
aether-lang/
  aether-core/          # Shared types (Dag, DagNode, TemplateRef, etc.)
  aether-compiler/      # Compiler (lexer, parser, semantic, codegen)
    src/
      lexer.rs          # Tokenizer using logos
      parser.rs         # Recursive descent parser
      ast.rs            # AST node definitions
      semantic.rs       # Symbol table and validation
      codegen.rs        # AST to DAG transformation
      main.rs           # CLI (aetherc)
      lib.rs            # Library exports
    tests/
      snapshot_tests.rs # DAG JSON snapshot tests
      error_tests.rs    # Error message tests
  aether-runtime/       # Execution engine
```

### Key Design Decisions

1. **Template placeholders preserved**: The compiler does not substitute `{{...}}` at compile time. Instead, it emits metadata (`template_refs`) that the runtime uses for deterministic substitution.

2. **Dependency-based parallelism**: No explicit `parallel {}` block. The DAG structure implicitly enables parallel execution of nodes with no dependencies.

3. **Shared types in aether-core**: Both compiler and runtime use the same `Dag`, `DagNode`, `TemplateRef` types to ensure compatibility.

4. **Enum variants with data**: Support for `Variant(Type)` syntax enables rich error handling and result types.

---

## Testing

### Running Tests

```bash
# Run all tests
cargo test --workspace

# Run compiler tests only
cargo test -p aether-compiler

# Update snapshots (after intentional changes)
cargo insta test --accept
```

### Test Categories

1. **Unit tests**: In-module tests for lexer, parser, semantic, codegen
2. **Snapshot tests**: DAG JSON output verification using insta
3. **Error tests**: Verify helpful error messages for malformed input
4. **Example files**: Real `.aether` programs in `/examples`

---

## Example Programs

### 1. Simple Sentiment Classification

[examples/sentiment.aether](../examples/sentiment.aether)

Single LLM function wrapped in a flow. Demonstrates basic compilation.

### 2. Parallel Document Analysis

[examples/parallel_flow.aether](../examples/parallel_flow.aether)

Multiple independent LLM calls that can execute in parallel. The DAG shows nodes with no interdependencies.

### 3. Chained Processing Pipeline

[examples/chained_flow.aether](../examples/chained_flow.aether)

Sequential LLM calls where each step depends on the previous. The DAG shows explicit dependency edges.

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to the compiler.

Key areas where help is needed:
- Type inference implementation
- Import resolution
- Additional error recovery in parser
- More comprehensive test coverage
