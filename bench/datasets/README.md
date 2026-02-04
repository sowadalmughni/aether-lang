# Aether Benchmark Datasets

Synthetic datasets for reproducible performance benchmarking of the Aether runtime.

## Datasets

### customer_support_100.jsonl

100 customer support queries for multi-agent routing benchmarks.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (cs_001 - cs_100) |
| `query` | string | Customer support query text |
| `expected_urgency` | string | Low, Medium, High, or Critical |
| `expected_category` | string | Category label (authentication, billing, etc.) |
| `context.customer_tier` | string | free, pro, or enterprise |
| `context.previous_tickets` | number | Number of prior support tickets |

**Distribution:**
- Urgency: ~30% Low, ~35% Medium, ~25% High, ~10% Critical
- Categories: 18 unique categories including authentication, billing, bug, outage, security, feature-request

**Usage:**
```bash
# Load in Python
import json
with open('customer_support_100.jsonl') as f:
    queries = [json.loads(line) for line in f]

# Use with Aether benchmark
python scripts/run_benchmark.py --dataset bench/datasets/customer_support_100.jsonl
```

---

### document_analysis_50.jsonl

50 documents across various domains for parallel extraction benchmarks.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (doc_001 - doc_050) |
| `document` | string | Document text for analysis |
| `expected_entities` | array | List of entities that should be extracted |
| `expected_summary_length` | number | Target word count for summaries |
| `domain` | string | Document domain classification |

**Domains (50 documents):**
- Technology: machine-learning, technology, data-science, networking, security
- Business: finance, business, marketing, operations, venture-capital, corporate
- Science: medical, biology, chemistry, astronomy, geology, astrophysics
- Legal: legal, intellectual-property, insurance, compliance
- Other: culinary, real-estate, education, entertainment, sports, etc.

**Usage:**
```bash
# Load in Python
import json
with open('document_analysis_50.jsonl') as f:
    docs = [json.loads(line) for line in f]

# Parallel extraction benchmark
for doc in docs:
    result = aether_runtime.execute({
        "dag": parallel_extraction_dag,
        "context": {"document": doc["document"]}
    })
```

---

## Benchmark Scenarios

### 1. Triage Agent (customer_support_100)

Tests multi-step LLM workflow:
1. Urgency classification
2. Category detection
3. Response generation or escalation

Measures: latency, cache effectiveness, parallel speedup

### 2. Parallel Extraction (document_analysis_50)

Tests parallel LLM execution:
1. Entity extraction
2. Summary generation
3. Domain classification

(All three run in parallel, then combined)

Measures: parallelization factor, level-based execution times

### 3. Sequential vs Parallel Ablation

Run same workflow with `?sequential=true` to measure parallel speedup:

```bash
# Parallel (default)
curl -X POST http://localhost:3000/execute -d @dag.json

# Sequential (ablation)
curl -X POST "http://localhost:3000/execute?sequential=true" -d @dag.json
```

---

## Creating New Datasets

Use JSONL format (one JSON object per line):

```jsonl
{"id": "item_001", "input": "...", "expected_output": "...", "metadata": {...}}
{"id": "item_002", "input": "...", "expected_output": "...", "metadata": {...}}
```

Requirements:
- Each line must be valid JSON
- Include unique `id` field for result tracking
- Include `expected_*` fields for accuracy validation
- Keep individual documents under 4KB for consistent LLM context usage
