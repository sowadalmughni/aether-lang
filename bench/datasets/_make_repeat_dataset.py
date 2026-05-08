"""Generate customer_support_repeat_100.jsonl from customer_support_100.jsonl.

Picks 30 unique items deterministically (random.Random(42).sample) and
replicates them round-robin to fill 100 entries. The query/expected fields are
preserved verbatim so the runtime sees identical prompts (same cache key);
each output row gets a fresh `id` and a `source_id` pointer.

Run from repo root:
    python3 bench/datasets/_make_repeat_dataset.py

Idempotent: deterministic output for fixed seed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "customer_support_100.jsonl"
DST = ROOT / "customer_support_repeat_100.jsonl"

SEED = 42
N_UNIQUE = 30
N_TOTAL = 100


def main() -> int:
    items = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(items) != 100:
        print(f"expected 100 items in {SRC}, got {len(items)}")
        return 1
    rng = random.Random(SEED)
    chosen = rng.sample(items, N_UNIQUE)
    out = []
    for i in range(N_TOTAL):
        src = chosen[i % N_UNIQUE]
        new = dict(src)
        new["id"] = f"csr_{i + 1:03d}"
        new["source_id"] = src["id"]
        out.append(new)
    with DST.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    distinct_queries = len({r["query"] for r in out})
    print(f"Wrote {DST}: {len(out)} rows, {distinct_queries} distinct queries (expected {N_UNIQUE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
