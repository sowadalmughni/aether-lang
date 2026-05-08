#!/usr/bin/env python3
"""Download the HotpotQA dev (distractor) split and write 500 examples
into bench/datasets/hotpotqa_dev_500.jsonl in the bench schema.

Schema written (one JSON object per line):
  {
    "id":                <str>   # HotpotQA's _id
    "query":             <str>   # the multi-hop question
    "expected_answer":   <str>   # gold short answer (yes/no/string)
    "context_paragraphs": [
        {"title": <str>, "sentences": [<str>, ...]},
        ...   # 10 paragraphs per item in the distractor split
    ],
    "supporting_facts":  [[<title>, <sent_idx>], ...],
    "level":             <str>,  # "easy" | "medium" | "hard"
    "type":              <str>   # "comparison" | "bridge"
  }

The HotpotQA dev distractor split has 7,405 questions; we write the first
500 in the order the `datasets` library yields them. Selection is
deterministic given a fixed dataset version.

Usage:
  py -3.13 scripts/download_hotpotqa.py
  py -3.13 scripts/download_hotpotqa.py --limit 500 --out bench/datasets/hotpotqa_dev_500.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "bench" / "datasets" / "hotpotqa_dev_500.jsonl"


def _normalize_context(raw_context) -> list[dict]:
    """HotpotQA on HuggingFace stores `context` as a dict-of-lists:
        {"title": [t1, t2, ...], "sentences": [[s1a, s1b, ...], [s2a, ...], ...]}
    Some older mirrors expose it as a list of (title, sentences) tuples.
    Normalise to: [{"title": str, "sentences": [str, ...]}, ...].
    """
    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentence_lists = raw_context.get("sentences", [])
        return [
            {"title": titles[i], "sentences": list(sentence_lists[i])}
            for i in range(min(len(titles), len(sentence_lists)))
        ]
    out = []
    for entry in raw_context:
        title, sentences = entry[0], entry[1]
        out.append({"title": str(title), "sentences": [str(s) for s in sentences]})
    return out


def _normalize_supporting_facts(raw_sf) -> list[list]:
    """Same layout drift as `context`. Return [[title, sent_idx], ...]."""
    if isinstance(raw_sf, dict):
        titles = raw_sf.get("title", [])
        sent_ids = raw_sf.get("sent_id", [])
        return [[str(titles[i]), int(sent_ids[i])] for i in range(min(len(titles), len(sent_ids)))]
    return [[str(t), int(i)] for (t, i) in raw_sf]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HotpotQA dev distractor split")
    parser.add_argument("--limit", type=int, default=500, help="Number of examples to write")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument(
        "--config",
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="HotpotQA config (distractor split is what reviewers expect)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(f"ERROR: `datasets` not installed: {exc}", file=sys.stderr)
        print("Install with: py -3.13 -m pip install 'datasets>=2.20,<4.0'", file=sys.stderr)
        return 2

    print(f"Loading hotpot_qa / {args.config} / validation ...", flush=True)
    try:
        ds = load_dataset("hotpot_qa", args.config, split="validation", trust_remote_code=True)
    except Exception as exc:
        print(f"ERROR: failed to load HotpotQA dataset: {exc}", file=sys.stderr)
        print(
            "If this is a network failure, retry with HF_HUB_OFFLINE=0 and a working "
            "internet connection. Do NOT generate fake HotpotQA examples.",
            file=sys.stderr,
        )
        return 3

    n_total = len(ds)
    print(f"Loaded {n_total} validation examples", flush=True)
    if n_total < args.limit:
        print(
            f"ERROR: dataset has only {n_total} examples but --limit is {args.limit}",
            file=sys.stderr,
        )
        return 4

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(args.limit):
            row = ds[i]
            entry = {
                "id": str(row.get("id") or row.get("_id")),
                "query": str(row["question"]),
                "expected_answer": str(row["answer"]),
                "context_paragraphs": _normalize_context(row["context"]),
                "supporting_facts": _normalize_supporting_facts(row["supporting_facts"]),
                "level": str(row.get("level", "")),
                "type": str(row.get("type", "")),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {out_path}", flush=True)
    if written != args.limit:
        print(f"ERROR: wrote {written} but expected {args.limit}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
