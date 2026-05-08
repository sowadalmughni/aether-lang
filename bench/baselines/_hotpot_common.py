"""Shared utilities for the HotpotQA benchmark across Aether / LangChain / DSPy.

What lives here:
  * load_hotpot_dataset(path)   -- read bench/datasets/hotpotqa_dev_500.jsonl
  * serialize_paragraphs(...)   -- list-of-{title,sentences} -> single string
  * RAG_PROMPTS                 -- the two LLM prompt templates (decompose, answer)
  * extract_short_answer(s)     -- post-process raw model output to a short answer
  * evaluate_qa(predictions, golds) -- thin wrapper around the vendored evaluator

Each baseline (langchain_rag.py, dspy_rag.py, aether_hotpot.py) imports from
this module so the prompt strings and the answer-cleaning logic are
identical across systems -- otherwise EM/F1 differences would partly
reflect prompt-wording differences instead of system behaviour.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `from bench.eval...` works

from bench.eval.hotpotqa import evaluate_predictions, exact_match_score, f1_score  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "bench" / "datasets" / "hotpotqa_dev_500.jsonl"


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def load_hotpot_dataset(path: Path | str = DEFAULT_DATASET) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def serialize_paragraphs(paragraphs: list[dict]) -> str:
    """Turn the dataset's [{title, sentences[]}, ...] into a single string
    suitable for inlining into a prompt. Format:

        [TITLE: <title>]
        <sent1> <sent2> ...

        [TITLE: ...]
        ...

    Sentences are joined with single spaces; the original sentence-level
    granularity isn't needed for answer EM/F1 scoring.
    """
    out: list[str] = []
    for p in paragraphs:
        title = p.get("title", "")
        sents = p.get("sentences", [])
        body = " ".join(s.strip() for s in sents).strip()
        out.append(f"[TITLE: {title}]\n{body}")
    return "\n\n".join(out)


# -----------------------------------------------------------------------------
# Prompts -- byte-identical across the three baselines so EM/F1 differences
# reflect the systems, not prompt drift.
# -----------------------------------------------------------------------------

RAG_PROMPTS = {
    "decompose": (
        "You will be asked a multi-hop question. Decompose it into 2-3 single-hop "
        "subqueries that, answered together, would answer the original question. "
        "Return as a JSON array of strings.\n\n"
        "Question: {question}\n\n"
        "Subqueries:"
    ),
    "answer": (
        "Answer the multi-hop question using only the provided context. Give the "
        "shortest possible answer (yes/no or a short noun phrase). Do not explain "
        "your reasoning.\n\n"
        "Context:\n{context}\n\n"
        "Subqueries you should resolve:\n{subqueries}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
}


# -----------------------------------------------------------------------------
# Answer post-processing
# -----------------------------------------------------------------------------

# Strip leading "Answer:" / surrounding quotes / common boilerplate. The mock
# providers used in `--mode mock` produce structured non-answers (e.g. the
# Aether runtime mock returns "[Mock Response]\nModel: gpt-4o-mini\n...") which
# means EM/F1 in mock mode will be ~0 by construction. That's expected: the
# acceptance criterion is "EM and F1 fields populated", not "non-zero".

_ANSWER_PREFIX_RE = re.compile(r"^\s*answer\s*[:\-]\s*", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[\s\.\!\?,;:]+$")


def extract_short_answer(raw: str) -> str:
    """Trim a model response to a single short answer phrase.

    Rules (kept identical across baselines):
      1. Strip leading 'Answer:' / 'Answer -' if present.
      2. Take only the first non-empty line.
      3. Strip surrounding quotes / brackets.
      4. Strip trailing punctuation.

    The HotpotQA evaluator normalises further (lowercase, drop articles,
    strip all punctuation), so this only needs to keep the answer span on
    the first line free of explanatory boilerplate.
    """
    if raw is None:
        return ""
    s = str(raw)
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _ANSWER_PREFIX_RE.sub("", line)
        # Iterate stripping passes: trailing punctuation can hide an outer
        # quote, and after that quote is stripped a second pass of trailing
        # punctuation may apply. Two rounds is enough for the patterns we
        # actually see in HotpotQA-style answers.
        for _ in range(2):
            line = line.strip()
            line = _TRAILING_PUNCT_RE.sub("", line)
            line = line.strip('"').strip("'").strip("[]")
        if line:
            return line
    return ""


# -----------------------------------------------------------------------------
# Evaluation wrapper -- thin layer over the vendored HotpotQA evaluator so
# the per-trial dict has consistent EM/F1 fields across all three baselines.
# -----------------------------------------------------------------------------

def evaluate_qa(predictions: dict[str, str], golds: dict[str, str]) -> dict:
    """Returns {em, f1, prec, recall, n} as means in [0, 1]."""
    return evaluate_predictions(predictions, golds)


def per_item_em_f1(prediction: str, gold: str) -> tuple[float, float]:
    """Convenience: per-item EM (0/1) and F1 in [0, 1]."""
    em = 1.0 if exact_match_score(prediction, gold) else 0.0
    f1, _, _ = f1_score(prediction, gold)
    return em, f1
