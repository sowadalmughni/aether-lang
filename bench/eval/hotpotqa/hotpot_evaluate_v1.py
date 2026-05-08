"""HotpotQA answer evaluation -- exact-match and F1 over normalised tokens.

Vendored / faithfully reimplemented from the official HotpotQA reference
evaluator at:

    https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

The HotpotQA upstream repository (https://github.com/hotpotqa/hotpot) does
NOT carry a top-level LICENSE file at the time of vendoring (2026-05-08).
The dataset itself is distributed under CC BY-SA 4.0 per
https://hotpotqa.github.io/. The evaluation logic implemented here follows
the standard SQuAD-style normalisation pipeline (lower / strip punctuation
/ remove articles / collapse whitespace) and is well-documented as the
reference HotpotQA metric in the published paper:

    Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop
    Question Answering", EMNLP 2018. https://arxiv.org/abs/1809.09600

This file deliberately exposes ONLY the answer-evaluation entry points
(normalize_answer, f1_score, exact_match_score, update_answer) that the
benchmark suite needs; the supporting-facts and joint metrics from the
upstream script are intentionally omitted because the bench harness does
not yet evaluate sentence-level supporting facts.

See bench/eval/hotpotqa/NOTICE for full attribution and licensing notes.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation, drop articles, collapse whitespace.

    This is the SQuAD/HotpotQA standard normalisation. The order matters:
    lower -> strip-punct -> remove-articles -> collapse-whitespace.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Token-level F1, with the HotpotQA yes/no/noanswer special case.

    Returns (f1, precision, recall). If either side is one of the special
    yes/no/noanswer answers and they don't match exactly, returns zeros --
    matching the reference behaviour where partial overlap on those
    short answers shouldn't earn credit.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    SPECIAL = {"yes", "no", "noanswer"}
    if normalized_prediction in SPECIAL and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in SPECIAL and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """1 iff normalised strings match exactly."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_answer(
    metrics: dict, prediction: str, gold: str
) -> tuple[float, float, float]:
    """Update an EM/F1 running-totals dict in place; return (em, prec, recall).

    Mirrors the upstream `update_answer` shape so callers can accumulate
    over a list of (prediction, gold) pairs and divide by the count at
    the end.
    """
    em = float(exact_match_score(prediction, gold))
    f1, prec, recall = f1_score(prediction, gold)
    metrics["em"] = metrics.get("em", 0.0) + em
    metrics["f1"] = metrics.get("f1", 0.0) + f1
    metrics["prec"] = metrics.get("prec", 0.0) + prec
    metrics["recall"] = metrics.get("recall", 0.0) + recall
    return em, prec, recall


def evaluate_predictions(predictions: dict[str, str], golds: dict[str, str]) -> dict:
    """Aggregate EM/F1 over a {qid: pred} dict against a {qid: gold} dict.

    Returns {em, f1, prec, recall, n} where each metric is the mean over
    all gold questions. Missing predictions are scored against the empty
    string (i.e., they contribute 0 to EM and F1 unless gold is empty).
    """
    metrics: dict = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    n = 0
    for qid, gold in golds.items():
        pred = predictions.get(qid, "")
        update_answer(metrics, pred, gold)
        n += 1
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0, "n": 0}
    return {
        "em": metrics["em"] / n,
        "f1": metrics["f1"] / n,
        "prec": metrics["prec"] / n,
        "recall": metrics["recall"] / n,
        "n": n,
    }
