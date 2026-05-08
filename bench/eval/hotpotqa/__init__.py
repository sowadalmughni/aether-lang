"""HotpotQA answer-evaluation primitives. See NOTICE for attribution."""

from .hotpot_evaluate_v1 import (
    evaluate_predictions,
    exact_match_score,
    f1_score,
    normalize_answer,
    update_answer,
)

__all__ = [
    "evaluate_predictions",
    "exact_match_score",
    "f1_score",
    "normalize_answer",
    "update_answer",
]
