"""tm_04 (DSPy): returns enum value where struct is declared."""
from __future__ import annotations
import sys
from enum import Enum
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ReportSig(dspy.Signature):
    text: str = dspy.InputField()
    body: str = dspy.OutputField()
    score: int = dspy.OutputField()


def classify(text: str) -> Sentiment:
    return Sentiment.POSITIVE


def run(input_text: str) -> dspy.Prediction:
    return classify(input_text)  # type: ignore[return-value]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
