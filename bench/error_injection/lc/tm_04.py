"""tm_04 (LC): flow returns enum-like value where struct is declared.

Aether catches as TypeMismatch. Python silently returns the enum.
"""
from __future__ import annotations
import sys
from enum import Enum
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Report(BaseModel):
    text: str
    score: int


def classify(text: str) -> Sentiment:
    _ = MockChatModel().invoke(text)
    return Sentiment.POSITIVE


def run(input_text: str) -> Report:
    return classify(input_text)  # bug: Sentiment, declared Report


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
