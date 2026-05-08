"""dd_01 (DSPy): two functions named `classify`. Python silently overrides."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class ClassifySig(dspy.Signature):
    text: str = dspy.InputField()
    answer: str = dspy.OutputField()


def classify(text: str) -> str:
    return "A"


def classify(text: str) -> str:  # noqa: F811  bug: silent override
    return "B"


def analyze(input_text: str):
    return classify(input_text)


if __name__ == "__main__":
    out = analyze("hello")
    print(f"output={out!r}")
