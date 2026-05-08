"""ur_01 (DSPy): pipeline calls a function that does not exist."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class SummarizeSig(dspy.Signature):
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()


def summarize(text: str) -> str:
    return f"summary-of-{text}"


def analyze(input_text: str):
    s = summarize(input_text)
    return nonexistent_function(s)  # noqa: F821  bug


if __name__ == "__main__":
    print(analyze("hello"))
