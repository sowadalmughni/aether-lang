"""ur_02 (DSPy): returns a variable that was never bound."""
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


def run(input_text: str):
    s = summarize(input_text)
    return result  # noqa: F821  bug


if __name__ == "__main__":
    print(run("hello"))
