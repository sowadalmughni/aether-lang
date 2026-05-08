"""ur_05 (DSPy): typoed function name (typo of `polish`)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class SumSig(dspy.Signature):
    text: str = dspy.InputField()
    out: str = dspy.OutputField()


def summarize(text: str) -> str:
    return f"sum:{text}"


def polish(text: str) -> str:
    return text.upper()


def run(input_text: str):
    s = summarize(input_text)
    return polishe(s)  # noqa: F821  bug


if __name__ == "__main__":
    print(run("hello"))
