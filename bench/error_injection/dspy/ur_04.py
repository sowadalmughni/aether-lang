"""ur_04 (DSPy): middle of a 3-step pipeline calls a missing function."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class FirstSig(dspy.Signature):
    text: str = dspy.InputField()
    out: str = dspy.OutputField()


class ThirdSig(dspy.Signature):
    text: str = dspy.InputField()
    out: str = dspy.OutputField()


def first(text: str) -> str:
    return f"first:{text}"


def third(text: str) -> str:
    return f"third:{text}"


def run(input_text: str):
    a = first(input_text)
    b = missing_middle(a)  # noqa: F821  bug
    return third(b)


if __name__ == "__main__":
    print(run("hello"))
