"""ur_09 (DSPy): second flow calls an absent helper function."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class FirstSig(dspy.Signature):
    text: str = dspy.InputField()
    out: str = dspy.OutputField()


def first(text: str) -> str:
    return f"first:{text}"


def primary(input_text: str):
    return first(input_text)


def secondary(input_text: str):
    return absent_helper(input_text)  # noqa: F821  bug


if __name__ == "__main__":
    primary("ok")
    print(secondary("hello"))
