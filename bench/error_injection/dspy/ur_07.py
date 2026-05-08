"""ur_07 (DSPy): second flow's first statement references an undefined name."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class ShapeSig(dspy.Signature):
    text: str = dspy.InputField()
    out: str = dspy.OutputField()


def shape(text: str) -> str:
    return f"shape:{text}"


def primary(input_text: str):
    return shape(input_text)


def secondary(input_text: str):
    return shape(missing_input)  # noqa: F821  bug


if __name__ == "__main__":
    primary("ok")
    print(secondary("hello"))
