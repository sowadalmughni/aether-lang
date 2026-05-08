"""ur_10 (DSPy): return statement references a typoed variable name."""
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


def run(input_text: str):
    outcome = shape(input_text)
    return outcomes  # noqa: F821  bug


if __name__ == "__main__":
    print(run("hello"))
