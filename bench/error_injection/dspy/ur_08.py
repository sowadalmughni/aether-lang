"""ur_08 (DSPy): function arg references an undefined identifier."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class StepSig(dspy.Signature):
    text: str = dspy.InputField()
    out: str = dspy.OutputField()


def step(text: str) -> str:
    return f"step:{text}"


def run(input_text: str):
    return step(unseen)  # noqa: F821  bug


if __name__ == "__main__":
    print(run("hello"))
