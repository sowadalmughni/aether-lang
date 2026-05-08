"""ur_06 (DSPy): second let-binding references a never-declared symbol."""
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
    a = step(input_text)
    b = step(unbound_intermediate)  # noqa: F821  bug
    return b


if __name__ == "__main__":
    print(run("hello"))
