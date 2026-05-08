"""ur_03 (DSPy): let-binding rhs references a symbol that does not exist."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class EchoSig(dspy.Signature):
    text: str = dspy.InputField()
    out: str = dspy.OutputField()


def echo(text: str) -> str:
    return text


def run(input_text: str):
    return echo(ghost_value)  # noqa: F821  bug


if __name__ == "__main__":
    print(run("hello"))
