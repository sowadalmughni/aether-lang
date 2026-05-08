"""mf_05 (DSPy): caller passes too many args to a 2-arg function."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class JoinSig(dspy.Signature):
    a: str = dspy.InputField()
    b: str = dspy.InputField()
    out: str = dspy.OutputField()


def join(a: str, b: str) -> str:
    return a + b


def step(text: str) -> str:
    return text


def run(input_text: str) -> str:
    s = step(input_text)
    return join(s, input_text, input_text)  # type: ignore[call-arg]  bug


if __name__ == "__main__":
    print(run("hello"))
