"""tm_02 (DSPy): returns list[str] where string is declared.

Aether catches as TypeMismatch. DSPy is silent.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class ExtractSig(dspy.Signature):
    text: str = dspy.InputField()
    items: list[str] = dspy.OutputField()


def extract(text: str) -> list[str]:
    return ["a", "b", "c"]


def run(input_text: str) -> str:
    return extract(input_text)  # type: ignore[return-value]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
