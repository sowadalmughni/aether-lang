"""tm_05 (DSPy): returns plain string where Item-shaped Prediction is expected."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class ItemSig(dspy.Signature):
    text: str = dspy.InputField()
    name: str = dspy.OutputField()


def make_item(text: str) -> str:
    return "alpha"


def run(input_text: str) -> dspy.Prediction:
    return make_item(input_text)  # type: ignore[return-value]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
