"""tm_06 (DSPy): returns Wrapped Prediction where its inner string is expected."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class WrappedSig(dspy.Signature):
    text: str = dspy.InputField()
    value: str = dspy.OutputField()


def wrap(text: str) -> dspy.Prediction:
    return dspy.Prediction(value=text)


def run(input_text: str) -> str:
    return wrap(input_text)  # type: ignore[return-value]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
