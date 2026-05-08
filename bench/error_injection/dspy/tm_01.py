"""tm_01 (DSPy): flow returns string where struct-shaped Prediction is expected.

Aether catches as TypeMismatch. DSPy has no static checking on the return type
of a Module.forward(); the wrong-shaped value is forwarded silently.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class ClassifyText(dspy.Signature):
    """Classify input text into a result struct."""
    text: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: int = dspy.OutputField()


def analyze(input_text: str) -> dspy.Prediction:
    # Aether equivalent: flow analyze(input: string) -> Result
    # bug: returns a plain string instead of a Prediction with answer/confidence.
    return "wrong return type"  # type: ignore[return-value]


if __name__ == "__main__":
    out = analyze("hello world")
    print(f"output={out!r} type={type(out).__name__}")
