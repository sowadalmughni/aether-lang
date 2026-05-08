"""mf_04 (DSPy): accessing a misspelled field name on an existing Prediction."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class OrderSig(dspy.Signature):
    text: str = dspy.InputField()
    id: str = dspy.OutputField()
    total: int = dspy.OutputField()


def build_order(text: str) -> dspy.Prediction:
    return dspy.Prediction(id="ord-1", total=100)


def run(input_text: str) -> int:
    o = build_order(input_text)
    return o.totl  # bug: typo of `total`


if __name__ == "__main__":
    print(run("hello"))
