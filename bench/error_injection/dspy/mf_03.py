"""mf_03 (DSPy): accessing a nested Prediction field that does not exist."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class AddressSig(dspy.Signature):
    text: str = dspy.InputField()
    city: str = dspy.OutputField()
    zip: str = dspy.OutputField()


class PersonSig(dspy.Signature):
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    address: dspy.Prediction = dspy.OutputField()


def build(text: str) -> dspy.Prediction:
    addr = dspy.Prediction(city="seattle", zip="98101")
    return dspy.Prediction(name="alice", address=addr)


def run(input_text: str) -> str:
    p = build(input_text)
    return p.address.country  # bug: address has no `country`


if __name__ == "__main__":
    print(run("hello"))
