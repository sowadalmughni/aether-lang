"""dd_03 (DSPy): a Signature with two same-named field annotations."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class Account(dspy.Signature):
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    name: str = dspy.OutputField()  # noqa: F811  bug: dup field, second wins


def build(text: str) -> dspy.Prediction:
    return dspy.Prediction(name="alice")


if __name__ == "__main__":
    a = build("hello")
    fields = list(Account.fields.keys())
    print(f"output={a!r} fields={fields}")
