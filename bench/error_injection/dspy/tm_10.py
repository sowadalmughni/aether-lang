"""tm_10 (DSPy): passes UserMessage Prediction where SystemMessage is expected.

Both have the same fields; DSPy's Prediction is duck-typed, so no nominal check.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class UserSig(dspy.Signature):
    text: str = dspy.InputField()
    body: str = dspy.OutputField()


class SystemSig(dspy.Signature):
    msg_body: str = dspy.InputField()
    out: str = dspy.OutputField()


def shape_user(text: str) -> dspy.Prediction:
    return dspy.Prediction(body=text)  # "UserMessage"


def deliver_system(msg: dspy.Prediction) -> str:
    return msg.body


def run(input_text: str) -> str:
    u = shape_user(input_text)
    return deliver_system(u)  # bug: structurally identical to UserMessage


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r}")
