"""mf_01 (DSPy): accessing a field that does not exist on a Prediction."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class ProfileSig(dspy.Signature):
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    age: int = dspy.OutputField()


def make_profile(text: str) -> dspy.Prediction:
    return dspy.Prediction(name="alice", age=30)


def show(input_text: str) -> str:
    p = make_profile(input_text)
    return p.email  # bug: Prediction has no `email` field


if __name__ == "__main__":
    print(show("hello"))
