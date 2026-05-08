"""dd_02 (DSPy): two Signatures share the same name."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class Profile(dspy.Signature):
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    age: int = dspy.OutputField()


class Profile(dspy.Signature):  # noqa: F811  bug: silent override
    text: str = dspy.InputField()
    handle: str = dspy.OutputField()
    score: int = dspy.OutputField()


def build(text: str) -> dspy.Prediction:
    return dspy.Prediction(handle="alice", score=42)


if __name__ == "__main__":
    p = build("hello")
    fields = [name for name, field in Profile.fields.items() if field.json_schema_extra.get("__dspy_field_type") == "output"]
    print(f"output={p!r} sig_outputs={fields}")
