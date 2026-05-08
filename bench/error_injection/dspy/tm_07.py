"""tm_07 (DSPy): enum value returned where flow declares string output."""
from __future__ import annotations
import sys
from enum import Enum
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class Status(Enum):
    OK = 1
    ERROR = 2
    PENDING = 3


class CheckSig(dspy.Signature):
    text: str = dspy.InputField()
    status: Status = dspy.OutputField()


def check(text: str) -> Status:
    return Status.OK


def run(input_text: str) -> str:
    return check(input_text)  # type: ignore[return-value]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
