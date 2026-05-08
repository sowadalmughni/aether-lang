"""tm_07 (LC): enum value returned where flow declares string output.

Aether catches as TypeMismatch. Python returns the enum silently.
"""
from __future__ import annotations
import sys
from enum import Enum
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402


class Status(Enum):
    OK = 1
    ERROR = 2
    PENDING = 3


def check(text: str) -> Status:
    _ = MockChatModel().invoke(text)
    return Status.OK


def run(input_text: str) -> str:
    return check(input_text)  # bug: Status, declared str


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
