"""tm_06 (LC): returns the Wrapped struct where its inner string is expected.

Aether catches as TypeMismatch. Python silently returns the struct instance.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class Wrapped(BaseModel):
    value: str


def wrap(text: str) -> Wrapped:
    _ = MockChatModel().invoke(text)
    return Wrapped(value="hello")


def run(input_text: str) -> str:
    return wrap(input_text)  # bug: Wrapped, declared str


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
