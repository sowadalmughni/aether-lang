"""tm_05 (LC): chain returns plain string where Item struct is expected.

Aether catches as TypeMismatch. Python silently passes the string.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class Item(BaseModel):
    name: str


def make_item(text: str) -> str:
    _ = MockChatModel().invoke(text)
    return "alpha"


def run(input_text: str) -> Item:
    return make_item(input_text)  # bug: str, declared Item


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
