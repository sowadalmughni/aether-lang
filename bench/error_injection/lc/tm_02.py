"""tm_02 (LC): flow returns list[str] where a string is declared.

Aether catches as TypeMismatch. Python returns the wrong type silently.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402

llm = MockChatModel(fixed_text='["a", "b", "c"]')


def extract(text: str):
    return ["a", "b", "c"]


def run(input_text: str) -> str:
    return extract(input_text)  # bug: returns list[str], declared str


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} type={type(out).__name__}")
