"""tm_08 (LC): returns list[str] where flow declares list[int].

Aether catches as TypeMismatch on the inner generic. Python returns silently.
A downstream consumer that does arithmetic on list elements would crash later;
this file demonstrates the silent miss at the boundary.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402


def extract_ids(text: str) -> list[str]:
    _ = MockChatModel().invoke(text)
    return ["1", "2", "3"]


def run(input_text: str) -> list[int]:
    return extract_ids(input_text)  # bug: list[str], declared list[int]


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} elem_type={type(out[0]).__name__}")
