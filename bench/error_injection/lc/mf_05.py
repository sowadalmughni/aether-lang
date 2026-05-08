"""mf_05 (LC): caller passes too many args to a 2-arg function.

Aether catches as ArgumentCountMismatch. Python raises TypeError.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402


def join(a: str, b: str) -> str:
    _ = MockChatModel().invoke(a + b)
    return a


def step(text: str) -> str:
    return text


def run(input_text: str) -> str:
    s = step(input_text)
    return join(s, input_text, input_text)  # type: ignore[call-arg]  bug: 3 args


if __name__ == "__main__":
    print(run("hello"))
