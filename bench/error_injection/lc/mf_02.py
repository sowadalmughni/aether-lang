"""mf_02 (LC): caller passes too few args to a 2-arg function.

Aether catches as ArgumentCountMismatch. Python raises TypeError.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402


def answer(question: str, ctx: str) -> str:
    _ = MockChatModel().invoke(question + ctx)
    return "ok"


def run(input_text: str) -> str:
    return answer(input_text)  # type: ignore[call-arg]  bug: missing ctx


if __name__ == "__main__":
    print(run("hello"))
