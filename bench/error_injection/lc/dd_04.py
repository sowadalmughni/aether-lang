"""dd_04 (LC): function declares the same parameter name twice.

Aether catches as DuplicateParameter. Python raises a SyntaxError at parse
time -- which is technically "compile time" for the bytecode compiler, but the
error surfaces only when the file is loaded. The runner classifies SyntaxError
as runtime detection (the file doesn't run); the per-bug breakdown shows the
distinction from Aether's pre-execution semantic check.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402  (kept for parity with siblings)


def answer(question: str, question: str) -> str:  # type: ignore[no-redef]  bug
    return question


if __name__ == "__main__":
    print(answer("hello", "world"))
