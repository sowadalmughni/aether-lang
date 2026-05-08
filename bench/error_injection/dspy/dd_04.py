"""dd_04 (DSPy): function with same parameter name twice. Python SyntaxError."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402  (parity)


def answer(question: str, question: str) -> str:  # type: ignore[no-redef]  bug
    return question


if __name__ == "__main__":
    print(answer("hello", "world"))
