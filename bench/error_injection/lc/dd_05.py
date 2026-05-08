"""dd_05 (LC): an Enum class declares the same variant name twice.

Aether catches as DuplicateVariant. Python: defining the same name twice in
an Enum body is allowed -- the second binding silently overrides the first
inside the class body, and the resulting Enum has only one member with that
name. No exception, dangerous miss.
"""
from __future__ import annotations
import sys
from enum import Enum
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402  (parity)


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGH = 4  # noqa: F811  bug: silent override; HIGH ends up == 4


def classify(text: str) -> Severity:
    _ = MockChatModel().invoke(text)
    return Severity.HIGH


if __name__ == "__main__":
    s = classify("hello")
    print(f"output={s} value={s.value}")
