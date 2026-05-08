"""dd_05 (DSPy): an Enum class declares the same variant name twice."""
from __future__ import annotations
import sys
from enum import Enum
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402  (parity)


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGH = 4  # noqa: F811  bug


def classify(text: str) -> Severity:
    return Severity.HIGH


if __name__ == "__main__":
    s = classify("hello")
    print(f"output={s} value={s.value}")
