"""tm_03 (DSPy): passes list[str] where string arg is required.

Aether catches as TypeMismatch. DSPy's signature call accepts the list and
formats it into the prompt; no validation.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class SummarizeSig(dspy.Signature):
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()


def summarize(text: str) -> str:
    return f"summary-of-{text}"


def run(input_text: str) -> str:
    parts: list[str] = ["alpha", "beta"]
    return summarize(parts)  # type: ignore[arg-type]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r}")
