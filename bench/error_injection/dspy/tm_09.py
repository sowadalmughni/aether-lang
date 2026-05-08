"""tm_09 (DSPy): passes list[str] where Bundle struct argument is required."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class BundleSig(dspy.Signature):
    body: str = dspy.InputField()
    out: str = dspy.OutputField()


def extract(text: str) -> list[str]:
    return ["a", "b"]


def finalize(bundle: str) -> str:
    return f"final:{bundle}"


def run(input_text: str) -> str:
    parts = extract(input_text)
    return finalize(parts)  # type: ignore[arg-type]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r}")
