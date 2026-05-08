"""tm_08 (DSPy): returns list[str] where flow declares list[int]."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class ExtractIdsSig(dspy.Signature):
    text: str = dspy.InputField()
    ids: list[str] = dspy.OutputField()


def extract_ids(text: str) -> list[str]:
    return ["1", "2", "3"]


def run(input_text: str) -> list[int]:
    return extract_ids(input_text)  # type: ignore[return-value]  bug


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r} elem_type={type(out[0]).__name__}")
