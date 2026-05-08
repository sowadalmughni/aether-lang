"""mf_02 (DSPy): caller passes too few args to a 2-arg function."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dspy  # noqa: E402


class AnswerSig(dspy.Signature):
    question: str = dspy.InputField()
    ctx: str = dspy.InputField()
    answer: str = dspy.OutputField()


def answer(question: str, ctx: str) -> str:
    return f"{question}|{ctx}"


def run(input_text: str) -> str:
    return answer(input_text)  # type: ignore[call-arg]  bug


if __name__ == "__main__":
    print(run("hello"))
