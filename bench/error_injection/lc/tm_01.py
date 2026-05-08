"""tm_01 (LangChain): flow returns a string literal where a Result struct is expected.

Aether catches this at compile time as TypeMismatch. LangChain has no compile-time
type system; this RunnableLambda-style workflow returns the wrong-shaped value and
the framework happily forwards it to the caller. Silent miss.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

llm = MockChatModel()
classify_chain = ChatPromptTemplate.from_template("Classify: {text}") | llm | StrOutputParser()


def analyze(input_text: str):
    # .aether equivalent declares: flow analyze(input: string) -> Result
    # bug: returns a plain string instead of {"answer": str, "confidence": int}.
    _ = classify_chain.invoke({"text": input_text})
    return "wrong return type"


if __name__ == "__main__":
    out = analyze("hello world")
    print(f"output={out!r} type={type(out).__name__}")
