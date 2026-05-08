"""dd_01 (LangChain): two RunnableSequence chains bound to the same name.

Aether catches this at compile time as DuplicateDefinition. Python silently
binds the second definition and the first is unreachable -- no error, wrong
behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

llm = MockChatModel()

classify = ChatPromptTemplate.from_template("Classify A: {text}") | llm | StrOutputParser()
# bug: redefining `classify` silently shadows the first chain.
classify = ChatPromptTemplate.from_template("Classify B: {text}") | llm | StrOutputParser()


def analyze(input_text: str):
    return classify.invoke({"text": input_text})


if __name__ == "__main__":
    out = analyze("hello")
    print(f"output={out!r}")
