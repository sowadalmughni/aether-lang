"""ur_01 (LangChain): pipeline calls a function that does not exist.

Aether catches this at compile time as UndefinedFunction. Python raises NameError
at runtime when control reaches the call site.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

llm = MockChatModel()
summarize = ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()


def analyze(input_text: str):
    s = summarize.invoke({"text": input_text})
    t = nonexistent_function(s)  # noqa: F821  bug: undefined symbol
    return t


if __name__ == "__main__":
    print(analyze("hello"))
