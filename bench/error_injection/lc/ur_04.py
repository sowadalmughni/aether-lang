"""ur_04 (LC): middle of a 3-step pipeline calls a missing function.

Aether catches at compile time. Python raises NameError at runtime.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

llm = MockChatModel()
first = ChatPromptTemplate.from_template("First: {text}") | llm | StrOutputParser()
third = ChatPromptTemplate.from_template("Third: {text}") | llm | StrOutputParser()


def run(input_text: str):
    a = first.invoke({"text": input_text})
    b = missing_middle(a)  # noqa: F821  bug: undefined function
    c = third.invoke({"text": b})
    return c


if __name__ == "__main__":
    print(run("hello"))
