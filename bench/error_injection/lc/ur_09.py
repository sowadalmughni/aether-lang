"""ur_09 (LC): second flow calls an absent helper function.

Aether catches at compile time. Python raises NameError when the second flow
is invoked.
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


def primary(input_text: str):
    return first.invoke({"text": input_text})


def secondary(input_text: str):
    return absent_helper(input_text)  # noqa: F821  bug


if __name__ == "__main__":
    primary("ok")
    print(secondary("hello"))
