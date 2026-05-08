"""ur_07 (LC): second flow's first statement references an undefined name.

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
shape = ChatPromptTemplate.from_template("Shape: {text}") | llm | StrOutputParser()


def primary(input_text: str):
    return shape.invoke({"text": input_text})


def secondary(input_text: str):
    return shape.invoke({"text": missing_input})  # noqa: F821  bug


if __name__ == "__main__":
    primary("ok")
    print(secondary("hello"))  # exercises the bug
