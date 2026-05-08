"""ur_02 (LC): returns a variable that was never bound.

Aether catches as UndefinedSymbol at compile time. Python raises NameError.
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


def run(input_text: str):
    s = summarize.invoke({"text": input_text})
    return result  # noqa: F821  bug: never declared


if __name__ == "__main__":
    print(run("hello"))
