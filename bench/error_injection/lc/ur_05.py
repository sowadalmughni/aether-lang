"""ur_05 (LC): typoed function name (typo of `polish`).

Aether catches at compile time with did-you-mean suggestion. Python raises
NameError at runtime.
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


def polish(text: str) -> str:
    return text.upper()


def run(input_text: str):
    s = summarize.invoke({"text": input_text})
    return polishe(s)  # noqa: F821  bug: typo of `polish`


if __name__ == "__main__":
    print(run("hello"))
