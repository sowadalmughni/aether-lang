"""tm_09 (LC): passes list[str] where Bundle struct argument is required.

Aether catches as TypeMismatch on the function call. Python's invoke({...})
forwards the list into the prompt and the chain runs to completion silently.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class Bundle(BaseModel):
    body: str


llm = MockChatModel()
finalize_chain = ChatPromptTemplate.from_template("Finalize: {bundle}") | llm | StrOutputParser()


def extract(text: str) -> list[str]:
    return ["a", "b"]


def run(input_text: str) -> str:
    parts = extract(input_text)
    return finalize_chain.invoke({"bundle": parts})  # bug: list[str], expected Bundle


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r}")
