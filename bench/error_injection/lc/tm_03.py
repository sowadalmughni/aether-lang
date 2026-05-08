"""tm_03 (LC): passes list[str] where string arg is required.

Aether catches as TypeMismatch. LangChain's chain.invoke({"text": list_value})
silently formats the list into the prompt -- no validation.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

llm = MockChatModel()
extract_chain = ChatPromptTemplate.from_template("Extract: {text}") | llm | StrOutputParser()
summarize_chain = ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()


def run(input_text: str) -> str:
    parts = ["alpha", "beta"]  # would be extract_chain.invoke({"text": input_text})
    s = summarize_chain.invoke({"text": parts})  # bug: list passed as string
    return s


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r}")
