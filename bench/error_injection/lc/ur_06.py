"""ur_06 (LC): second let-binding references a never-declared symbol.

Aether catches at compile time. Python raises NameError when the second call
is reached.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

llm = MockChatModel()
step = ChatPromptTemplate.from_template("Step: {text}") | llm | StrOutputParser()


def run(input_text: str):
    a = step.invoke({"text": input_text})
    b = step.invoke({"text": unbound_intermediate})  # noqa: F821  bug
    return b


if __name__ == "__main__":
    print(run("hello"))
