"""ur_03 (LC): let-binding rhs references a symbol that does not exist.

Aether catches at compile time. Python raises NameError when control reaches the
function call.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

llm = MockChatModel()
echo = ChatPromptTemplate.from_template("Echo: {text}") | llm | StrOutputParser()


def run(input_text: str):
    return echo.invoke({"text": ghost_value})  # noqa: F821  bug: undefined


if __name__ == "__main__":
    print(run("hello"))
