"""mf_04 (LC): accessing a misspelled field name on an existing struct.

Aether catches as UnknownField. Python raises AttributeError.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class Order(BaseModel):
    id: str
    total: int


def build_order(text: str) -> Order:
    _ = MockChatModel().invoke(text)
    return Order(id="ord-1", total=100)


def run(input_text: str) -> int:
    o = build_order(input_text)
    return o.totl  # bug: typo of `total`


if __name__ == "__main__":
    print(run("hello"))
