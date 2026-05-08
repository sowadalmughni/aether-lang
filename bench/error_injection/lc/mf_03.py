"""mf_03 (LC): accessing a nested struct field that does not exist on the inner type.

Aether catches as UnknownField. Python raises AttributeError on Pydantic models.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class Address(BaseModel):
    city: str
    zip: str


class Person(BaseModel):
    name: str
    address: Address


def build(text: str) -> Person:
    _ = MockChatModel().invoke(text)
    return Person(name="alice", address=Address(city="seattle", zip="98101"))


def run(input_text: str) -> str:
    p = build(input_text)
    return p.address.country  # bug: Address has no `country` field


if __name__ == "__main__":
    print(run("hello"))
