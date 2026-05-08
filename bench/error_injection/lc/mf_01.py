"""mf_01 (LangChain): accessing a field that does not exist on a Pydantic profile.

Aether catches this at compile time as UnknownField. Python raises AttributeError
when control reaches the field access.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402

from pydantic import BaseModel  # noqa: E402


class Profile(BaseModel):
    name: str
    age: int


def make_profile(text: str) -> Profile:
    _ = MockChatModel().invoke(text)
    return Profile(name="alice", age=30)


def show(input_text: str) -> str:
    p = make_profile(input_text)
    return p.email  # bug: Profile has no `email` field


if __name__ == "__main__":
    print(show("hello"))
