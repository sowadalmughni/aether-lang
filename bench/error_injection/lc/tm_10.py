"""tm_10 (LC): passes UserMessage where SystemMessage is expected.

Both have the same field names but different identities. Aether catches the
nominal-type mismatch. Python is structurally typed for dicts/Pydantic models
with matching shapes -- no error.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class UserMessage(BaseModel):
    body: str


class SystemMessage(BaseModel):
    body: str


def shape_user(text: str) -> UserMessage:
    _ = MockChatModel().invoke(text)
    return UserMessage(body=text)


def deliver_system(msg: SystemMessage) -> str:
    return msg.body


def run(input_text: str) -> str:
    u = shape_user(input_text)
    return deliver_system(u)  # bug: UserMessage passed where SystemMessage expected


if __name__ == "__main__":
    out = run("hello")
    print(f"output={out!r}")
