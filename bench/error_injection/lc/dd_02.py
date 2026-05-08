"""dd_02 (LC): two Pydantic models share the same name.

Aether catches as DuplicateDefinition. Python silently overrides; the first
class is unreachable.
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


class Profile(BaseModel):  # noqa: F811  bug: silent override
    handle: str
    score: int


def build(text: str) -> Profile:
    _ = MockChatModel().invoke(text)
    return Profile(handle="alice", score=42)


if __name__ == "__main__":
    p = build("hello")
    print(f"output={p!r} fields={list(p.model_fields.keys())}")
