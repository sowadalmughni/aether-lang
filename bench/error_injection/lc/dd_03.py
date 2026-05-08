"""dd_03 (LC): a Pydantic model with two same-named field annotations.

Aether catches as DuplicateField. Python: the second annotation silently
overrides the first; Pydantic does not raise. This is the dangerous case --
no signal that one field declaration was discarded.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _mock_llm import MockChatModel  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class Account(BaseModel):
    id: str
    name: str
    name: str  # noqa: F811  bug: duplicate field, second silently wins


def build(text: str) -> Account:
    _ = MockChatModel().invoke(text)
    return Account(id="acct-1", name="alice")


if __name__ == "__main__":
    a = build("hello")
    print(f"output={a!r} fields={list(a.model_fields.keys())}")
