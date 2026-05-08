"""Minimal deterministic mock LLMs for the type-safety ablation.

Each error-injection .py file exercises a specific bug pattern in either
LangChain or DSPy. The point is to test framework-level type checking, not
LLM behavior, so the mock returns a fixed string (or a fixed JSON-shaped
string) at zero latency. The buggy workflow then either crashes at runtime
(framework validation catches the bug) or completes silently with wrong
output (framework validation misses the bug).

LC mock: BaseChatModel subclass with no latency.
DSPy mock: dspy.LM subclass returning a single completion in OpenAI shape.

Using these mocks instead of real APIs keeps the ablation deterministic,
free, and re-runnable on CI.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import dspy

MOCK_TEXT = "mock-output"
MOCK_JSON_OBJECT = '{"answer": "mock-output", "confidence": 7}'
MOCK_JSON_LIST = '["mock-a", "mock-b"]'


class MockChatModel(BaseChatModel):
    """Always returns MOCK_TEXT. No latency. Used by all LC injection files."""

    fixed_text: str = MOCK_TEXT

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.fixed_text))]
        )


class MockLM(dspy.LM):
    """DSPy LM that returns a single fixed completion. No latency."""

    def __init__(self, fixed_text: str = MOCK_TEXT) -> None:
        super().__init__(model="mock", model_type="chat")
        self.fixed_text = fixed_text
        self.kwargs = {"temperature": 0.0, "max_tokens": 256}
        self.history: list[dict] = []

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> list[str]:
        return [self.fixed_text]


def configure_dspy_mock(fixed_text: str = MOCK_TEXT) -> None:
    """Wire the MockLM into dspy's global config. Idempotent."""
    dspy.configure(lm=MockLM(fixed_text=fixed_text))
