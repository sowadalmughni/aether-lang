"""Thin wrapper around the OpenAI client with cost tracking + a hard
ceiling. Mirrors the pattern in bench/baselines/langchain_baseline.py
(BASELINE_PROVIDER, --confirm-cost, $5 ceiling).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

# Pricing per 1M tokens (USD) for gpt-4o-mini, as of 2026-05.
# Keeping these explicit rather than reading from an external table so
# the reproducible cost tally is fully deterministic from the source.
PRICING = {
    "gpt-4o-mini": {"input_per_1m": 0.150, "output_per_1m": 0.600},
}

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_S = 60


@dataclass
class CostMeter:
    spent_usd: float = 0.0
    cap_usd: float = 5.0
    call_count: int = 0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rates = PRICING.get(model)
        if not rates:
            return 0.0
        cost = (
            prompt_tokens * rates["input_per_1m"] / 1_000_000
            + completion_tokens * rates["output_per_1m"] / 1_000_000
        )
        self.spent_usd += cost
        self.call_count += 1
        by = self.by_model.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        )
        by["calls"] += 1
        by["prompt_tokens"] += prompt_tokens
        by["completion_tokens"] += completion_tokens
        by["cost_usd"] += cost
        return cost

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)


class CostCeilingExceeded(RuntimeError):
    pass


@dataclass
class LlmResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


def _client_lazy_init():
    # Imported lazily so that pure-static configs (aether_taint_on, which
    # does no LLM calls) can run without openai installed.
    from openai import OpenAI  # type: ignore

    return OpenAI()


def call_llm(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    cost_meter: CostMeter,
    max_tokens: int = 256,
    temperature: float = 0.0,
    client=None,
) -> LlmResponse:
    """Make a single chat-completion call. Updates the shared cost
    meter; raises CostCeilingExceeded if the cap would be exceeded.
    """
    if cost_meter.spent_usd >= cost_meter.cap_usd:
        raise CostCeilingExceeded(
            f"cost ceiling ${cost_meter.cap_usd:.2f} reached "
            f"(spent ${cost_meter.spent_usd:.4f})"
        )
    if client is None:
        client = _client_lazy_init()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=DEFAULT_TIMEOUT_S,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = (resp.choices[0].message.content or "").strip()
    pt = resp.usage.prompt_tokens
    ct = resp.usage.completion_tokens
    cost = cost_meter.record(model, pt, ct)
    return LlmResponse(
        text=text, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost, latency_ms=latency_ms
    )


def require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set. The security benchmark requires a real LLM "
            "for ASR to be meaningful (mock LLMs trivially pass)."
        )
