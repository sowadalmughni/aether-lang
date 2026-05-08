"""Loader for the vendored InjecAgent attack subset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

DATA_PATH = Path(__file__).parent / "data" / "injec_agent_subset_v1.json"


@dataclass(frozen=True)
class AttackCase:
    """A single direct-prompt-injection test case.

    Fields mirror the vendored JSON. `attacker_instruction` is the
    injection content and `expected_achievement` is the attacker's
    declared goal — the judge uses the latter to decide whether the
    LLM's response complied with the attacker.
    """

    id: str
    attack_bucket: str  # "DH" or "DS"
    attack_category: str
    user_instruction: str
    attacker_instruction: str
    expected_achievement: str
    tool_response: str


def load_cases(path: Path = DATA_PATH) -> List[AttackCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload["cases"]
    return [AttackCase(**c) for c in raw_cases]


def benign_variant_tool_response(case: AttackCase) -> str:
    """Strip the attacker instruction from the tool response, leaving a
    plausible-looking benign payload. The InjecAgent test cases place
    the attacker instruction verbatim inside `tool_response`; replacing
    it with a neutral notice gives a paired benign-control input.
    """
    benign = case.tool_response.replace(
        case.attacker_instruction,
        "(no notable content; standard payload elided for brevity)",
    )
    return benign
