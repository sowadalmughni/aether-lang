"""Build a small, deduped subset of InjecAgent test cases for the security
benchmark.

Source / attribution
--------------------
Upstream: https://github.com/uiuc-kang-lab/InjecAgent (MIT-licensed)
Paper:    "InjecAgent: Benchmarking Indirect Prompt Injections in
          Tool-Integrated Large Language Model Agents"
          arXiv:2403.02691

We vendor a 40-case subset (20 DH + 20 DS, deduped on
`(User Instruction, Attacker Instruction)`). Per case we keep only the
attacker-instruction string, expected-achievement descriptor, and
realized tool-response content — the fields where the injection lands.
The tool-dispatch scaffolding (Tool Parameters / Thought / etc.) is
discarded because Aether's compile-time taint tracking targets
prompt-level taint flow, not tool-dispatch flow.

Regeneration
------------
gh api -H "Accept: application/vnd.github.raw" \\
  "repos/uiuc-kang-lab/InjecAgent/contents/data/test_cases_dh_base.json" \\
  > _tmp_dh_base_full.json
gh api -H "Accept: application/vnd.github.raw" \\
  "repos/uiuc-kang-lab/InjecAgent/contents/data/test_cases_ds_base.json" \\
  > _tmp_ds_base_full.json
python _make_subset.py
rm _tmp_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
TARGETS = {
    "DH": HERE / "_tmp_dh_base_full.json",
    "DS": HERE / "_tmp_ds_base_full.json",
}
OUT = HERE / "injec_agent_subset_v1.json"

PER_BUCKET = 20

PROVENANCE = {
    "source_repo": "https://github.com/uiuc-kang-lab/InjecAgent",
    "source_license": "MIT",
    "source_paper": "arXiv:2403.02691",
    "subset_method": (
        "First 20 deduped cases per bucket from test_cases_{dh,ds}_base.json. "
        "Tool-dispatch fields stripped; only attacker-instruction string, "
        "expected-achievement descriptor, attack category, user instruction, "
        "and realized tool response retained."
    ),
}


def main() -> int:
    cases: list[dict] = []
    for bucket, path in TARGETS.items():
        full = json.loads(path.read_text(encoding="utf-8"))
        seen: set[tuple[str, str]] = set()
        bucket_cases: list[dict] = []
        for raw in full:
            user_instr = raw.get("User Instruction", "").strip()
            atk_instr = raw.get("Attacker Instruction", "").strip()
            tool_resp = raw.get("Tool Response", "").strip()
            if not user_instr or not atk_instr or not tool_resp:
                continue
            key = (user_instr, atk_instr)
            if key in seen:
                continue
            seen.add(key)
            bucket_cases.append(
                {
                    "id": f"{bucket.lower()}_{len(bucket_cases):03d}",
                    "attack_bucket": bucket,
                    "user_instruction": user_instr,
                    "attacker_instruction": atk_instr,
                    "expected_achievement": raw.get("Expected Achievements", "").strip(),
                    "attack_category": raw.get("Attack Type", "").strip(),
                    "tool_response": tool_resp,
                }
            )
            if len(bucket_cases) >= PER_BUCKET:
                break
        cases.extend(bucket_cases)
    payload = {"_provenance": PROVENANCE, "cases": cases}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {OUT}")
    print(f"file size: {OUT.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
