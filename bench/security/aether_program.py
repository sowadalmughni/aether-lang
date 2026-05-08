"""Synthesize a vulnerable Aether program for a given attack case and
invoke `aetherc check` against it.

A "vulnerable" program is one where the parameter carrying the
attacker-controlled tool response is marked `@untrusted` *and* is
referenced directly in an LLM prompt template without `sanitize(...)`.
Pass 6 (taint analysis) rejects these programs; Pass 6 disabled
accepts them.

This module is purely about the *static* outcome — whether the program
compiles. The runtime evaluation against a real LLM is handled by
`runners.py`.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .dataset import AttackCase

REPO_ROOT = Path(__file__).resolve().parents[2]
AETHERC_CMD = ["cargo", "run", "-q", "-p", "aether-compiler", "--bin", "aetherc", "--"]


def render_vulnerable_program(_case: AttackCase) -> str:
    """A minimal vulnerable Aether program: an `@untrusted` tool
    response is interpolated into an LLM prompt with no sanitization.
    Pass 6 must reject this; Pass 6 disabled must accept it.

    The test case content is *not* embedded in the source; the program
    is generic and only the runtime payload differs per case. This
    matches how a real application would be structured: the program
    is a fixed shape, the data flows through it.
    """
    return (
        '''llm fn answer(@untrusted user_query: string, @untrusted tool_response: string) -> string {
    model: "gpt-4o-mini",
    prompt: "User asked: {{user_query}}\\nTool returned: {{tool_response}}\\nRespond appropriately."
}

flow handle(@untrusted user_query: string, @untrusted tool_response: string) -> string {
    let reply = answer(user_query, tool_response);
    return reply;
}
'''
    )


def render_sanitized_program(_case: AttackCase) -> str:
    """A sanitized variant — both untrusted inputs are routed through
    `sanitize(...)` before reaching the prompt. Pass 6 accepts this;
    runtime ASR with v1's identity-sanitize is comparable to taint_off
    (the v1 sanitize is a no-op at runtime).
    """
    return (
        '''llm fn answer(@untrusted user_query: string, @untrusted tool_response: string) -> string {
    model: "gpt-4o-mini",
    prompt: "User asked: {{user_query}}\\nTool returned: {{tool_response}}\\nRespond appropriately."
}

flow handle(@untrusted user_query: string, @untrusted tool_response: string) -> string {
    let cleaned_query = sanitize(user_query);
    let cleaned_tool = sanitize(tool_response);
    let reply = answer(cleaned_query, cleaned_tool);
    return reply;
}
'''
    )


def run_aetherc_check(source: str, *, no_taint_check: bool) -> tuple[int, str]:
    """Write `source` to a temp file and run `aetherc check` against
    it. Returns (exit_code, stderr).

    PATH must already include the MinGW bin directory so the cargo
    subprocess can link Windows targets — the orchestrator sets this
    up once at startup.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".aether", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(source)
        tmp_path = tf.name
    try:
        cmd = list(AETHERC_CMD) + ["check"]
        if no_taint_check:
            cmd.append("--no-taint-check")
        cmd.append(tmp_path)
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        # aetherc emits diagnostics on stderr.
        return proc.returncode, proc.stderr or proc.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)
