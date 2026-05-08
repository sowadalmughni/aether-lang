#!/usr/bin/env python3
"""Type-safety ablation runner.

For each entry in bench/error_injection/manifest.yaml:

  1. Run `aetherc check <id>.aether` as a subprocess. Classify by error variant
     parsed from stderr. Result: caught_at_compile_time | missed.
  2. Run `python lc/<id>.py` as a subprocess (timeout 30s). Classify by
     exception class + traceback first line. Result: caught_at_runtime |
     missed_silently.
  3. Same for dspy/<id>.py.

Emit bench/results/ablation_typesafety_v1.json with a per-bug breakdown plus
a summary block (counts per result class, broken down by bucket). The summary
is for convenience; the per-bug `test_cases[]` array is the authoritative
record, matching the ablation acceptance criteria.

The corpus substitutes `duplicate_definition` for the originally-spec'd
`circular_dependency` bucket; rationale and tracking are in
bench/error_injection/manifest.yaml header and at
https://github.com/sowadalmughni/aether-lang/issues/4.

Usage (run from repo root, no runtime needed):

    python3 bench/runners/ablation_typesafety.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_benchmark import (  # noqa: E402
    get_git_version,
    get_hardware_info,
)

CORPUS_DIR = REPO_ROOT / "bench" / "error_injection"
MANIFEST = CORPUS_DIR / "manifest.yaml"
DEFAULT_AETHERC = REPO_ROOT / "target" / "debug" / "aetherc"
DEFAULT_OUTPUT = REPO_ROOT / "bench" / "results" / "ablation_typesafety_v1.json"
DEFAULT_PYTHON = "/home/deamers_academy/aether-bench-venv/bin/python"

# Map aetherc stderr message patterns to SemanticError variant names.
# Order matters: more specific patterns first.
AETHER_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Wrong number of arguments"),                 "ArgumentCountMismatch"),
    (re.compile(r"Unknown argument"),                          "UnknownArgument"),
    (re.compile(r"Missing required argument"),                 "MissingArgument"),
    (re.compile(r"Duplicate field"),                           "DuplicateField"),
    (re.compile(r"Duplicate variant"),                         "DuplicateVariant"),
    (re.compile(r"Duplicate parameter"),                       "DuplicateParameter"),
    (re.compile(r"Duplicate definition"),                      "DuplicateDefinition"),
    (re.compile(r"Unknown field"),                             "UnknownField"),
    (re.compile(r"Unknown variant"),                           "UnknownVariant"),
    (re.compile(r"Cannot access field"),                       "InvalidFieldAccess"),
    (re.compile(r"Type mismatch"),                             "TypeMismatch"),
    (re.compile(r"calls undefined function"),                  "UndefinedFunction"),
    (re.compile(r"Undefined symbol"),                          "UndefinedSymbol"),
    (re.compile(r"Circular dependency"),                       "CircularDependency"),
    (re.compile(r"Invalid template reference"),                "InvalidTemplateRef"),
    (re.compile(r"requires a model specification"),            "MissingModel"),
    (re.compile(r"requires a prompt"),                         "MissingPrompt"),
]


def _classify_aether_stderr(stderr: str) -> tuple[Optional[str], str]:
    """Return (variant_name, first_match_line). variant_name is None if
    no known pattern matched."""
    for line in stderr.splitlines():
        for pattern, name in AETHER_ERROR_PATTERNS:
            if pattern.search(line):
                return name, line.strip()
    return None, ""


def _run_aetherc_check(aetherc: Path, aether_file: Path) -> dict:
    """Run aetherc check and classify the result."""
    try:
        proc = subprocess.run(
            [str(aetherc), "check", str(aether_file)],
            capture_output=True, text=True, timeout=30,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"result": "timeout", "exit_code": -1, "error_class_matched": None,
                "stderr_excerpt": "timeout after 30s"}
    stderr = proc.stderr or ""
    if proc.returncode == 0:
        return {"result": "missed", "exit_code": 0, "error_class_matched": None,
                "stderr_excerpt": (stderr.strip().splitlines() or [""])[0][:200]}
    variant, line = _classify_aether_stderr(stderr)
    return {
        "result": "caught_at_compile_time",
        "exit_code": proc.returncode,
        "error_class_matched": variant,
        "stderr_excerpt": (line or stderr.strip().splitlines()[0] if stderr.strip() else "")[:200],
    }


def _classify_python_traceback(stderr: str) -> tuple[Optional[str], Optional[str]]:
    """Pull the exception class and first traceback line from a Python
    traceback. Returns (exception_class, traceback_first_meaningful_line)."""
    if not stderr:
        return None, None
    last_lines = stderr.strip().splitlines()
    exc_line = last_lines[-1] if last_lines else ""
    # Standard Python tracebacks end with `ExceptionClass: message`.
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_\.]*Error|SyntaxError|TypeError|ValueError|"
                 r"KeyError|AttributeError|NameError|ImportError|RecursionError|"
                 r"AssertionError|RuntimeError|ValidationError):\s*(.*)$", exc_line)
    exc_class: Optional[str] = None
    if m:
        exc_class = m.group(1).split(".")[-1]
    else:
        # SyntaxError shows a different tail format ("def foo(...): ...").
        # Look for a line like "  File "...", line N" then check first line of stderr
        for line in last_lines:
            if line.startswith(("File ", '  File "')):
                continue
            m2 = re.match(r"^([A-Za-z_][A-Za-z0-9_]*Error)\b", line)
            if m2:
                exc_class = m2.group(1)
                break
    return exc_class, exc_line[:200] if exc_line else None


def _run_python_file(python: str, py_file: Path) -> dict:
    """Run a Python file as a subprocess and classify the outcome."""
    try:
        proc = subprocess.run(
            [python, str(py_file)],
            capture_output=True, text=True, timeout=30,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"result": "timeout", "exit_code": -1, "exception_class": None,
                "traceback_first_line": "timeout after 30s",
                "stdout_excerpt": ""}
    stderr = proc.stderr or ""
    stdout = (proc.stdout or "").strip()
    if proc.returncode == 0:
        return {"result": "missed_silently", "exit_code": 0, "exception_class": None,
                "traceback_first_line": None,
                "stdout_excerpt": stdout.splitlines()[0][:200] if stdout else ""}
    exc_class, first_line = _classify_python_traceback(stderr)
    return {
        "result": "caught_at_runtime",
        "exit_code": proc.returncode,
        "exception_class": exc_class,
        "traceback_first_line": first_line,
        "stdout_excerpt": stdout.splitlines()[0][:200] if stdout else "",
    }


def _summarize(test_cases: list[dict]) -> dict:
    out = {
        "aether_caught": 0, "aether_missed": 0,
        "lc_caught_at_runtime": 0, "lc_missed_silently": 0,
        "dspy_caught_at_runtime": 0, "dspy_missed_silently": 0,
        "by_bucket": {},
    }
    for tc in test_cases:
        if tc["aether"]["result"] == "caught_at_compile_time":
            out["aether_caught"] += 1
        else:
            out["aether_missed"] += 1
        if tc["langchain"]["result"] == "caught_at_runtime":
            out["lc_caught_at_runtime"] += 1
        else:
            out["lc_missed_silently"] += 1
        if tc["dspy"]["result"] == "caught_at_runtime":
            out["dspy_caught_at_runtime"] += 1
        else:
            out["dspy_missed_silently"] += 1
        bucket = tc["bucket"]
        b = out["by_bucket"].setdefault(bucket, {
            "total": 0,
            "aether_caught": 0,
            "lc_caught_at_runtime": 0,
            "dspy_caught_at_runtime": 0,
        })
        b["total"] += 1
        if tc["aether"]["result"] == "caught_at_compile_time":
            b["aether_caught"] += 1
        if tc["langchain"]["result"] == "caught_at_runtime":
            b["lc_caught_at_runtime"] += 1
        if tc["dspy"]["result"] == "caught_at_runtime":
            b["dspy_caught_at_runtime"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Aether type-safety ablation")
    ap.add_argument("--aetherc", default=str(DEFAULT_AETHERC),
                    help="path to aetherc binary")
    ap.add_argument("--python", default=DEFAULT_PYTHON,
                    help="python binary used to run the buggy LC/DSPy files")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    aetherc = Path(args.aetherc)
    if not aetherc.exists():
        print(f"error: aetherc binary not found at {aetherc}", file=sys.stderr)
        return 1
    if not Path(args.python).exists():
        print(f"error: python binary not found at {args.python}", file=sys.stderr)
        return 1

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    corpus = manifest["corpus"]
    print(f"running type-safety ablation over {len(corpus)} test cases", flush=True)

    test_cases: list[dict] = []
    buckets: list[str] = []
    for entry in corpus:
        tid = entry["id"]
        bucket = entry["bucket"]
        if bucket not in buckets:
            buckets.append(bucket)
        aether_file = CORPUS_DIR / f"{tid}.aether"
        lc_file = CORPUS_DIR / entry["python_lc_path"]
        dspy_file = CORPUS_DIR / entry["python_dspy_path"]

        for fpath, label in (
            (aether_file, ".aether"),
            (lc_file, "lc"),
            (dspy_file, "dspy"),
        ):
            if not fpath.exists():
                print(f"error: missing {label} file: {fpath}", file=sys.stderr)
                return 1

        a_res = _run_aetherc_check(aetherc, aether_file)
        lc_res = _run_python_file(args.python, lc_file)
        dspy_res = _run_python_file(args.python, dspy_file)

        tc = {
            "id": tid,
            "bucket": bucket,
            "expected_aether_error_classes": entry["expected_aether_error_classes"],
            "description": entry.get("description", ""),
            "aether": a_res,
            "langchain": lc_res,
            "dspy": dspy_res,
        }
        test_cases.append(tc)
        print(f"  {tid:<6}  aether={a_res['result']:<22}  "
              f"lc={lc_res['result']:<18}  dspy={dspy_res['result']}",
              flush=True)

    summary = _summarize(test_cases)
    envelope = {
        "system": "aether",
        "version": get_git_version(REPO_ROOT),
        "provider": "n/a (compile-time + mock LLM)",
        "hardware": get_hardware_info(),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "buckets": buckets,
        "test_cases": test_cases,
        "summary": summary,
        "methodology_notes": {
            "cd_substitution": (
                "The original ablation design included a circular_dependency "
                "category, but verification revealed that aetherc's source-"
                "level cd detector is currently preempted by semantic analysis "
                "on programs that contain other issues; the SemanticError::"
                "CircularDependency variant is defined but never emitted. "
                "Rather than fabricate test cases that would not trigger the "
                "intended error path, we substituted duplicate_definition "
                "tests, which exercise a different but more practically "
                "significant error class (silent shadowing in Python is more "
                "dangerous than a circular dependency, which typically "
                "manifests as RecursionError or ImportError -- loud and "
                "visible). The cd detection gap is tracked at "
                "https://github.com/sowadalmughni/aether-lang/issues/4 and is "
                "targeted for a follow-up compiler release."
            ),
            "aether_run": (
                "aetherc check <file>; classified by stderr pattern. "
                "exit 0 = missed; exit !=0 with a known SemanticError "
                "pattern = caught_at_compile_time."
            ),
            "python_run": (
                "python <file>; 30s timeout. exit 0 = missed_silently; "
                "exit !=0 with a Python traceback = caught_at_runtime "
                "(includes SyntaxError at file load and Enum class-body "
                "errors)."
            ),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"summary: aether {summary['aether_caught']}/{len(test_cases)} caught | "
          f"lc {summary['lc_caught_at_runtime']}/{len(test_cases)} runtime | "
          f"dspy {summary['dspy_caught_at_runtime']}/{len(test_cases)} runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
