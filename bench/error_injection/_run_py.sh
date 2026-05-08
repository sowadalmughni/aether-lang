#!/usr/bin/env bash
# Run every Python file in bench/error_injection/lc/ and bench/error_injection/dspy/
# and report exit code + stdout/stderr excerpts. Used during corpus authoring;
# the real type-safety ablation runner does the same in Python.
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-/home/deamers_academy/aether-bench-venv/bin/python}"
for sub in lc dspy; do
    for f in bench/error_injection/$sub/*.py; do
        [ -f "$f" ] || continue
        out=$("$PY" "$f" 2>&1)
        rc=$?
        echo "FILE $f"
        echo "EXIT $rc"
        echo "OUT  $(echo "$out" | head -3)"
        echo "----"
    done
done
