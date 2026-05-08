#!/usr/bin/env bash
# Run aetherc check on every .aether file under bench/error_injection/ and
# report exit code + first error line. Used by the type-safety ablation
# during authoring + by the runner at measurement time.
set -u
cd "$(dirname "$0")/../.."
AETHERC="${AETHERC:-./target/debug/aetherc}"
for f in bench/error_injection/*.aether; do
    out=$("$AETHERC" check "$f" 2>&1)
    rc=$?
    echo "FILE $f"
    echo "EXIT $rc"
    echo "OUT  $out" | head -5
    echo "----"
done
