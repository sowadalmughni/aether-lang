#!/usr/bin/env bash
# Run all three ablations sequentially (after the cache one has completed)
# and emit the markdown summary. Idempotent: re-running it overwrites the
# JSON outputs.
#
# Each runner spawns its own runtime (via `cargo run --release`) and shuts it
# down at the end. The first run of the session does the cargo-incremental
# rebuild; subsequent runs reuse the cached binary so spawn time is ~5s.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/home/deamers_academy/aether-bench-venv/bin/python}"
TRIALS="${TRIALS:-5}"

echo "=== parallelization ablation (--trials $TRIALS) ==="
"$PY" bench/runners/ablation_parallel.py --trials "$TRIALS"

echo
echo "=== type-safety ablation (--trials $TRIALS — N/A; 30 fixed cases) ==="
"$PY" bench/runners/ablation_typesafety.py

echo
echo "=== emit ablations_v1.md ==="
"$PY" bench/runners/emit_ablations_md.py

echo
echo "=== outputs ==="
ls -la bench/results/ablation_cache_v1.json bench/results/ablation_parallel_v1.json bench/results/ablation_typesafety_v1.json bench/results/ablations_v1.md
