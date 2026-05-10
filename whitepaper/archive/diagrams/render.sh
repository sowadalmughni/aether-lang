#!/usr/bin/env bash
# whitepaper/diagrams/render.sh
#
# Render every .mmd source under whitepaper/diagrams/ to a vector PDF
# under whitepaper/latex/figures/ using mermaid-cli (mmdc).
#
# Requires:
#   npm install -g @mermaid-js/mermaid-cli@10.9.x
# (the Dockerfile installs this; for local use, see whitepaper/diagrams/README
#  or run the same command in a shell where Node 20+ is available).
#
# Flags:
#   -i <input.mmd>          input file
#   -o <output.pdf>          output (extension drives format)
#   -f                       overwrite existing output
#   -b transparent           transparent background
#   -t default               default theme (no decorative colours)
#
# The script exits non-zero on the first failure so it can be wired into
# CI. Run from repo root:
#   bash whitepaper/diagrams/render.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$SCRIPT_DIR"
OUT_DIR="$REPO_ROOT/whitepaper/latex/figures"

mkdir -p "$OUT_DIR"

shopt -s nullglob
SOURCES=("$SRC_DIR"/*.mmd)
if (( ${#SOURCES[@]} == 0 )); then
    echo "render.sh: no .mmd sources under $SRC_DIR" >&2
    exit 1
fi

for src in "${SOURCES[@]}"; do
    stem="$(basename "$src" .mmd)"
    out="$OUT_DIR/${stem}.pdf"
    echo "render.sh: $src -> $out"
    mmdc -i "$src" -o "$out" -f -b transparent -t default
done

echo "render.sh: rendered ${#SOURCES[@]} diagram(s) to $OUT_DIR"
