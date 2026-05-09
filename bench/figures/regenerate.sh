#!/usr/bin/env bash
# bench/figures/regenerate.sh
#
# Re-runs bench/figures/generate_figures.py inside the bench Docker
# container so output is byte-reproducible against the matplotlib
# version pinned in bench/requirements.txt. The container mounts the
# repo at /aether (see docker-compose.yml -> bench service), so the
# script writes its PDFs and JSON sidecars directly into the host
# whitepaper/latex/figures/ directory.
#
# Usage (from repo root or anywhere — script cd's to the repo root):
#   bash bench/figures/regenerate.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
exec docker compose run --rm --no-deps bench python bench/figures/generate_figures.py
