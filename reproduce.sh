#!/usr/bin/env bash
# reproduce.sh
#
# End-to-end mock-mode reproducer for the Aether whitepaper benchmarks.
# A third party with Docker installed runs this script from a fresh clone
# and gets:
#
#   1. A built reproducible Docker image (aether:repro)
#   2. A mock-mode runtime service running on http://runtime:3000
#   3. Every mock-runnable benchmark re-executed end-to-end, with outputs
#      written to bench/results/repro/ (committed reference JSONs are
#      not touched)
#   4. A diff report against the committed reference set, with structural
#      mismatches as hard fails and latency drift reported informatively
#
# Real-API JSONs (security_v1, *_real_api_v1, real_api_v1) cannot be
# reproduced without OPENAI_API_KEY and incurring real cost; they are
# explicitly skipped and marked "n/a" in the diff report. See
# REPRODUCIBILITY.md for how to reproduce them deliberately.
#
# Expected total runtime on the reference hardware (Intel i5-8250U,
# 8 GB RAM, Ubuntu 24.04):
#   - Docker image build (cold):   ~15 min
#   - Mock benchmark suite:        ~95-110 min
#   - Diff:                        <30 s
#   - Total cold-cache:            ~110-125 min
#
# Run from Git Bash, WSL, or any POSIX shell with Docker accessible.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="aether:repro"
COMPOSE_PROJECT="aether-repro"

log() { printf '\n==> %s\n' "$*"; }
err() { printf 'ERROR: %s\n' "$*" >&2; }

cleanup() {
    log "tearing down docker compose services"
    docker compose -p "$COMPOSE_PROJECT" down --remove-orphans || true
}
trap cleanup EXIT INT TERM

if ! command -v docker >/dev/null 2>&1; then
    err "docker is not on PATH; install Docker Desktop or Docker Engine first"
    exit 1
fi

log "Step 1/4: building Docker image ($IMAGE_TAG)"
docker build -t "$IMAGE_TAG" .

log "Step 2/4: starting mock-mode runtime service"
# Tag the compose-built image so both runtime and bench services use our
# freshly built image (compose's image: aether:latest is just a label).
docker tag "$IMAGE_TAG" aether:latest
docker compose -p "$COMPOSE_PROJECT" up -d runtime
log "waiting for runtime to become healthy"
# docker compose's healthcheck loop -- bound to the Dockerfile's HEALTHCHECK
for _ in $(seq 1 60); do
    status=$(docker inspect -f '{{.State.Health.Status}}' aether-runtime 2>/dev/null || echo "missing")
    if [ "$status" = "healthy" ]; then
        log "runtime healthy"
        break
    fi
    sleep 2
done
if [ "$status" != "healthy" ]; then
    err "runtime did not become healthy within 120s (last status: $status)"
    docker logs aether-runtime --tail 50 || true
    exit 1
fi

log "Step 3/4: running full mock-mode benchmark suite (~95-110 min)"
# Override the compose 'bench' service default command. Run as a one-shot.
docker compose -p "$COMPOSE_PROJECT" run --rm bench \
    python bench/run_all_mock.py

log "Step 4/4: diffing produced JSONs against bench/results/"
docker compose -p "$COMPOSE_PROJECT" run --rm bench \
    python bench/verify_reproduction.py \
        --reference bench/results \
        --produced  bench/results/repro \
        --tolerance bench/verify_tolerance.toml \
        --report    bench/results/repro/DIFF_REPORT.md

log "Done. See bench/results/repro/DIFF_REPORT.md for the field-by-field diff."
