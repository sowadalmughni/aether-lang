#!/usr/bin/env bash
# Real-API benchmark runner for Aether / LangChain / DSPy against gpt-4o-mini.
#
# Hard requirements:
#   - OPENAI_API_KEY must be set (the script aborts otherwise).
#   - Repo Python deps from bench/requirements.txt installed (langchain, dspy).
#   - aether-runtime built with --features llm-api (the script handles this).
#
# Outputs (all under bench/results/):
#   aether_real_api_v1.json   -- aether suite JSON (same schema as aether_mock_v1.json)
#   aether_real_api_v1.md     -- aether markdown report
#   langchain_real_api_v1.json
#   dspy_real_api_v1.json
#   real_api_v1.json          -- combined JSON (the artifact reviewers will read)
#
# Each per-trial dict carries `tokens_input` and `tokens_output` recorded from
# OpenAI response usage (see bench/baselines/langchain_baseline.py:UsageTracker
# and bench/baselines/dspy_baseline.py:_sum_lm_usage). The merge step computes
# total USD cost from these counts using the public gpt-4o-mini rates.

set -euo pipefail

# ---------- pricing constants (gpt-4o-mini, public list price) ----------------
PRICE_INPUT_PER_M=0.15
PRICE_OUTPUT_PER_M=0.60
MODEL_NAME="gpt-4o-mini"

# ---------- defaults ----------------------------------------------------------
BUDGET_USD=10
TRIALS=3
RESULTS_DIR="bench/results"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_URL="${AETHER_RUNTIME_URL:-http://127.0.0.1:3000}"
SMOKE=0

usage() {
  cat <<EOF
Usage: scripts/run_real_api_benchmark.sh [options]

Options:
  --budget USD        Abort if upfront cost estimate exceeds this. Default: 10
  --trials N          Trials per (dataset, config). Default: 3
  --runtime-url URL   Aether runtime URL. Default: ${RUNTIME_URL}
  --python BIN        Python interpreter. Default: ${PYTHON_BIN}
  --smoke             Internal flag for tiny smoke run (1 trial, no full suite).
  -h | --help         Show this help.

Required env: OPENAI_API_KEY

The script:
  1) verifies OPENAI_API_KEY is set
  2) computes upfront cost estimate using the formula:
       (3 triage_calls + 5 extraction_calls) * 100 queries * 3 systems * \$TRIALS trials
       at avg 500 input + 200 output tokens per call, and aborts if the
       estimate exceeds --budget
  3) builds aether-runtime --features llm-api (incremental)
  4) starts the runtime, waits for /health, runs the aether suite with
     AETHER_PROVIDER=openai
  5) runs LangChain + DSPy baselines with BASELINE_PROVIDER=openai
  6) calls scripts/merge_real_api_results.py to compute actual cost from
     recorded token usage and write bench/results/real_api_v1.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget)      BUDGET_USD="$2"; shift 2 ;;
    --trials)      TRIALS="$2"; shift 2 ;;
    --runtime-url) RUNTIME_URL="$2"; shift 2 ;;
    --python)      PYTHON_BIN="$2"; shift 2 ;;
    --smoke)       SMOKE=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# ---------- 1. key check ------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo "       Export it before running, e.g. 'export OPENAI_API_KEY=sk-...'" >&2
  exit 1
fi
echo "[1/6] OPENAI_API_KEY set (length=${#OPENAI_API_KEY})."

# ---------- 2. upfront cost estimate -----------------------------------------
EST_JSON="$("${PYTHON_BIN}" - "${TRIALS}" "${BUDGET_USD}" "${PRICE_INPUT_PER_M}" "${PRICE_OUTPUT_PER_M}" <<'PY'
import json, sys
trials = int(sys.argv[1])
budget = float(sys.argv[2])
in_per_m = float(sys.argv[3])
out_per_m = float(sys.argv[4])
# formula per task spec: (3 triage + 5 extract) * 100 queries * 3 systems * trials
calls = (3 + 5) * 100 * 3 * trials
in_tok = calls * 500
out_tok = calls * 200
cost = in_tok * in_per_m / 1e6 + out_tok * out_per_m / 1e6
print(json.dumps({"calls": calls, "in_tok": in_tok, "out_tok": out_tok,
                  "cost_usd": cost, "budget_usd": budget,
                  "within_budget": cost <= budget}))
PY
)"
EST_CALLS=$(echo "${EST_JSON}" | "${PYTHON_BIN}" -c "import sys,json;d=json.loads(sys.stdin.read());print(d['calls'])")
EST_COST=$(echo "${EST_JSON}"  | "${PYTHON_BIN}" -c "import sys,json;d=json.loads(sys.stdin.read());print(f\"{d['cost_usd']:.4f}\")")
EST_OK=$(echo "${EST_JSON}"   | "${PYTHON_BIN}" -c "import sys,json;d=json.loads(sys.stdin.read());print(int(d['within_budget']))")

echo "[2/6] Upfront cost estimate (per task formula):"
echo "        formula  : (3 triage + 5 extract) * 100 queries * 3 systems * ${TRIALS} trials"
echo "        calls    : ${EST_CALLS}"
echo "        avg toks : 500 input + 200 output per call"
echo "        rates    : \$${PRICE_INPUT_PER_M}/1M input, \$${PRICE_OUTPUT_PER_M}/1M output (${MODEL_NAME})"
echo "        estimate : \$${EST_COST}"
echo "        budget   : \$${BUDGET_USD}"
if [[ "${EST_OK}" != "1" ]]; then
  echo "ABORT: estimate \$${EST_COST} exceeds budget \$${BUDGET_USD}" >&2
  exit 3
fi
echo "        status   : OK (\$${EST_COST} <= \$${BUDGET_USD})"

mkdir -p "${RESULTS_DIR}"

# Track total wall time of the run.
T0=$(date +%s)

# ---------- 3. build aether-runtime ------------------------------------------
echo "[3/6] Building aether-runtime --features llm-api (incremental)..."
cargo build --release --features llm-api -p aether-runtime
RUNTIME_BIN="$(pwd)/target/release/aether-runtime"
if [[ ! -x "${RUNTIME_BIN}" ]]; then
  # cargo on Windows produces .exe; on Linux just the bare name.
  if [[ -x "${RUNTIME_BIN}.exe" ]]; then
    RUNTIME_BIN="${RUNTIME_BIN}.exe"
  else
    echo "ERROR: aether-runtime binary not found at ${RUNTIME_BIN}" >&2
    exit 4
  fi
fi
echo "        runtime binary: ${RUNTIME_BIN}"

# ---------- 4. start runtime --------------------------------------------------
RUNTIME_LOG="${RESULTS_DIR}/runtime_real_api.log"
echo "[4/6] Starting aether-runtime (AETHER_PROVIDER=openai)..."
AETHER_PROVIDER=openai \
RUST_LOG="${RUST_LOG:-info}" \
  "${RUNTIME_BIN}" > "${RUNTIME_LOG}" 2>&1 &
RUNTIME_PID=$!
trap 'echo "[cleanup] stopping runtime pid=${RUNTIME_PID}"; kill "${RUNTIME_PID}" 2>/dev/null || true; wait "${RUNTIME_PID}" 2>/dev/null || true' EXIT

# wait for /health
echo "        waiting for ${RUNTIME_URL}/health ..."
for _ in $(seq 1 60); do
  if curl -fsS "${RUNTIME_URL}/health" > /dev/null 2>&1; then
    echo "        runtime healthy."
    break
  fi
  sleep 1
done
if ! curl -fsS "${RUNTIME_URL}/health" > /dev/null 2>&1; then
  echo "ERROR: runtime did not become healthy. Last 40 log lines:" >&2
  tail -n 40 "${RUNTIME_LOG}" >&2
  exit 5
fi

AETHER_JSON="${RESULTS_DIR}/aether_real_api_v1.json"
AETHER_MD="${RESULTS_DIR}/aether_real_api_v1.md"
LANGCHAIN_JSON="${RESULTS_DIR}/langchain_real_api_v1.json"
DSPY_JSON="${RESULTS_DIR}/dspy_real_api_v1.json"
COMBINED_JSON="${RESULTS_DIR}/real_api_v1.json"

# ---------- 5. run all three systems ------------------------------------------
echo "[5/6] Running all three systems with --trials ${TRIALS}..."

echo "       [a] aether suite (AETHER_PROVIDER=openai)..."
AETHER_PROVIDER=openai \
  "${PYTHON_BIN}" scripts/run_benchmark.py \
  --suite \
  --trials "${TRIALS}" \
  --no-autostart \
  --runtime-url "${RUNTIME_URL}" \
  --output-json "${AETHER_JSON}" \
  --output-md "${AETHER_MD}"

echo "       [b] langchain baseline (BASELINE_PROVIDER=openai)..."
BASELINE_PROVIDER=openai \
  "${PYTHON_BIN}" bench/baselines/langchain_baseline.py \
  --trials "${TRIALS}" \
  --mode openai \
  --confirm-cost \
  --output "${LANGCHAIN_JSON}"

echo "       [c] dspy baseline (BASELINE_PROVIDER=openai)..."
BASELINE_PROVIDER=openai \
  "${PYTHON_BIN}" bench/baselines/dspy_baseline.py \
  --trials "${TRIALS}" \
  --mode openai \
  --confirm-cost \
  --output "${DSPY_JSON}"

# ---------- 6. merge & compute actual cost -----------------------------------
echo "[6/6] Merging into ${COMBINED_JSON} and computing actual cost..."
"${PYTHON_BIN}" scripts/merge_real_api_results.py \
  --aether    "${AETHER_JSON}" \
  --langchain "${LANGCHAIN_JSON}" \
  --dspy      "${DSPY_JSON}" \
  --budget    "${BUDGET_USD}" \
  --price-input-per-m  "${PRICE_INPUT_PER_M}" \
  --price-output-per-m "${PRICE_OUTPUT_PER_M}" \
  --model     "${MODEL_NAME}" \
  --estimated-cost-usd "${EST_COST}" \
  --output    "${COMBINED_JSON}"

T1=$(date +%s)
echo ""
echo "Done in $((T1 - T0)) s. Combined real-API benchmark at ${COMBINED_JSON}"
