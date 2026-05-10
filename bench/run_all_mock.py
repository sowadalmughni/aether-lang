#!/usr/bin/env python3
"""
bench/run_all_mock.py

Master mock-mode reproducer. Invokes every benchmark runner that can run
without an OPENAI_API_KEY, writing outputs to `bench/results/repro/` so the
committed reference set in `bench/results/` is never touched (hard rule:
the repo's authoritative numbers are the ones currently committed).

Designed to be run from inside the docker-compose `bench` service, where:
  - The `runtime` service is already healthy at AETHER_RUNTIME_URL
  - bench/results/ is bind-mounted from the host so outputs persist

Real-API JSONs (security_v1, *_real_api_v1, real_api_v1) are explicitly
excluded; bench/verify_reproduction.py marks them n/a in the diff report.

Fail-fast: any nonzero exit from any runner aborts the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
REPRO_DIR = REPO_ROOT / "bench" / "results" / "repro"
RUNTIME_URL = os.environ.get("AETHER_RUNTIME_URL", "http://127.0.0.1:3000")


def banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def wait_for_runtime(url: str, timeout_s: int = 60) -> None:
    """Defensive wait for /health, even though docker-compose depends_on already gates this."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/health", timeout=3) as resp:
                if resp.status == 200:
                    print(f"runtime healthy at {url}", flush=True)
                    return
        except (URLError, TimeoutError, OSError) as e:
            last_err = e
        time.sleep(1)
    raise SystemExit(f"runtime did not become healthy at {url} within {timeout_s}s ({last_err})")


def run(label: str, cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    """Run a subprocess; abort the whole suite on nonzero exit."""
    banner(f"[{label}] {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
    dt = time.monotonic() - t0
    print(f"[{label}] exit={rc} elapsed={dt:.1f}s", flush=True)
    if rc != 0:
        raise SystemExit(f"[{label}] FAILED with rc={rc}; aborting suite")


def main() -> None:
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {REPRO_DIR}")
    print(f"runtime URL: {RUNTIME_URL}")

    wait_for_runtime(RUNTIME_URL)

    py = sys.executable

    # 1. Aether mock suite (the master /scripts/run_benchmark.py --suite runner).
    run(
        "aether_mock_v1",
        [
            py, "scripts/run_benchmark.py", "--suite",
            "--trials", "5",
            "--runtime-url", RUNTIME_URL,
            "--no-autostart",
            "--output-json", str(REPRO_DIR / "aether_mock_v1.json"),
            "--output-md",   str(REPRO_DIR / "aether_mock_v1.md"),
        ],
    )

    # 2. Caching ablation.
    run(
        "ablation_cache_v1",
        [
            py, "bench/runners/ablation_cache.py",
            "--trials", "5",
            "--runtime-url", RUNTIME_URL,
            "--no-autostart",
            "--output", str(REPRO_DIR / "ablation_cache_v1.json"),
        ],
    )

    # 3. Parallelization ablation.
    run(
        "ablation_parallel_v1",
        [
            py, "bench/runners/ablation_parallel.py",
            "--trials", "5",
            "--runtime-url", RUNTIME_URL,
            "--no-autostart",
            "--output", str(REPRO_DIR / "ablation_parallel_v1.json"),
        ],
    )

    # 4. Type-safety ablation (compile-time + mock LLM; no live runtime needed).
    #    Pass --aetherc and --python explicitly: the runner's defaults
    #    (target/debug/aetherc and /home/deamers_academy/.../python) are baked
    #    against the maintainer's host machine and do not exist in the docker
    #    image. The Dockerfile builds release-only, and the bench venv lives
    #    at /opt/venv (== sys.executable inside this container).
    run(
        "ablation_typesafety_v1",
        [
            py, "bench/runners/ablation_typesafety.py",
            "--aetherc", str(REPO_ROOT / "target" / "release" / "aetherc"),
            "--python",  py,
            "--output", str(REPRO_DIR / "ablation_typesafety_v1.json"),
        ],
    )

    # 5. LangChain baseline (mock).
    run(
        "langchain_v1",
        [
            py, "bench/baselines/langchain_baseline.py",
            "--mode", "mock",
            "--trials", "5",
            "--output", str(REPRO_DIR / "langchain_v1.json"),
        ],
        extra_env={"BASELINE_PROVIDER": "mock"},
    )

    # 6. DSPy baseline (mock).
    run(
        "dspy_v1",
        [
            py, "bench/baselines/dspy_baseline.py",
            "--mode", "mock",
            "--trials", "5",
            "--output", str(REPRO_DIR / "dspy_v1.json"),
        ],
        extra_env={"BASELINE_PROVIDER": "mock"},
    )

    # 7. HotpotQA: Aether RAG (mock).
    run(
        "hotpotqa_aether_v1",
        [
            py, "bench/baselines/aether_hotpot.py",
            "--mode", "mock",
            "--trials", "3",
            "--runtime-url", RUNTIME_URL,
            "--no-autostart",
            "--output", str(REPRO_DIR / "hotpotqa_aether_v1.json"),
        ],
    )

    # 8. HotpotQA: LangChain RAG (mock).
    run(
        "hotpotqa_langchain_v1",
        [
            py, "bench/baselines/langchain_rag.py",
            "--mode", "mock",
            "--trials", "3",
            "--output", str(REPRO_DIR / "hotpotqa_langchain_v1.json"),
        ],
        extra_env={"BASELINE_PROVIDER": "mock"},
    )

    # 9. HotpotQA: DSPy RAG (mock).
    run(
        "hotpotqa_dspy_v1",
        [
            py, "bench/baselines/dspy_rag.py",
            "--mode", "mock",
            "--trials", "3",
            "--output", str(REPRO_DIR / "hotpotqa_dspy_v1.json"),
        ],
        extra_env={"BASELINE_PROVIDER": "mock"},
    )

    banner("All mock-mode benchmarks completed.")
    print(f"Outputs in: {REPRO_DIR}")
    print("Run bench/verify_reproduction.py to diff against bench/results/")


if __name__ == "__main__":
    main()
