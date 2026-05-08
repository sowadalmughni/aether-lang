"""Security benchmark orchestrator.

Runs three configurations (aether_taint_on, aether_taint_off,
langchain_baseline) over N InjecAgent-derived attack cases and N
benign-control variants for T trials each. Aggregates ASR, benign
task success rate, and compile-time catch rate per config with
bootstrapped 95% CIs. Writes `bench/results/security_v1.json`.

Usage
-----
    BASELINE_PROVIDER=openai python -m bench.security.run_security_bench \\
        --trials 3 --max-cases 40 --confirm-cost

The `--confirm-cost` flag is required for live runs (mirrors the
LangChain baseline tradition). Without it the orchestrator runs in
dry-run mode: configs that would call the LLM are skipped and only
the static aether_taint_on results are produced.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
OUT_PATH = RESULTS_DIR / "security_v1.json"

# When invoked as `python bench/security/run_security_bench.py` (not
# `python -m bench.security.run_security_bench`), the package import
# can fail. Detect and adjust sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from bench.security.dataset import AttackCase, load_cases  # noqa: E402
from bench.security.openai_caller import CostMeter, require_api_key  # noqa: E402
from bench.security.runners import (  # noqa: E402
    CaseResult,
    run_aether_taint_off,
    run_aether_taint_on,
    run_langchain_baseline,
)


@dataclass
class TrialAggregate:
    config: str
    mode: str
    n: int
    metric_name: str
    mean: float
    std: float
    ci95: List[float] = field(default_factory=list)


@dataclass
class RunMetadata:
    git_sha: str
    benchmark: str
    suite: str
    provider: str
    model: str
    trials_per_config: int
    cases_per_trial: int
    measured_at: str
    cost_cap_usd: float


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=10
        ).strip()
        return out
    except Exception:
        return "unknown"


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 42) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return [lo, hi]


def aggregate(per_trial_rates: list[float]) -> tuple[float, float, list[float]]:
    if not per_trial_rates:
        return 0.0, 0.0, [0.0, 0.0]
    mean = statistics.fmean(per_trial_rates)
    std = statistics.pstdev(per_trial_rates) if len(per_trial_rates) > 1 else 0.0
    ci = bootstrap_ci(per_trial_rates)
    return mean, std, ci


def run_one_config(
    *,
    name: str,
    runner: Callable[..., CaseResult],
    cases: list[AttackCase],
    cost_meter: CostMeter,
    trials: int,
    log_prefix: str = "",
) -> tuple[List[CaseResult], List[CaseResult]]:
    """Run `runner` once per (case, mode) per trial. Returns
    (attack_results, benign_results) flat lists across trials.
    """
    attack_results: List[CaseResult] = []
    benign_results: List[CaseResult] = []
    for trial in range(trials):
        for i, case in enumerate(cases):
            ar = runner(case, cost_meter=cost_meter, mode="attack")
            attack_results.append(ar)
            br = runner(case, cost_meter=cost_meter, mode="benign")
            benign_results.append(br)
            if (i + 1) % 10 == 0:
                print(
                    f"  {log_prefix}{name} trial {trial + 1}/{trials} "
                    f"case {i + 1}/{len(cases)} (spent ${cost_meter.spent_usd:.4f})",
                    flush=True,
                )
    return attack_results, benign_results


def per_trial_rates(
    results: List[CaseResult],
    *,
    n_cases: int,
    trials: int,
    field_name: str,
    only_when_ran: bool = False,
) -> list[float]:
    """Group results back into trials and compute the mean of
    `field_name` per trial. None values are coerced to 0 (e.g.
    aether_taint_on has compiled=False -> attack_succeeded=False).
    Returns [trial_0_rate, trial_1_rate, ...].
    """
    rates: list[float] = []
    for t in range(trials):
        trial_slice = results[t * n_cases : (t + 1) * n_cases]
        if only_when_ran:
            trial_slice = [r for r in trial_slice if r.ran_llm]
        if not trial_slice:
            rates.append(0.0)
            continue
        flags = []
        for r in trial_slice:
            v = getattr(r, field_name)
            flags.append(1.0 if v else 0.0)
        rates.append(sum(flags) / len(flags))
    return rates


def build_config_block(
    config_name: str,
    attack_results: List[CaseResult],
    benign_results: List[CaseResult],
    *,
    n_cases: int,
    trials: int,
) -> dict:
    asr_per_trial = per_trial_rates(
        attack_results, n_cases=n_cases, trials=trials, field_name="attack_succeeded"
    )
    benign_per_trial = per_trial_rates(
        benign_results, n_cases=n_cases, trials=trials, field_name="user_task_completed"
    )
    catch_per_trial = per_trial_rates(
        attack_results, n_cases=n_cases, trials=trials, field_name="compiled"
    )
    # `compiled` is True when the program compiled, so catch rate = 1 - compile rate.
    catch_per_trial = [1.0 - x for x in catch_per_trial]

    asr_mean, asr_std, asr_ci = aggregate(asr_per_trial)
    benign_mean, benign_std, benign_ci = aggregate(benign_per_trial)
    catch_mean, catch_std, catch_ci = aggregate(catch_per_trial)

    return {
        "config": config_name,
        "metrics": [
            {
                "metric": "attack_success_rate",
                "mean": asr_mean,
                "std": asr_std,
                "ci95": asr_ci,
                "per_trial": asr_per_trial,
            },
            {
                "metric": "benign_task_success_rate",
                "mean": benign_mean,
                "std": benign_std,
                "ci95": benign_ci,
                "per_trial": benign_per_trial,
            },
            {
                "metric": "compile_time_catch_rate",
                "mean": catch_mean,
                "std": catch_std,
                "ci95": catch_ci,
                "per_trial": catch_per_trial,
            },
        ],
        "raw_results": {
            "attack": [asdict(r) for r in attack_results],
            "benign": [asdict(r) for r in benign_results],
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3)
    p.add_argument(
        "--max-cases", type=int, default=40, help="Cap N cases per trial."
    )
    p.add_argument(
        "--cost-cap-usd",
        type=float,
        default=5.0,
        help="Hard ceiling on total OpenAI spend for this run.",
    )
    p.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Required for any live LLM call. Without it, only the "
        "static aether_taint_on config runs.",
    )
    p.add_argument(
        "--skip-langchain",
        action="store_true",
        help="Skip the LangChain baseline (still runs both Aether configs).",
    )
    args = p.parse_args(argv)

    cases = load_cases()
    if args.max_cases:
        cases = cases[: args.max_cases]
    n_cases = len(cases)
    trials = args.trials

    cost_meter = CostMeter(cap_usd=args.cost_cap_usd)
    print(f"== Security benchmark — {n_cases} cases × {trials} trials ==")
    print(f"   git={get_git_sha()[:12]}  cap=${args.cost_cap_usd:.2f}", flush=True)

    if args.confirm_cost:
        require_api_key()

    blocks: list[dict] = []

    # Config 1: aether_taint_on (static; no LLM)
    print("\n[1/3] aether_taint_on (compile-time check only) ...", flush=True)
    on_attack, on_benign = run_one_config(
        name="aether_taint_on",
        runner=run_aether_taint_on,
        cases=cases,
        cost_meter=cost_meter,
        trials=trials,
    )
    blocks.append(
        build_config_block(
            "aether_taint_on", on_attack, on_benign, n_cases=n_cases, trials=trials
        )
    )

    if not args.confirm_cost:
        print(
            "\n--confirm-cost not set; skipping live-LLM configs (aether_taint_off, "
            "langchain_baseline). Re-run with --confirm-cost for full results."
        )
    else:
        # Config 2: aether_taint_off (live LLM)
        print("\n[2/3] aether_taint_off (Pass 6 disabled, live LLM) ...", flush=True)
        off_attack, off_benign = run_one_config(
            name="aether_taint_off",
            runner=run_aether_taint_off,
            cases=cases,
            cost_meter=cost_meter,
            trials=trials,
        )
        blocks.append(
            build_config_block(
                "aether_taint_off",
                off_attack,
                off_benign,
                n_cases=n_cases,
                trials=trials,
            )
        )
        # Config 3: langchain_baseline (live LLM)
        if not args.skip_langchain:
            print(
                "\n[3/3] langchain_baseline (LCEL, no taint tracking) ...", flush=True
            )
            lc_attack, lc_benign = run_one_config(
                name="langchain_baseline",
                runner=run_langchain_baseline,
                cases=cases,
                cost_meter=cost_meter,
                trials=trials,
            )
            blocks.append(
                build_config_block(
                    "langchain_baseline",
                    lc_attack,
                    lc_benign,
                    n_cases=n_cases,
                    trials=trials,
                )
            )

    metadata = {
        "git_sha": get_git_sha(),
        "benchmark": "InjecAgent-adapted (direct prompt injection)",
        "suite": "dh+ds base, 20+20 deduped",
        "provider": os.environ.get("BASELINE_PROVIDER", "openai") if args.confirm_cost else "static-only",
        "model": "gpt-4o-mini",
        "trials_per_config": trials,
        "cases_per_trial": n_cases,
        "measured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "cost_cap_usd": args.cost_cap_usd,
        "cost_spent_usd": cost_meter.spent_usd,
        "cost_by_model": cost_meter.by_model,
        "completed_live_configs": [b["config"] for b in blocks if b["config"] != "aether_taint_on"],
    }
    payload = {"metadata": metadata, "configs": blocks}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print(f"Cost spent: ${cost_meter.spent_usd:.4f} of ${args.cost_cap_usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
