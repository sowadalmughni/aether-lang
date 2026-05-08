#!/usr/bin/env python3
"""Emit bench/results/ablations_v1.md from the three ablation JSONs.

Reads (defaults relative to repo root):
  bench/results/ablation_cache_v1.json
  bench/results/ablation_parallel_v1.json
  bench/results/ablation_typesafety_v1.json

Writes:
  bench/results/ablations_v1.md

Format mirrors the existing bench/results/aether_mock_v1.md (metadata header,
tables, methodology footer). Per the type-safety acceptance criterion the
per-bug rows are required -- no aggregate-only output.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "bench" / "results"


def _fmt_metric(m: dict, decimals: int = 2) -> str:
    if m is None:
        return "n/a"
    mean = m.get("mean", 0.0)
    std = m.get("std", 0.0)
    lo, hi = m.get("ci95", [0.0, 0.0])
    return f"{mean:.{decimals}f} ± {std:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _fmt_delta(d: dict, decimals: int = 4) -> str:
    if d is None:
        return "n/a"
    mean = d.get("mean", 0.0)
    lo, hi = d.get("ci95", [0.0, 0.0])
    return f"{mean:+.{decimals}f} [{lo:+.{decimals}f}, {hi:+.{decimals}f}]"


def _header(meta: dict, title: str) -> list[str]:
    return [
        f"# {title}",
        "",
        f"- **System:** {meta.get('system')}",
        f"- **Provider:** {meta.get('provider')}",
        f"- **Version (git SHA):** `{meta.get('version')}`",
        f"- **CPU:** {meta.get('hardware', {}).get('cpu')}",
        f"- **RAM (GiB):** {meta.get('hardware', {}).get('ram_gb')}",
        f"- **OS:** {meta.get('hardware', {}).get('os')}",
        f"- **Trials per config:** {meta.get('trials_per_config', 'n/a')}",
        f"- **Measured at (UTC):** {meta.get('measured_at')}",
    ]


def _emit_caching(d: dict) -> list[str]:
    out: list[str] = ["", "## 1. Caching ablation", ""]
    out.append(
        "Three modes per dataset, 5 trials each. `no_cache` clears the cache before "
        "every individual `/execute` request (the runtime has no per-request cache-"
        "disable knob; this is the workaround). `l1_exact_match` is the runtime's "
        "default L1 behavior, clearing once per trial. `repeat_warm` runs the dataset "
        "once as warmup (latencies discarded; warmup tokens recorded separately) "
        "and then a second time as the measured pass."
    )
    out.append("")
    by_dataset: dict[str, list[dict]] = {}
    cross_by_dataset: dict[str, dict] = {}
    for r in d.get("results", []):
        if "cross_mode_deltas" in r:
            cmd = r["cross_mode_deltas"]
            cross_by_dataset[cmd["dataset"]] = cmd
        elif "dataset" in r and "config" in r:
            by_dataset.setdefault(r["dataset"], []).append(r)
    for ds_name, rows in by_dataset.items():
        out.append(f"### Dataset: `{ds_name}`")
        out.append("")
        out.append(
            "| Config | Trials | p50 (ms) | Cache hit rate | Tokens saved (total) |"
            " Δ hit_rate vs no_cache | Δ p50 (ms) vs no_cache |"
        )
        out.append("| --- | ---: | --- | --- | --- | --- | --- |")
        cmd = cross_by_dataset.get(ds_name, {})
        for r in rows:
            cfg = r["config"]
            d_hit = "—"
            d_p50 = "—"
            if cfg == "l1_exact_match":
                d_hit = _fmt_delta(cmd.get("cache_hit_rate_delta_l1_vs_no_cache"))
            if cfg == "repeat_warm":
                d_hit = _fmt_delta(cmd.get("cache_hit_rate_delta_warm_vs_no_cache"))
                d_p50 = _fmt_delta(cmd.get("latency_p50_delta_warm_vs_no_cache"), decimals=2)
            out.append(
                f"| `{cfg}` | {r['trials']} | "
                f"{_fmt_metric(r.get('latency_p50_ms'))} | "
                f"{_fmt_metric(r.get('cache_hit_rate'), decimals=4)} | "
                f"{_fmt_metric(r.get('tokens_saved_total'), decimals=1)} | "
                f"{d_hit} | {d_p50} |"
            )
        out.append("")
    return out


def _emit_parallel(d: dict) -> list[str]:
    out: list[str] = ["## 2. Parallelization ablation", ""]
    out.append(
        "Two modes per dataset, 5 trials each. `sequential` posts to "
        "`/execute?sequential=true`; `parallel` posts to `/execute` (default). "
        "The cache is cleared before every trial in both modes so the parallelization "
        "signal is not confounded with caching effects. `parallelization_factor` is "
        "the runtime response field of the same name (`sum(node_execution_times_ms) "
        "/ total_execution_time_ms`) -- a single definition used identically across "
        "both datasets and modes."
    )
    out.append("")
    by_dataset: dict[str, list[dict]] = {}
    speedup_by_dataset: dict[str, dict] = {}
    for r in d.get("results", []):
        if "speedup" in r:
            speedup_by_dataset[r["speedup"]["dataset"]] = r["speedup"]
        elif "dataset" in r and "config" in r:
            by_dataset.setdefault(r["dataset"], []).append(r)
    for ds_name, rows in by_dataset.items():
        out.append(f"### Dataset: `{ds_name}`")
        out.append("")
        out.append(
            "| Config | Trials | p50 (ms) | p95 (ms) | p99 (ms) | parallelization_factor |"
        )
        out.append("| --- | ---: | --- | --- | --- | --- |")
        for r in rows:
            out.append(
                f"| `{r['config']}` | {r['trials']} | "
                f"{_fmt_metric(r.get('latency_p50_ms'))} | "
                f"{_fmt_metric(r.get('latency_p95_ms'))} | "
                f"{_fmt_metric(r.get('latency_p99_ms'))} | "
                f"{_fmt_metric(r.get('parallelization_factor'), decimals=3)} |"
            )
        out.append("")
    out.append("### Speedup (paired-trial bootstrap, ratio sequential.p50 / parallel.p50)")
    out.append("")
    out.append("| Dataset | speedup_p50 (mean) | 95% CI |")
    out.append("| --- | --- | --- |")
    for ds_name, sp in speedup_by_dataset.items():
        m = sp["speedup_p50"]
        out.append(
            f"| `{ds_name}` | {m['mean']:.3f}x | [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] |"
        )
    out.append("")
    return out


def _emit_typesafety(d: dict) -> list[str]:
    out: list[str] = ["## 3. Type-safety ablation", ""]
    out.append(
        "Per-bug breakdown over a 30-test corpus (10 type_mismatch + 10 "
        "undefined_reference + 5 missing_field + 5 duplicate_definition). For each "
        "bug, an `aetherc check` is run on the .aether file, and a `python` is run "
        "on the LangChain and DSPy equivalents. Aether result `caught_at_compile_"
        "time` means stderr matched a known SemanticError variant; Python result "
        "`caught_at_runtime` means the subprocess exited non-zero with a Python "
        "traceback (includes SyntaxError at file load); `missed_silently` means "
        "the file ran to exit 0."
    )
    out.append("")
    out.append(
        "| ID | Bucket | Expected (aetherc) | Aether | LangChain | DSPy |"
    )
    out.append("| --- | --- | --- | --- | --- | --- |")
    for tc in d.get("test_cases", []):
        a = tc["aether"]
        lc = tc["langchain"]
        dp = tc["dspy"]
        a_cell = f"{a['result']} ({a.get('error_class_matched') or '?'})"
        lc_cell = f"{lc['result']}" + (f" ({lc.get('exception_class')})" if lc.get('exception_class') else "")
        dp_cell = f"{dp['result']}" + (f" ({dp.get('exception_class')})" if dp.get('exception_class') else "")
        expected = "/".join(tc.get("expected_aether_error_classes", []))
        out.append(
            f"| `{tc['id']}` | {tc['bucket']} | {expected} | {a_cell} | {lc_cell} | {dp_cell} |"
        )
    out.append("")
    s = d.get("summary", {})
    n = len(d.get("test_cases", []))
    out.append("### Aggregate")
    out.append("")
    out.append("| Detector | Caught (compile-time + runtime) | Missed silently |")
    out.append("| --- | ---: | ---: |")
    out.append(f"| Aether (compile-time) | {s.get('aether_caught', 0)}/{n} | {s.get('aether_missed', 0)}/{n} |")
    out.append(f"| LangChain (runtime) | {s.get('lc_caught_at_runtime', 0)}/{n} | {s.get('lc_missed_silently', 0)}/{n} |")
    out.append(f"| DSPy (runtime) | {s.get('dspy_caught_at_runtime', 0)}/{n} | {s.get('dspy_missed_silently', 0)}/{n} |")
    out.append("")
    by_bucket = s.get("by_bucket", {})
    if by_bucket:
        out.append("### By bucket")
        out.append("")
        out.append("| Bucket | Total | Aether caught | LC runtime | DSPy runtime |")
        out.append("| --- | ---: | ---: | ---: | ---: |")
        for bucket, vals in by_bucket.items():
            out.append(
                f"| {bucket} | {vals['total']} | {vals['aether_caught']} | "
                f"{vals['lc_caught_at_runtime']} | {vals['dspy_caught_at_runtime']} |"
            )
        out.append("")
    out.append("### Methodology note: cd → dd substitution")
    out.append("")
    note = d.get("methodology_notes", {}).get("cd_substitution") or ""
    out.append("> " + note.replace("\n", "\n> "))
    out.append("")
    return out


def main() -> int:
    cache = json.loads((RESULTS / "ablation_cache_v1.json").read_text(encoding="utf-8"))
    parallel = json.loads((RESULTS / "ablation_parallel_v1.json").read_text(encoding="utf-8"))
    types = json.loads((RESULTS / "ablation_typesafety_v1.json").read_text(encoding="utf-8"))

    lines: list[str] = []
    lines += _header(cache, "Aether ablation results — caching, parallelization, type-safety (v1)")
    lines += _emit_caching(cache)
    lines += _emit_parallel(parallel)
    lines += _emit_typesafety(types)

    lines.append("## Methodology footer")
    lines.append("")
    lines.append(
        "Cache and parallelization measurements use the existing `_bootstrap_ci` "
        "helper from `scripts/run_benchmark.py:546-593` (`scipy.stats.bootstrap`, "
        "BCa, 10 000 resamples, seed=42; percentile fallback on degenerate "
        "variance). Cell entries are `mean ± std [95% CI]`. Cross-mode deltas "
        "and the parallelization speedup ratio use paired-trial bootstrapping "
        "with the same parameters. Type-safety classification is by stderr-pattern "
        "match against the SemanticError variant names in "
        "`aether-compiler/src/semantic.rs:36-177`. The mock provider is "
        "deterministic (50 ms flat per LLM call); trial-to-trial variance "
        "reflects scheduling and HTTP-loopback jitter only."
    )
    lines.append("")

    out_path = RESULTS / "ablations_v1.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
