#!/usr/bin/env python3
"""
bench/figures/generate_figures.py

Produces the five publication figures referenced from whitepaper/latex/aether.tex
by reading the committed JSON results in bench/results/. No values are
fabricated, smoothed, rounded, or interpolated: every plotted number is a
direct json.load() lookup, and every figure has a sibling <name>.json sidecar
listing the exact source file + JSON path + value for each plotted point.

Run (from repo root):
    python bench/figures/generate_figures.py

Or, for byte-stable container output:
    bash bench/figures/regenerate.sh

Output: whitepaper/latex/figures/{cross_system_latency,parallel_speedup,
        cache_hit_rate,type_safety_corpus,security_outcome}.{pdf,json}

The figure column width is set for a single-column article (6.5in). The paper
template (whitepaper/latex/aether.tex via header.tex) is single-column; if it
were ever switched to twocolumn, drop COL_W to 3.3in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "bench" / "results"
OUT = REPO / "whitepaper" / "latex" / "figures"

COL_W = 6.5  # inches; single-column article

plt.rcParams.update({
    "font.family": "sans-serif",
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
})


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def find_result(results: list, *, dataset: str, config: str) -> tuple[int, dict]:
    """Locate a result element by (dataset, config). Raise on miss — never fall
    back to a positional guess. Returns (index, element)."""
    for i, r in enumerate(results):
        if r.get("dataset") == dataset and r.get("config") == config:
            return i, r
    raise KeyError(f"no result element for dataset={dataset!r} config={config!r}")


def find_speedup(results: list, *, dataset: str) -> tuple[int, dict]:
    for i, r in enumerate(results):
        sp = r.get("speedup")
        if isinstance(sp, dict) and sp.get("dataset") == dataset:
            return i, r
    raise KeyError(f"no speedup element for dataset={dataset!r}")


def asym_err(mean: float, ci95: list[float]) -> tuple[float, float]:
    """Return (lower_err, upper_err) for matplotlib's yerr=[[lows], [highs]]."""
    return max(mean - ci95[0], 0.0), max(ci95[1] - mean, 0.0)


def write_sidecar(stem: str, caption: str, sources: list[dict]) -> None:
    payload = {"figure": stem, "caption": caption, "sources": sources}
    (OUT / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save(fig, stem: str) -> None:
    path = OUT / f"{stem}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO).as_posix()}  (sidecar: {stem}.json)")


# ---------------------------------------------------------------------------
# FIG-1: cross_system_latency
# ---------------------------------------------------------------------------

def fig1_cross_system_latency() -> None:
    aether = load_json("aether_mock_v1.json")
    lc = load_json("langchain_v1.json")
    dspy = load_json("dspy_v1.json")

    datasets = ["customer_support_100", "document_analysis_50"]
    configs = ["sequential", "parallel", "parallel_cached"]
    systems = [
        ("Aether", "aether_mock_v1.json", aether),
        ("LangChain", "langchain_v1.json", lc),
        ("DSPy", "dspy_v1.json", dspy),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.8), sharey=False)
    sources: list[dict] = []
    x = np.arange(len(configs))
    width = 0.26

    for ax, dataset in zip(axes, datasets):
        for j, (sys_name, sys_file, sys_data) in enumerate(systems):
            means: list[float] = []
            errs_lo: list[float] = []
            errs_hi: list[float] = []
            for cfg in configs:
                idx, elem = find_result(sys_data["results"], dataset=dataset, config=cfg)
                m = elem["latency_p50_ms"]["mean"]
                ci = elem["latency_p50_ms"]["ci95"]
                lo, hi = asym_err(m, ci)
                means.append(m)
                errs_lo.append(lo)
                errs_hi.append(hi)
                sources.append({
                    "file": sys_file,
                    "json_path": f"results[{idx}].latency_p50_ms.mean",
                    "system": sys_name, "dataset": dataset, "config": cfg,
                    "field": "latency_p50_ms.mean", "value": m,
                })
                sources.append({
                    "file": sys_file,
                    "json_path": f"results[{idx}].latency_p50_ms.ci95",
                    "system": sys_name, "dataset": dataset, "config": cfg,
                    "field": "latency_p50_ms.ci95", "value": ci,
                })
            ax.bar(
                x + (j - 1) * width, means, width,
                yerr=[errs_lo, errs_hi], capsize=2.5,
                label=sys_name, edgecolor="black", linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=15, ha="right")
        ax.set_title(dataset)
        ax.set_ylabel("p50 latency (ms)")
        ax.set_ylim(bottom=0)

    axes[0].legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout()

    caption = (
        "Per-config p50 latency (ms) across systems on synthetic mock-LLM "
        "benchmarks. Error bars are 95% paired BCa intervals. n=5 trials per "
        "config. Source: aether_mock_v1.json, langchain_v1.json, dspy_v1.json."
    )
    write_sidecar("cross_system_latency", caption, sources)
    save(fig, "cross_system_latency")


# ---------------------------------------------------------------------------
# FIG-2: parallel_speedup
# ---------------------------------------------------------------------------

def fig2_parallel_speedup() -> None:
    par = load_json("ablation_parallel_v1.json")
    datasets = ["customer_support_100", "document_analysis_50"]
    configs = ["sequential", "parallel"]

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.8), sharey=False)
    sources: list[dict] = []
    x = np.arange(len(configs))
    width = 0.55

    for ax, dataset in zip(axes, datasets):
        means: list[float] = []
        errs_lo: list[float] = []
        errs_hi: list[float] = []
        for cfg in configs:
            idx, elem = find_result(par["results"], dataset=dataset, config=cfg)
            m = elem["latency_p50_ms"]["mean"]
            ci = elem["latency_p50_ms"]["ci95"]
            lo, hi = asym_err(m, ci)
            means.append(m)
            errs_lo.append(lo)
            errs_hi.append(hi)
            sources.append({
                "file": "ablation_parallel_v1.json",
                "json_path": f"results[{idx}].latency_p50_ms.mean",
                "dataset": dataset, "config": cfg,
                "field": "latency_p50_ms.mean", "value": m,
            })
            sources.append({
                "file": "ablation_parallel_v1.json",
                "json_path": f"results[{idx}].latency_p50_ms.ci95",
                "dataset": dataset, "config": cfg,
                "field": "latency_p50_ms.ci95", "value": ci,
            })
        ax.bar(
            x, means, width, yerr=[errs_lo, errs_hi], capsize=3,
            color=["#7f7f7f", "#1f77b4"], edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_title(dataset)
        ax.set_ylabel("p50 latency (ms)")
        ax.set_ylim(bottom=0)

        sp_idx, sp_elem = find_speedup(par["results"], dataset=dataset)
        sp = sp_elem["speedup"]["speedup_p50"]
        sp_mean = sp["mean"]
        sp_ci = sp["ci95"]
        sources.append({
            "file": "ablation_parallel_v1.json",
            "json_path": f"results[{sp_idx}].speedup.speedup_p50.mean",
            "dataset": dataset, "field": "speedup_p50.mean", "value": sp_mean,
        })
        sources.append({
            "file": "ablation_parallel_v1.json",
            "json_path": f"results[{sp_idx}].speedup.speedup_p50.ci95",
            "dataset": dataset, "field": "speedup_p50.ci95", "value": sp_ci,
        })

        top = max(m + e for m, e in zip(means, errs_hi))
        ax.set_ylim(top=top * 1.25)
        ax.text(
            x[1], means[1] + errs_hi[1] + top * 0.04,
            f"{sp_mean:.4f}×\n[95% CI {sp_ci[0]:.4f}–{sp_ci[1]:.4f}]",
            ha="center", va="bottom", fontsize=7.5,
        )

    fig.tight_layout()
    caption = (
        "Sequential vs parallel p50 latency (ms) on the parallelization "
        "ablation. Speedup factors are paired BCa bootstrap with n_resamples="
        "10000, seed=42 (definition stored alongside the value in the source "
        "JSON). Source: ablation_parallel_v1.json."
    )
    write_sidecar("parallel_speedup", caption, sources)
    save(fig, "parallel_speedup")


# ---------------------------------------------------------------------------
# FIG-3: cache_hit_rate
# ---------------------------------------------------------------------------

def fig3_cache_hit_rate() -> None:
    cache = load_json("ablation_cache_v1.json")
    modes = ["no_cache", "l1_exact_match", "repeat_warm"]
    datasets = ["customer_support_100", "customer_support_repeat_100"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    sources: list[dict] = []
    x = np.arange(len(modes))
    width = 0.38
    colors = {"customer_support_100": "#7f7f7f", "customer_support_repeat_100": "#1f77b4"}

    for j, dataset in enumerate(datasets):
        means: list[float] = []
        degenerate: list[bool] = []
        errs_lo: list[float] = []
        errs_hi: list[float] = []
        for mode in modes:
            idx, elem = find_result(cache["results"], dataset=dataset, config=mode)
            m = elem["cache_hit_rate"]["mean"]
            ci = elem["cache_hit_rate"]["ci95"]
            lo, hi = asym_err(m, ci)
            means.append(m)
            errs_lo.append(lo)
            errs_hi.append(hi)
            degenerate.append(ci[0] == ci[1])
            sources.append({
                "file": "ablation_cache_v1.json",
                "json_path": f"results[{idx}].cache_hit_rate.mean",
                "dataset": dataset, "config": mode,
                "field": "cache_hit_rate.mean", "value": m,
            })
            sources.append({
                "file": "ablation_cache_v1.json",
                "json_path": f"results[{idx}].cache_hit_rate.ci95",
                "dataset": dataset, "config": mode,
                "field": "cache_hit_rate.ci95", "value": ci,
            })
        offsets = x + (j - 0.5) * width
        ax.bar(
            offsets, means, width, yerr=[errs_lo, errs_hi], capsize=3,
            color=colors[dataset], edgecolor="black", linewidth=0.5,
            label=dataset,
        )
        for xi, m, deg in zip(offsets, means, degenerate):
            if deg:
                ax.text(
                    xi, m + 0.02, "deg.",
                    ha="center", va="bottom", fontsize=6.0, rotation=90,
                    color="#444",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("cache hit rate")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    caption = (
        "Cache hit rate by mode on the caching ablation. The unique-query "
        "workload (customer_support_100) demonstrates L1's null effect; the "
        "repeat-injected workload (customer_support_repeat_100) demonstrates "
        "L1 effectiveness. Bars whose 95% CI collapses to a single value "
        "(per_trial values identical across all 5 trials) are labeled "
        "'(degenerate CI)'. Source: ablation_cache_v1.json."
    )
    write_sidecar("cache_hit_rate", caption, sources)
    save(fig, "cache_hit_rate")


# ---------------------------------------------------------------------------
# FIG-4: type_safety_corpus  (3-way stack: compile / runtime / missed)
# ---------------------------------------------------------------------------

def fig4_type_safety_corpus() -> None:
    ts = load_json("ablation_typesafety_v1.json")
    buckets = ["type_mismatch", "undefined_reference", "missing_field", "duplicate_definition"]
    bucket_labels = ["TypeMismatch", "UndefinedReference", "MissingField", "DuplicateDefinition"]
    systems = ["Aether", "LangChain", "DSPy"]

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    sources: list[dict] = []
    x = np.arange(len(buckets))
    width = 0.26

    color_compile = "#2ca02c"  # green: caught at compile time
    color_runtime = "#ff7f0e"  # orange: caught at runtime
    color_missed = "#d62728"   # red: missed silently

    legend_done = False
    for j, sys_name in enumerate(systems):
        compile_seg: list[int] = []
        runtime_seg: list[int] = []
        missed_seg: list[int] = []
        for b in buckets:
            cell = ts["summary"]["by_bucket"][b]
            total = cell["total"]
            if sys_name == "Aether":
                caught_compile = cell["aether_caught"]
                caught_runtime = 0
                missed = total - caught_compile
                paths = [
                    (f"summary.by_bucket.{b}.aether_caught", caught_compile, "compile"),
                    (None, caught_runtime, "runtime"),
                    (f"summary.by_bucket.{b}.total - summary.by_bucket.{b}.aether_caught", missed, "missed"),
                ]
            elif sys_name == "LangChain":
                caught_compile = 0
                caught_runtime = cell["lc_caught_at_runtime"]
                missed = total - caught_runtime
                paths = [
                    (None, caught_compile, "compile"),
                    (f"summary.by_bucket.{b}.lc_caught_at_runtime", caught_runtime, "runtime"),
                    (f"summary.by_bucket.{b}.total - summary.by_bucket.{b}.lc_caught_at_runtime", missed, "missed"),
                ]
            else:  # DSPy
                caught_compile = 0
                caught_runtime = cell["dspy_caught_at_runtime"]
                missed = total - caught_runtime
                paths = [
                    (None, caught_compile, "compile"),
                    (f"summary.by_bucket.{b}.dspy_caught_at_runtime", caught_runtime, "runtime"),
                    (f"summary.by_bucket.{b}.total - summary.by_bucket.{b}.dspy_caught_at_runtime", missed, "missed"),
                ]
            compile_seg.append(caught_compile)
            runtime_seg.append(caught_runtime)
            missed_seg.append(missed)
            for jp, val, kind in paths:
                sources.append({
                    "file": "ablation_typesafety_v1.json",
                    "json_path": jp if jp else f"(constant 0; {sys_name} has no {kind} segment)",
                    "system": sys_name, "bucket": b, "segment": kind,
                    "value": val,
                })
            sources.append({
                "file": "ablation_typesafety_v1.json",
                "json_path": f"summary.by_bucket.{b}.total",
                "bucket": b, "field": "total", "value": total,
            })

        offsets = x + (j - 1) * width
        ax.bar(offsets, compile_seg, width, color=color_compile,
               edgecolor="black", linewidth=0.5,
               label="caught_at_compile_time" if not legend_done else None)
        ax.bar(offsets, runtime_seg, width, bottom=compile_seg,
               color=color_runtime, edgecolor="black", linewidth=0.5,
               label="caught_at_runtime" if not legend_done else None)
        bottom2 = [a + b for a, b in zip(compile_seg, runtime_seg)]
        ax.bar(offsets, missed_seg, width, bottom=bottom2,
               color=color_missed, edgecolor="black", linewidth=0.5,
               label="missed_silently" if not legend_done else None)
        legend_done = True

        for xi, b in zip(offsets, buckets):
            cell = ts["summary"]["by_bucket"][b]
            ax.text(xi, cell["total"] + 0.15, sys_name[0], ha="center",
                    va="bottom", fontsize=7, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=10, ha="right")
    ax.set_ylabel("test cases (n)")
    ax.set_ylim(0, max(ts["summary"]["by_bucket"][b]["total"] for b in buckets) + 1.5)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.set_title("Per-bucket detection outcome (A=Aether, L=LangChain, D=DSPy)")

    fig.tight_layout()

    cd_note = ts.get("methodology_notes", {}).get("cd_substitution", "")
    cd_summary = (
        "the original 'circular_dependency' bucket was substituted with "
        "'duplicate_definition' due to a compiler pass-ordering gap "
        "(SemanticError::CircularDependency is defined but preempted by other "
        "semantic-analysis errors)"
    )
    caption = (
        f"Type-safety bug-detection outcome by class "
        f"(n={ts['summary']['by_bucket']['type_mismatch']['total']}/"
        f"{ts['summary']['by_bucket']['undefined_reference']['total']}/"
        f"{ts['summary']['by_bucket']['missing_field']['total']}/"
        f"{ts['summary']['by_bucket']['duplicate_definition']['total']}). "
        f"Aether catches {ts['summary']['aether_caught']} cases at compile time; "
        f"LangChain and DSPy catch {ts['summary']['lc_caught_at_runtime']} at runtime "
        f"via Python exceptions and miss {ts['summary']['lc_missed_silently']} silently. "
        f"Methodology note: {cd_summary} (full text in "
        f"ablation_typesafety_v1.json#methodology_notes.cd_substitution). "
        f"Source: ablation_typesafety_v1.json."
    )
    sources.append({
        "file": "ablation_typesafety_v1.json",
        "json_path": "methodology_notes.cd_substitution",
        "field": "methodology_notes.cd_substitution",
        "value": cd_note,
    })
    write_sidecar("type_safety_corpus", caption, sources)
    save(fig, "type_safety_corpus")


# ---------------------------------------------------------------------------
# FIG-5: security_outcome
# ---------------------------------------------------------------------------

def fig5_security_outcome() -> None:
    sec = load_json("security_v1.json")
    cfg_order = ["aether_taint_on", "aether_taint_off", "langchain_baseline"]
    cfg_index = {c["config"]: (i, c) for i, c in enumerate(sec["configs"])}
    completed = sec["metadata"].get("completed_live_configs", [])

    def metric(cfg_name: str, metric_name: str) -> tuple[int, int, dict]:
        i, c = cfg_index[cfg_name]
        for k, m in enumerate(c["metrics"]):
            if m["metric"] == metric_name:
                return i, k, m
        raise KeyError(f"missing metric {metric_name!r} for config {cfg_name!r}")

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 3.0))
    sources: list[dict] = []

    # Panel A: compile_time_catch_rate
    axA = axes[0]
    means_a: list[float] = []
    errs_lo: list[float] = []
    errs_hi: list[float] = []
    for cfg_name in cfg_order:
        ci_idx, m_idx, m = metric(cfg_name, "compile_time_catch_rate")
        means_a.append(m["mean"])
        lo, hi = asym_err(m["mean"], m["ci95"])
        errs_lo.append(lo)
        errs_hi.append(hi)
        sources.append({
            "file": "security_v1.json",
            "json_path": f"configs[{ci_idx}].metrics[{m_idx}].mean",
            "config": cfg_name, "field": "compile_time_catch_rate.mean", "value": m["mean"],
        })
        sources.append({
            "file": "security_v1.json",
            "json_path": f"configs[{ci_idx}].metrics[{m_idx}].ci95",
            "config": cfg_name, "field": "compile_time_catch_rate.ci95", "value": m["ci95"],
        })
    xA = np.arange(len(cfg_order))
    axA.bar(
        xA, means_a, 0.6, yerr=[errs_lo, errs_hi], capsize=3,
        color=["#2ca02c", "#d62728", "#d62728"], edgecolor="black", linewidth=0.5,
    )
    axA.set_xticks(xA)
    axA.set_xticklabels(cfg_order, rotation=15, ha="right")
    axA.set_ylabel("compile-time catch rate")
    axA.set_ylim(0, 1.18)
    axA.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axA.set_title("Panel A: compile-time defense")
    for i, cfg_name in enumerate(cfg_order):
        if cfg_name != "aether_taint_on":
            axA.text(xA[i], 0.04, "no compile-time\ndefense", ha="center", va="bottom",
                     fontsize=7, color="#444")

    # Panel B: attack_success_rate (taint_on did not run any LLM calls)
    axB = axes[1]
    xB = np.arange(len(cfg_order))
    live_zero_count = 0
    for i, cfg_name in enumerate(cfg_order):
        if cfg_name == "aether_taint_on":
            # render hatched empty bar at full height with explicit "n/a" annotation
            axB.bar([xB[i]], [1.0], 0.6, color="white", hatch="////",
                    edgecolor="#444", linewidth=0.5, alpha=0.5)
            axB.text(xB[i], 0.5, "compiled-away,\nno runtime\nexposure",
                     ha="center", va="center", fontsize=7, color="#222")
            sources.append({
                "file": "security_v1.json",
                "json_path": "metadata.completed_live_configs",
                "config": cfg_name, "field": "completed_live_configs",
                "value": completed,
                "note": "aether_taint_on omitted from live runs: every case has compiled=false, ran_llm=false; the panel-B slot is rendered as 'no runtime exposure'",
            })
        else:
            ci_idx, m_idx, m = metric(cfg_name, "attack_success_rate")
            lo, hi = asym_err(m["mean"], m["ci95"])
            color = "#d62728" if cfg_name == "langchain_baseline" else "#ff7f0e"
            axB.bar([xB[i]], [m["mean"]], 0.6,
                    yerr=[[lo], [hi]], capsize=3,
                    color=color, edgecolor="black", linewidth=0.5)
            sources.append({
                "file": "security_v1.json",
                "json_path": f"configs[{ci_idx}].metrics[{m_idx}].mean",
                "config": cfg_name, "field": "attack_success_rate.mean", "value": m["mean"],
            })
            sources.append({
                "file": "security_v1.json",
                "json_path": f"configs[{ci_idx}].metrics[{m_idx}].ci95",
                "config": cfg_name, "field": "attack_success_rate.ci95", "value": m["ci95"],
            })
            if m["mean"] == 0.0 and m["ci95"][0] == m["ci95"][1]:
                live_zero_count += 1
                axB.text(xB[i], 0.04, "0.0", ha="center", va="bottom",
                         fontsize=7, color="#444")
    if live_zero_count == 2:
        # Single, non-overlapping note placed under the panel title.
        axB.text(0.5, 0.94, "live configs: ASR=0.0 (model self-refused)",
                 transform=axB.transAxes, ha="center", va="top",
                 fontsize=7, color="#444",
                 bbox=dict(facecolor="white", edgecolor="#bbb",
                           boxstyle="round,pad=0.3", linewidth=0.5))
    axB.set_xticks(xB)
    axB.set_xticklabels(cfg_order, rotation=15, ha="right")
    axB.set_ylabel("attack success rate (ASR)")
    axB.set_ylim(0, 1.18)
    axB.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axB.set_title("Panel B: runtime ASR")

    fig.tight_layout()

    caption = (
        f"Panel A: compile-time taint-tracking catch rate by configuration. "
        f"Panel B: runtime attack success rate (ASR) on the InjecAgent-adapted "
        f"direct-prompt-injection suite (n={sec['metadata']['cases_per_trial']}+"
        f"{sec['metadata']['cases_per_trial']} deduped, "
        f"trials_per_config={sec['metadata']['trials_per_config']}, "
        f"model={sec['metadata']['model']}). aether_taint_on rejects every "
        f"unsafe program at compile time, so no LLM calls are issued — its "
        f"Panel-B slot is rendered hatched and labeled 'compiled-away, no "
        f"runtime exposure' (per metadata.completed_live_configs="
        f"{completed}). aether_taint_off and langchain_baseline both show "
        f"measured ASR=0.0 in this run, reflecting the model's own refusal "
        f"behavior in the absence of compile-time defense; the asymmetry "
        f"between Panel A (1.0 vs 0.0 vs 0.0) and Panel B (n/a vs 0.0 vs 0.0) "
        f"is the headline finding. Source: security_v1.json."
    )
    write_sidecar("security_outcome", caption, sources)
    save(fig, "security_outcome")


# ---------------------------------------------------------------------------

def main() -> int:
    if not RESULTS.is_dir():
        print(f"ERROR: results dir not found: {RESULTS}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    fig1_cross_system_latency()
    fig2_parallel_speedup()
    fig3_cache_hit_rate()
    fig4_type_safety_corpus()
    fig5_security_outcome()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
