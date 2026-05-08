# Reproducibility — Aether Whitepaper

This document describes how a third party can reproduce the benchmark
numbers cited in `whitepaper/WHITEPAPER_ACADEMIC.md` (Aether 3.2-academic).
It is the canonical companion to `reproduce.sh` and the committed JSON
result files under `bench/results/`.

The TL;DR is one line:

```bash
bash reproduce.sh
```

…on a fresh clone with Docker installed produces `bench/results/repro/`
and a diff report at `bench/results/repro/DIFF_REPORT.md`. That report
must show every mock-mode JSON either `ok` (within tolerance) or
`n/a (requires OPENAI_API_KEY)` (real-API; see §6).

---

## 1. Hardware used to produce the committed numbers

The 15 JSON files under `bench/results/` were produced on two distinct
host environments. Each JSON's `hardware` block carries the truth for
that file; the table below summarises:

| File family                          | CPU                                          | RAM    | OS                  |
| ------------------------------------ | -------------------------------------------- | ------ | ------------------- |
| `ablation_*_v1.json`, `aether_mock_v1.json`, `langchain_v1.json`, `dspy_v1.json`, `*_real_api_*.json`, `real_api_v1.json`, `security_v1.json` | Intel(R) Core(TM) i5-8250U @ 1.60 GHz | 7.71 GB | Ubuntu 24.04.4 LTS  |
| `hotpotqa_*_v1.json`                 | Intel64 Family 6 Model 142 Stepping 10 (i5-8250U) | (Windows perf counter, 0.0 reported) | Windows 11 (10.0.26200) |

Mock-mode wall-clock latency numbers (`latency_p50_ms`, etc.) are
hardware-bound. Hardware variance of `±10–25%` is expected and accepted
by `bench/verify_tolerance.toml`. Deterministic mock fields (token counts,
EM/F1 zero in the mock-LLM regime, cache hit rates) must match exactly.

A reproducer running on faster hardware (e.g. Apple M-series, recent
Xeon) will see lower absolute latencies and a tighter CI; the diff
script flags such drift as informational, not a failure.

## 2. Software versions

### 2.1 Pinned

These are the artefacts committed in the repo at the tagged release
commit; they fully determine the build inputs.

| Component        | Pin                                          | Where pinned                |
| ---------------- | -------------------------------------------- | --------------------------- |
| Rust toolchain   | `rust:latest` Docker image (debian-trixie at the time of writing, ships rustc/cargo + Python 3.13) | `Dockerfile` line 12 |
| Cargo workspace  | SHA-256 of `Cargo.lock` = `27b13b9ce6a26fae3b2932c99844932c67fc7f40d65fbb8f8e1d37bd1ab7f749` | `Cargo.lock` |
| pnpm             | `10.4.1`                                     | `Dockerfile` `PNPM_VERSION` |
| Node.js          | `20` (NodeSource)                            | `Dockerfile` `NODE_MAJOR`   |
| Python deps (bench) | exact pins (see below)                    | `bench/requirements.txt`    |

Python pinned packages — these are tied to the numbers in
`bench/results/langchain_v1.json`, `bench/results/dspy_v1.json`, and the
real-API counterparts. Loosening the pins would break artefact
reproducibility.

```
langchain==0.3.28
langchain-core==0.3.84
langchain-openai==0.2.14
langchain-anthropic==0.2.4
dspy-ai==2.6.27
dspy==2.6.27
openai==1.55.0
```

DSPy is double-pinned (`dspy-ai==2.6.27` and `dspy==2.6.27`) because
`dspy-ai` is now a metadata-only alias for `dspy`; pinning only the
former lets `dspy` float to 3.2.x. See `bench/requirements.txt` comments.

### 2.2 Floating (intentionally)

| Component | Reason |
| --------- | ------ |
| `rust:latest` upstream tag | The Dockerfile chose to let `rustc` track the upstream `latest` rather than pinning a major version. This is acknowledged in `Dockerfile` lines 5–6; for strict bit-for-bit reproduction, downstream consumers can repin to `rust:1.<MAJOR>-<distro>` once Cargo.lock is reconciled with `--locked`. The current Dockerfile intentionally regenerates the lock at build time. |
| `datasets`, `numpy`, `scipy`, `httpx` | range pins; the result JSON files carry the `version` (git SHA) of the producing repo, not these libs' versions. Empirically these libs' minor releases have not perturbed the numbers. |

## 3. Git commit producing this document

```
HEAD = 2d3cff49bd48064c9ceaa0824afc91a85b787004
branch = chore/linux-build-ci
```

When the `paper-v3.2-academic` tag is created, that commit is the
canonical reference for everything below. To reproduce against the tag:

```bash
git clone <repo-url>
cd aether-lang
git checkout paper-v3.2-academic
bash reproduce.sh
```

## 4. Expected runtime

On the reference Linux hardware (i5-8250U, 8 GB RAM, Ubuntu 24.04):

| Phase                                    | Expected wall-clock |
| ---------------------------------------- | ------------------- |
| Docker image build (cold cache)          | ~15 min             |
| `aether_mock_v1` suite (`scripts/run_benchmark.py --suite --trials 5`) | ~15 min |
| `ablation_cache` (5 trials × 3 modes × 2 datasets) | ~25 min |
| `ablation_parallel` (5 trials × 2 modes × 2 datasets) | ~10 min |
| `ablation_typesafety` (30 fixed cases)   | ~2 min              |
| `langchain_v1` baseline (mock, 5 trials) | ~15 min             |
| `dspy_v1` baseline (mock, 5 trials)      | ~15 min             |
| HotpotQA × 3 systems (mock, 3 trials × 500 items) | ~33 min     |
| Diff (verify_reproduction.py)            | <30 s               |
| **Total cold cache**                     | **~125–145 min**    |
| **Total warm cache** (image already built) | **~110–125 min** |

Faster hardware shortens the benchmark phases proportionally. The
1-hour goal that originally drove this script was relaxed in design
review (in favour of full trial counts that actually pair with the
committed CIs); the script's `--trials` and dataset sizes therefore
mirror the committed runs.

## 5. Expected variance and what the diff tolerates

`bench/verify_tolerance.toml` declares:

* **Exact match required** — `system`, `provider`, `datasets`, `configs`,
  `dataset_size`, `trials_per_config`, `tokens_*`, `cache_hit_rate`,
  `em`, `f1`, `prec`, `recall`, type-safety per-case `result`, `exit_code`,
  `error_class_matched`, etc. These are deterministic in mock mode.

* **Tolerant numeric** — `latency_p50_ms` (±25%), `latency_p95_ms`
  (±30%), `latency_p99_ms` (±50%), `total_time_ms` (±30%). The diff
  widens the reference's `ci95` by the listed fractional tolerance and
  fails only if the produced `mean` falls outside.

* **Ignored** — `measured_at`, `git_sha`, `version` (git-SHA labels),
  `hardware`, free-text `notes`, `stderr_excerpt`, `stdout_excerpt`,
  `traceback_first_line`, and large `raw_trial_results.*.latencies_ms`
  arrays (the diff uses derived percentiles, not raw arrays).

Outcomes:

* **Exit 0 + "VERDICT: clean"** — every mock JSON matched within
  tolerance, every real-API JSON skipped as expected.
* **Exit 0 + "numeric drift within informational tolerance"** — at
  least one latency was outside the widened CI but no structural fields
  diverged. Acceptable on faster/slower hardware.
* **Exit 1 + "STRUCTURAL MISMATCH"** — a deterministic field changed
  (e.g. tokens or `result` flipped). This indicates a real regression
  or a bug in the runner; investigate before claiming reproduction.

## 6. Real-API runs — explicitly out of scope for `reproduce.sh`

Five JSON files were produced against the real OpenAI API (`gpt-4o-mini`)
and cannot be reproduced in mock mode:

| File                          | Why mock can't reproduce it                               |
| ----------------------------- | --------------------------------------------------------- |
| `aether_real_api_v1.json`     | Real LLM responses, real latency, real cost               |
| `langchain_real_api_v1.json`  | "                                                         |
| `dspy_real_api_v1.json`       | "                                                         |
| `real_api_v1.json`            | Aggregate of the three above + cost accounting            |
| `security_v1.json`            | InjecAgent attack-success rate measured against gpt-4o-mini |

`reproduce.sh` skips them; `bench/verify_reproduction.py` marks them
`n/a (requires OPENAI_API_KEY)` in the diff report. To reproduce them
deliberately:

```bash
export OPENAI_API_KEY=sk-...
bash bench/run_real_api_benchmark.sh           # all 4 baseline real-API JSONs
python -m bench.security.run_security_bench --confirm-cost  # security_v1.json
```

These cost approximately USD 0.50 in total on the committed configuration
(see `bench/results/real_api_v1.json` `metadata.cost_usd`).

## 7. SHA-256 hashes of committed result JSONs

These are the authoritative hashes anchored to commit `2d3cff4`. The
release-tag annotation duplicates them; running `sha256sum
bench/results/*.json` on a fresh clone must match.

```
cd379ff1cf4e33d0a72f6fe691f3f5dd41b5c6a6b426bdb61e9d590073792e35  bench/results/ablation_cache_v1.json
4be6390e46469feddb4bcc9505b19d6fee1bc850f7228bb93080b82cc7077dc5  bench/results/ablation_parallel_v1.json
0d6b434b2d02d9280a852bc75ee375c8b6c0b574afc5dcafa2b56b4b1c4dd238  bench/results/ablation_typesafety_v1.json
0af99d9baef75d3af4f3c5d37b4e85928239bf42104ffc538d23f51d5b4e50b1  bench/results/aether_mock_v1.json
81310a7bc978f3c3fc417bedc5906a4b6c85854e02d9bbb70ca3609325ba9356  bench/results/aether_real_api_v1.json
72ff4917ed2953b9913029c421a50ff7cab807224b80c01c4960ab1daa683a6b  bench/results/benchmark_20260503_084135.json
61a16879858b1e58962fc05c428c0395db7b154dff2707f82b826ce206195c39  bench/results/dspy_real_api_v1.json
bcff87d3953dbec86ce336baa80cdcee5dac09e11e0d2cfb4098d4cee99e90cf  bench/results/dspy_v1.json
9d178b3576126669afaf859c344f6692f0e7f730b1cef39ef90b3739ecf2a7d9  bench/results/hotpotqa_aether_v1.json
1550917ae36e28f84b43517d88751fc456c59bdd833e5d8e385fbf6854b74b33  bench/results/hotpotqa_dspy_v1.json
59476627cf5edfe3a3af454624cc1d0860b780c8b40d7188927a9ae2d8e59e2f  bench/results/hotpotqa_langchain_v1.json
260554826659524537ce9e4223ea99897b9a34691d8c4993379271d6343b374a  bench/results/langchain_real_api_v1.json
2a61f96f2dba2456dc39e227d0999d6e7ebd0c248b50be69accc68879ee3cac0  bench/results/langchain_v1.json
f9e47ad10eadae0228e2e5a397d5929ca5353b9b285329e9d8f73b5fcc124c33  bench/results/real_api_v1.json
9d3ba96e0f049ede4718634135a486a00439678af2fd537957da1f53deb497b0  bench/results/security_v1.json
```

## 8. Troubleshooting checklist

If `reproduce.sh` exits non-zero or the diff report flags structural
mismatches, work down this list:

1. **Docker version** — `docker --version` should report ≥ 24. Older
   Docker may not support `docker compose` (v2). Use `docker-compose`
   (legacy v1) only as a last resort.
2. **`AETHER_PROVIDER=mock`** is set inside the bench container — the
   compose file declares this; verify with
   `docker compose run --rm bench env | grep AETHER`.
3. **Runtime healthy** — `docker inspect -f '{{.State.Health.Status}}'
   aether-runtime` should return `healthy`. If `unhealthy`, inspect
   `docker logs aether-runtime`.
4. **`bench/results/repro/` writable** — the compose file bind-mounts
   `./bench/results:/aether/bench/results`; on Windows, ensure the
   path is shared in Docker Desktop settings.
5. **Trials count drift** — if you changed `--trials` in
   `bench/run_all_mock.py`, the diff will flag `trials_per_config` as
   a structural mismatch. Restore the committed values (5 for
   ablations and baselines, 3 for HotpotQA).
6. **`security_v1.json` flagged not-n/a** — verify
   `bench/verify_tolerance.toml` `requires_real_api` still lists it.

If the structural mismatch persists after this checklist, file an issue
including `bench/results/repro/DIFF_REPORT.md` and the output of
`docker logs aether-runtime`.
