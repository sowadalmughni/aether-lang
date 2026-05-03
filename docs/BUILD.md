# Building Aether

Aether builds and runs cleanly on Linux. This document covers two paths:

- [Docker (recommended)](#docker-recommended) — single command, no host
  toolchain setup, mirrors the CI environment.
- [Native Linux](#native-linux) — direct host install, useful for
  iterative development.

A note on Windows: native Windows (MSVC) builds are not supported. Use
[WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and follow
the native Linux path inside the WSL2 distro, or use
[Docker Desktop](https://docs.docker.com/desktop/install/windows-install/).

## Docker (recommended)

Prerequisites: Docker 24+ (Engine or Desktop). No other host packages.

### Build the image

```bash
docker build -t aether .
```

Verifies the entire workspace by running `cargo build --workspace --release`
followed by `cargo test --workspace --release` inside the image. A non-zero
exit on either step aborts the build.

### Run the test suite in a container

```bash
docker run --rm aether cargo test --workspace
```

### Run the runtime + benchmark via Compose

```bash
docker compose up --build runtime    # starts the runtime on :3000
docker compose run --rm bench         # runs the benchmark, writes JSON
docker compose down
```

The `bench` service depends on `runtime` becoming healthy (`/health`
returning 200), then runs `scripts/run_benchmark.py --all --requests 100`
against it with `AETHER_PROVIDER=mock`. Output JSON files land on the host
in `bench/results/` via a volume mount.

## Native Linux

Tested on Debian/Ubuntu. Other distributions work with equivalent packages.

### Prerequisites

- `rustup` with stable toolchain (and `clippy` + `rustfmt` components)
- Python 3.11+
- Node.js 20.x (for the visualizer)
- `pnpm@10.4.1` (matches the lockfile)
- `build-essential`, `pkg-config`, `libssl-dev` (for native deps)

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential pkg-config libssl-dev curl ca-certificates git \
  python3 python3-venv python3-pip
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup component add clippy rustfmt
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pnpm@10.4.1
```

### Build, lint, and test

```bash
cargo build --workspace --release
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

### Run the runtime and a benchmark

```bash
# Terminal 1 — start the runtime (binds 0.0.0.0:3000)
AETHER_PROVIDER=mock ./target/release/aether-runtime &

# Wait for it to be healthy
curl --retry 10 --retry-delay 1 --retry-connrefused http://localhost:3000/health

# Set up a Python venv for the benchmark deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r bench/requirements.txt

# Run the benchmark with the mock provider. run_benchmark.py reads
# AETHER_PROVIDER from the environment; there is no --mode flag.
AETHER_PROVIDER=mock python scripts/run_benchmark.py \
  --all --requests 100 --output bench/results/

ls bench/results/   # benchmark_YYYYMMDD_HHMMSS.json
```

### Visualizer (development)

```bash
cd aether-dag-visualizer
pnpm install
pnpm dev          # http://localhost:5173
```

For a static production bundle:

```bash
pnpm run build    # output in aether-dag-visualizer/dist/
```

## Continuous integration

Two GitHub Actions workflows run on `ubuntu-latest`:

- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — runs
  `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test
  --workspace`, then a mock-provider benchmark in a separate job. Uploads
  benchmark JSON as an artifact.
- [`.github/workflows/benchmark.yml`](../.github/workflows/benchmark.yml)
  — full benchmark with baselines, runs nightly and on PRs touching the
  runtime / bench paths. Comments results on PRs.

Both workflows write benchmark JSON to `bench/results/`.

## Trade-offs and known issues

- **`rust:latest` floats.** The Dockerfile uses `rust:latest` (per task
  spec), which currently tracks debian trixie + rust 1.x. Pin to a
  specific tag (e.g. `rust:1.82-bookworm`) when strict reproducibility is
  required for an artifact-evaluation submission.
- **No `--locked` on `cargo build` in the Dockerfile.** `Cargo.lock` on
  `main` has drifted from at least one workspace member's `Cargo.toml`,
  so `--locked` blocks the build. Once that drift is resolved (run
  `cargo update --workspace` and commit the new lockfile), `--locked`
  can be re-enabled in the Dockerfile and CI for stronger reproducibility.
- **`pnpm@10.4.1` is pinned** to match `aether-dag-visualizer/package.json`.
  Running with a different pnpm major version produces lockfile churn.
- **Provider selection** is via the `AETHER_PROVIDER` environment variable
  (`mock`, `openai`, `anthropic`). The CI mock-benchmark uses `mock`;
  real-provider runs require API keys and are not part of CI.
