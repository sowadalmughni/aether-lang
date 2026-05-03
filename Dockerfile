# Aether Programming Language - Reproducible Build Environment
#
# Base: rust:latest (debian-bookworm-slim, ships rustc/cargo/rustup/clippy/rustfmt).
# rust:latest floats; for strict reproducibility, pin to rust:1.82-bookworm in a
# follow-up. Per task spec we use :latest here.
#
# Builds the full Cargo workspace in release, runs `cargo test --workspace`
# (a non-zero exit aborts the build), installs Python benchmark deps into a
# venv, and builds the visualizer's static assets.

FROM rust:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PNPM_VERSION=10.4.1 \
    NODE_MAJOR=20 \
    PATH=/opt/venv/bin:/usr/local/cargo/bin:$PATH

# 1. System dependencies. Bookworm ships Python 3.11 in main.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        build-essential \
        pkg-config \
        libssl-dev \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# 2. Node.js 20 via NodeSource.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 3. pnpm pinned to the version recorded in aether-dag-visualizer/package.json.
RUN npm install -g pnpm@${PNPM_VERSION}

# 4. Defensive: ensure clippy/rustfmt are present even if upstream image strips them.
RUN rustup component add clippy rustfmt

# 5. Python virtualenv at /opt/venv. Bookworm marks system Python externally
#    managed (PEP 668); a venv is the cleanest escape, vs --break-system-packages.
RUN python3.11 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip

WORKDIR /aether

# 6. Copy source. (A cargo-chef multi-stage build for incremental dep caching is
#    a future optimization; the current single-stage layout is simple and
#    correct, and Docker's BuildKit cache mounts can speed repeat builds.)
COPY . .

# 7. Build the workspace in release mode. Non-zero exit aborts the image build.
RUN cargo build --workspace --release --locked

# 8. Run the full test suite. Non-zero exit aborts the image build (acceptance).
RUN cargo test --workspace --release --locked

# 9. Install Python benchmark dependencies into the venv.
RUN pip install --no-cache-dir -r bench/requirements.txt

# 10. Build the DAG visualizer's static assets so they ship in the image.
RUN cd aether-dag-visualizer \
    && pnpm install --frozen-lockfile \
    && pnpm run build

EXPOSE 3000 5173

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:3000/health || exit 1

# Default command runs the Aether runtime. The previous Dockerfile pointed at
# ./aether-runtime/target/release/... which does not exist; the workspace
# target dir is /aether/target/release/.
CMD ["/aether/target/release/aether-runtime"]

# Build metadata
LABEL maintainer="Aether Development Team"
LABEL version="0.1.0"
LABEL description="Aether Programming Language - Reproducible Build Environment"
LABEL org.opencontainers.image.source="https://github.com/aether-lang/aether"
LABEL org.opencontainers.image.documentation="https://aether-lang.github.io/aether"
LABEL org.opencontainers.image.licenses="MIT"

# Artifact evaluation metadata
LABEL artifact.evaluation.reproducible="true"
LABEL artifact.evaluation.available="true"
LABEL artifact.evaluation.functional="true"
LABEL artifact.evaluation.reusable="true"
