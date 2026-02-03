# Aether Programming Language - Reproducible Build Environment
# This Dockerfile creates a reproducible environment for building and testing
# the Aether programming language implementation.

FROM ubuntu:22.04

# Set environment variables for reproducible builds
ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_VERSION=1.75.0
ENV NODE_VERSION=20.18.0
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
    --default-toolchain $RUST_VERSION \
    --profile minimal \
    --no-modify-path \
    -y

# Install Node.js
RUN curl -fsSL https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz \
    | tar -xJ -C /usr/local --strip-components=1

# Install pnpm
RUN npm install -g pnpm@latest

# Create working directory
WORKDIR /aether

# Copy source code
COPY . .

# Build the Aether workspace (compiler + runtime)
RUN cargo build --workspace --release && \
    cargo test --workspace && \
    cargo doc --workspace --no-deps

# Run compiler-specific tests with verbose output
RUN cargo test -p aether-compiler --release -- --nocapture

# Build the DAG visualizer
RUN cd aether-dag-visualizer && \
    pnpm install && \
    pnpm run build

# Install Python dependencies for benchmarking
RUN cd bench/lm-evaluation-harness && \
    pip3 install -e . && \
    pip3 install evaluate datasets

# Set up runtime environment
EXPOSE 3000 5173

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Default command runs the Aether runtime
CMD ["./aether-runtime/target/release/aether-runtime"]

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

