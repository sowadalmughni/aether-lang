# Aether Benchmark Suite

This directory contains the benchmarking infrastructure for the Aether programming language runtime, including a forked version of `lm-evaluation-harness` with an Aether adapter.

## Hardware Specifications

**Test Environment:**
- OS: Ubuntu 22.04 LTS (linux/amd64)
- CPU: Virtual CPU (sandbox environment)
- Memory: Available system memory
- Network: Internet-connected sandbox environment

**Note:** These benchmarks are run in a sandboxed virtual environment. For production benchmarking, use dedicated hardware with consistent specifications.

## Dataset Links

The evaluation harness supports multiple datasets. Key datasets used for Aether evaluation include:

- **HellaSwag**: Common sense reasoning
  - Source: https://github.com/rowanz/hellaswag
  - Task: Multiple choice completion
  
- **PIQA**: Physical interaction reasoning
  - Source: https://github.com/ybisk/ybisk.github.io/tree/master/piqa
  - Task: Physical reasoning questions

- **Winogrande**: Commonsense reasoning
  - Source: https://github.com/allenai/winogrande
  - Task: Pronoun resolution

- **ARC**: AI2 Reasoning Challenge
  - Source: https://allenai.org/data/arc
  - Task: Grade-school science questions

## Running Benchmarks

### Prerequisites

1. Ensure the Aether runtime is running:
   ```bash
   cd ../aether-runtime
   cargo run &
   ```

2. Install Python dependencies:
   ```bash
   cd lm-evaluation-harness
   pip install -e .
   ```

### Basic Evaluation

Run a simple evaluation with the Aether stub:

```bash
cd lm-evaluation-harness
python -m lm_eval \
    --model aether_stub \
    --model_args aether_url=http://localhost:3000,model_name=gpt-4o \
    --tasks hellaswag \
    --num_fewshot 0 \
    --batch_size 1 \
    --limit 10
```

### Performance Benchmarking

Run latency and throughput benchmarks:

```bash
cd lm-evaluation-harness
python -m lm_eval \
    --model aether_stub \
    --model_args aether_url=http://localhost:3000,model_name=gpt-4o \
    --tasks piqa,hellaswag \
    --num_fewshot 0 \
    --batch_size 1 \
    --limit 50 \
    --output_path ../results/
```

### Exact CLI Commands

**Standard Evaluation:**
```bash
python -m lm_eval --model aether_stub --model_args aether_url=http://localhost:3000 --tasks hellaswag --num_fewshot 0 --limit 10
```

**Performance Test:**
```bash
python -m lm_eval --model aether_stub --model_args aether_url=http://localhost:3000 --tasks piqa --num_fewshot 0 --limit 20 --output_path results/performance_test.json
```

**Multi-task Evaluation:**
```bash
python -m lm_eval --model aether_stub --model_args aether_url=http://localhost:3000 --tasks hellaswag,piqa,winogrande --num_fewshot 0 --limit 5
```

## Metrics Collected

The Aether runtime exposes the following metrics via Prometheus at `http://localhost:3000/metrics`:

- `aether_executed_nodes_total`: Total number of executed nodes
- `aether_token_cost_total`: Total simulated token cost
- `aether_errors_total`: Total number of errors
- `aether_execution_time_seconds`: Node execution time distribution

## CI/CD Integration

For continuous benchmarking, use GitHub Actions with the following workflow:

```yaml
name: Nightly Benchmarks
on:
  schedule:
    - cron: '0 2 * * *'  # Run at 2 AM daily
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Build Aether Runtime
        run: cd aether-runtime && cargo build --release
      - name: Start Runtime
        run: cd aether-runtime && cargo run --release &
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: cd bench/lm-evaluation-harness && pip install -e .
      - name: Run benchmarks
        run: |
          cd bench/lm-evaluation-harness
          python -m lm_eval --model aether_stub --tasks hellaswag --limit 50 --output_path ../../results/
      - name: Upload results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'customSmallerIsBetter'
          output-file-path: results/results.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

## Results Directory Structure

```
bench/
├── results/
│   ├── performance_test.json
│   ├── nightly_YYYY-MM-DD.json
│   └── comparison_charts/
├── lm-evaluation-harness/
└── README.md
```

## Troubleshooting

**Runtime Connection Issues:**
- Ensure Aether runtime is running on port 3000
- Check firewall settings
- Verify runtime health: `curl http://localhost:3000/health`

**Evaluation Errors:**
- Check Python dependencies: `pip list | grep lm-eval`
- Verify task names: `python -m lm_eval --list_tasks`
- Review runtime logs for security violations or errors

**Performance Issues:**
- Monitor system resources during benchmarks
- Adjust batch sizes for available memory
- Use `--limit` parameter to control test size

