#!/usr/bin/env bash
# Run PID chunk-size controller with bursty (Poisson 4 req/s) workload.
# Usage: bash experiments/run_pid_bursty.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "Running: pid_bursty"
echo "============================================"
python -m sarathi.benchmark.main \
    --config "experiments/configs/pid_bursty.yaml"

echo ""
echo "Done. Results in experiments/results/pid_bursty/"
