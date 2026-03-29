#!/usr/bin/env bash
# Run PID chunk-size controller with steady (Poisson 2 req/s) workload.
# Usage: bash experiments/run_pid_steady.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "Running: pid_steady"
echo "============================================"
python -m sarathi.benchmark.main \
    --config "experiments/configs/pid_steady.yaml"

echo ""
echo "Done. Results in experiments/results/pid_steady/"
