#!/usr/bin/env bash
# Run static chunk-size controller with steady (Poisson 2 req/s) workload.
# Usage: bash experiments/run_static_steady.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "Running: static_steady"
echo "============================================"
python -m sarathi.benchmark.main \
    --config "experiments/configs/static_steady.yaml"

echo ""
echo "Done. Results in experiments/results/static_steady/"
