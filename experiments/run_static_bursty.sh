#!/usr/bin/env bash
# Run static chunk-size controller with bursty (Poisson 4 req/s) workload.
# Usage: bash experiments/run_static_bursty.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "Running: static_bursty"
echo "============================================"
python -m sarathi.benchmark.main \
    --config "experiments/configs/static_bursty.yaml"

echo ""
echo "Done. Results in experiments/results/static_bursty/"
