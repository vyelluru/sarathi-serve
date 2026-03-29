#!/usr/bin/env bash
# Run AIMD chunk-size controller with bursty (Poisson 4 req/s) workload.
# Usage: bash experiments/run_aimd_bursty.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "Running: aimd_bursty"
echo "============================================"
python -m sarathi.benchmark.main \
    --config "experiments/configs/aimd_bursty.yaml"

echo ""
echo "Done. Results in experiments/results/aimd_bursty/"
