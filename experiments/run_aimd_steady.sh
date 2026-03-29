#!/usr/bin/env bash
# Run AIMD chunk-size controller with steady (Poisson 2 req/s) workload.
# Usage: bash experiments/run_aimd_steady.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "Running: aimd_steady"
echo "============================================"
python -m sarathi.benchmark.main \
    --config "experiments/configs/aimd_steady.yaml"

echo ""
echo "Done. Results in experiments/results/aimd_steady/"
