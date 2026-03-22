#!/usr/bin/env bash
# Run all 6 benchmark conditions: 3 controllers x 2 traces
# Must be run from the sarathi-serve root directory on a GPU machine.
# Usage: bash experiments/run_experiments.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
    "experiments/configs/static_steady.yaml"
    "experiments/configs/static_bursty.yaml"
    "experiments/configs/aimd_steady.yaml"
    "experiments/configs/aimd_bursty.yaml"
    "experiments/configs/pid_steady.yaml"
    "experiments/configs/pid_bursty.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================"
    echo "Running: $config"
    echo "============================================"
    python -m sarathi.benchmark.main \
        --config "$config"
done

echo ""
echo "All experiments complete. Results in experiments/results/"
