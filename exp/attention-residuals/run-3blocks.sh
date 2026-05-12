#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

echo "=== Running attention-residuals experiments (3 blocks, 2 controls) ==="
uv run python exp/run_experiments.py exp/attention-residuals/attention-residuals-3blocks.json
echo "=== Experiments complete ==="

# Sync offline wandb runs
echo "=== Syncing wandb runs ==="
for run_dir in "$PROJECT_DIR"/wandb/offline-run-*; do
    if [ -d "$run_dir" ]; then
        echo "Syncing $run_dir ..."
        timeout 120 uv run wandb sync "$run_dir" || echo "Warning: failed to sync $run_dir"
    fi
done
echo "=== Done ==="
