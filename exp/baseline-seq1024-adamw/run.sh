#!/bin/bash
set -e


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Running baseline-seq1024-adamw experiments (2 runs) ==="
uv run python exp/run_experiments.py exp/baseline-seq1024-adamw/baseline-seq1024-adamw-4gpu.json
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
