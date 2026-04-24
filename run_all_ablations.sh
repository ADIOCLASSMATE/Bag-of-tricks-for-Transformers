#!/usr/bin/env bash
set -euo pipefail

cd /inspire/hdd/global_user/wanjiaxin-253108030048/code/bag-of-tricks-transformers
source .venv/bin/activate
export WANDB_MODE=offline

# Auto-discover all experiment manifests under exp/
mapfile -t EXPERIMENTS < <(find exp -maxdepth 2 -name "*.json" ! -path "*/logs/*" | sort)

TOTAL=${#EXPERIMENTS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

echo "=== Ablation Experiment Suite ==="
echo "Total: $TOTAL experiments"
echo "Start: $(date)"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM=$((i + 1))
    echo "=== [$NUM/$TOTAL] $EXP ==="
    echo "Start: $(date)"

    if python exp/run_experiments.py "$EXP"; then
        PASSED=$((PASSED + 1))
        echo "[$NUM/$TOTAL] PASSED: $EXP"
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$EXP")
        echo "[$NUM/$TOTAL] FAILED: $EXP"
    fi

    echo "End: $(date)"
    echo ""
done

echo "=== Summary ==="
echo "Total: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "Failed experiments:"
    for f in "${FAILED_LIST[@]}"; do
        echo "  - $f"
    done
fi
echo "End time: $(date)"
