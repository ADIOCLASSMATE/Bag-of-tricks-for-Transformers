#!/usr/bin/env bash
# add experiments for new tricks TRICKS=
    # "partial-key-offset"
    # "sparse-attn-gate"
    # "paired-head-attention"
    # "xsa"
    # "yarn"
    # "multi-token-prediction"
    # "factored-embedding"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/test_all_${TIMESTAMP}.log"

exec &> >(tee -a "$LOG_FILE")

TRICKS=(
    "partial-key-offset"
    "sparse-attn-gate"
    "paired-head-attention"
    "xsa"
    "yarn"
    "multi-token-prediction"
    "factored-embedding"
)

# TRICKS=(
#     "partial-key-offset"
# )

cd "$PROJECT_DIR"

NPROC=${NPROC:-4}
WALLCLOCK_SECONDS=${WALLCLOCK_SECONDS:-1200}

echo "========================================"
echo "=== ${NPROC}GPU Test Suite: ${#TRICKS[@]} tricks ==="
echo "=== Wallclock: ${WALLCLOCK_SECONDS}s ==="
echo "=== Started at $(date) ==="
echo "========================================"
echo ""

PASSED=0
FAILED=0
FAILED_LIST=()

for trick in "${TRICKS[@]}"; do
    echo "========================================"
    echo "=== Testing: ${trick} ==="
    echo "========================================"

    MANIFEST="exp/${trick}/${trick}.json"

    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: Manifest not found: ${MANIFEST}"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$trick")
        continue
    fi

    # Step 1: Dry-run validation
    echo "--- Dry-run validation ---"
    if uv run python exp/run_experiments.py "$MANIFEST" --dry-run --nproc-per-node "$NPROC" --wallclock-seconds "$WALLCLOCK_SECONDS"; then
        echo "--- Dry-run passed ---"
    else
        echo "--- Dry-run FAILED ---"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$trick")
        continue
    fi

    # Step 2: Launch training
    echo "--- Launching training ---"
    if uv run python exp/run_experiments.py "$MANIFEST" --nproc-per-node "$NPROC" --wallclock-seconds "$WALLCLOCK_SECONDS"; then
        echo "--- Training PASSED ---"
        PASSED=$((PASSED + 1))
    else
        echo "--- Training FAILED ---"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$trick")
    fi

    echo ""
done

echo "========================================"
echo "=== Test Suite Complete ==="
echo "=== Passed: ${PASSED}/${#TRICKS[@]} ==="
echo "=== Failed: ${FAILED}/${#TRICKS[@]} ==="
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "=== Failed tricks: ${FAILED_LIST[*]} ==="
fi
echo "=== Finished at $(date) ==="
echo "=== Log: ${LOG_FILE} ==="
echo "========================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
