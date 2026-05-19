#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/test_all_${TIMESTAMP}.log"

exec &> >(tee -a "$LOG_FILE")

##
#############################
## 给定exp id
TRICKS=(
    "baseline"
    "tie-embed"
    "loop-share-mid"
    "loop-share-first"
    "loop-share-all"
    "kv-sharing-dwide-mlp"
    "kv-sharing-only"
    "dwide-mlp-only"
    "resid-mix"
    "resid-mix-init11"
    "resid-mix-tie-embedding"
)

cd "$PROJECT_DIR"

NPROC=${NPROC:-4}
WALLCLOCK_SECONDS=${WALLCLOCK_SECONDS:-1200}
TOTAL_RUNS=$(( ${#TRICKS[@]} * 2 ))

echo "========================================"
echo "=== ${NPROC}GPU Test Suite: ${#TRICKS[@]} tricks (${TOTAL_RUNS} manifests) ==="
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

    ##
    ####################
    ## 给定每个实验的配置文件--json
    MANIFESTS=("exp/${trick}/${trick}.json" "exp/${trick}/${trick}-medium.json")

    for MANIFEST in "${MANIFESTS[@]}"; do
        echo "--- Manifest: ${MANIFEST} ---"

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
done

echo "========================================"
echo "=== Test Suite Complete ==="
echo "=== Passed: ${PASSED}/${TOTAL_RUNS} ==="
echo "=== Failed: ${FAILED}/${TOTAL_RUNS} ==="
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    # dedupe (a trick may fail twice — once per manifest)
    UNIQUE_FAILED=$(printf '%s\n' "${FAILED_LIST[@]}" | sort -u | tr '\n' ' ')
    echo "=== Failed manifests: ${FAILED_LIST[*]} ==="
    echo "=== Failed tricks (unique): ${UNIQUE_FAILED}==="
fi
echo "=== Finished at $(date) ==="
echo "=== Log: ${LOG_FILE} ==="
echo "========================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
