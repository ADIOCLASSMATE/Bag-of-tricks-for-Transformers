#!/usr/bin/env bash
# One-click launch for ALL medium-model (18L/1024d/16h, ~133M) ablation experiments.
# Model is ~7.8x larger than small (9L/512d/8h, ~17M).
# Estimated: fixed_tokens_10b takes ~124 min/experiment, ~62 hours total for 31 experiments.
#
# Usage:
#   ./scripts/run_all_medium_ablation.sh              # launch all
#   ./scripts/run_all_medium_ablation.sh --dry-run    # validate only
#   NPROC=8 ./scripts/run_all_medium_ablation.sh      # override GPU count

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/medium_ablation_${TIMESTAMP}.log"

exec &> >(tee -a "$LOG_FILE")

# Each entry is "display_name|manifest_path"
# attention-residuals has two variants — both must run.
EXPERIMENTS=(
    "baseline|exp/baseline/baseline-medium.json"
    "partial-key-offset|exp/partial-key-offset/partial-key-offset-medium.json"
    "sparse-attn-gate|exp/sparse-attn-gate/sparse-attn-gate-medium.json"
    "paired-head-attention|exp/paired-head-attention/paired-head-attention-medium.json"
    "xsa|exp/xsa/xsa-medium.json"
    "yarn|exp/yarn/yarn-medium.json"
    "factored-embedding|exp/factored-embedding/factored-embedding-medium.json"
    "untie-embed|exp/untie-embed/untie-embed-medium.json"
    "loop-share-mid|exp/loop-share-mid/loop-share-mid-medium.json"
    "loop-share-first|exp/loop-share-first/loop-share-first-medium.json"
    "loop-share-all|exp/loop-share-all/loop-share-all-medium.json"
    "kv-sharing-dwide-mlp|exp/kv-sharing-dwide-mlp/kv-sharing-dwide-mlp-medium.json"
    "resid-mix|exp/resid-mix/resid-mix-medium.json"
    "residual-lambda-init-11|exp/residual-lambda-init-11/residual-lambda-init-11-medium.json"
    "sandwich-norm|exp/sandwich-norm/sandwich-norm-medium.json"
    "scale-residual|exp/scale-residual/scale-residual-medium.json"
    "unet-skip|exp/unet-skip/unet-skip-medium.json"
    "attention-residuals (9 blocks)|exp/attention-residuals/attention-residuals-medium.json"
    "attention-residuals (3 blocks)|exp/attention-residuals/attention-residuals-4gpu-medium.json"
    "swiglu|exp/swiglu/swiglu-medium.json"
    "geglu|exp/geglu/geglu-medium.json"
    "leaky-relu-squared|exp/leaky-relu-squared/leaky-relu-squared-medium.json"
    "relu-sq|exp/relu-sq/relu-sq-medium.json"
    "logit-softcap|exp/logit-softcap/logit-softcap-medium.json"
    "q-gain|exp/q-gain/q-gain-medium.json"
    "attention-head-gating|exp/attention-head-gating/attention-head-gating-medium.json"
    "drop-attn|exp/drop-attn/drop-attn-medium.json"
    "per-layer-residual-input|exp/per-layer-residual-input/per-layer-residual-input-medium.json"
    "smear-gate|exp/smear-gate/smear-gate-medium.json"
    "engram-core|exp/engram-core/engram-core-medium.json"
    "engram-compressed|exp/engram-compressed/engram-compressed-medium.json"
)

cd "$PROJECT_DIR"

NPROC=${NPROC:-4}
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# wandb grouping — all medium-model runs in one group, tag for filtering
export WANDB_GROUP="${WANDB_GROUP:-medium-ablation}"
export WANDB_TAGS="${WANDB_TAGS:-medium-model,18L-1024d}"

echo "=============================================="
echo "=== Medium-Model Ablation Suite ==="
echo "=== Model: 18L / 1024d / 16 heads / 8 kv_heads ==="
echo "=== Params: ~133M (7.8x small) ==="
echo "=== GPUs: ${NPROC} ==="
echo "=== Experiments: ${#EXPERIMENTS[@]} ==="
echo "=== Est. total time: ~62 hours (124 min/exp for fixed_tokens_10b) ==="
if [ -n "$DRY_RUN" ]; then
    echo "=== MODE: DRY-RUN (validation only) ==="
fi
echo "=== Started at $(date) ==="
echo "=============================================="
echo ""

PASSED=0
FAILED=0
FAILED_LIST=()

for entry in "${EXPERIMENTS[@]}"; do
    DISPLAY_NAME="${entry%%|*}"
    MANIFEST="${entry##*|}"

    echo "=============================================="
    echo "=== [${DISPLAY_NAME}] ==="
    echo "=== Manifest: ${MANIFEST} ==="
    echo "=============================================="

    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: Manifest not found: ${MANIFEST}"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$DISPLAY_NAME")
        continue
    fi

    # Step 1: Dry-run validation
    echo "--- Dry-run validation ---"
    if uv run python exp/run_experiments.py "$MANIFEST" --dry-run --nproc-per-node "$NPROC"; then
        echo "--- Dry-run passed ---"
    else
        echo "--- Dry-run FAILED ---"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$DISPLAY_NAME")
        continue
    fi

    if [ -n "$DRY_RUN" ]; then
        PASSED=$((PASSED + 1))
        echo ""
        continue
    fi

    # Step 2: Launch training
    echo "--- Launching training ---"
    if uv run python exp/run_experiments.py "$MANIFEST" --nproc-per-node "$NPROC"; then
        echo "--- PASSED ---"
        PASSED=$((PASSED + 1))
    else
        echo "--- FAILED ---"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$DISPLAY_NAME")
    fi

    echo ""
done

echo "=============================================="
echo "=== Medium-Model Ablation Complete ==="
echo "=== Passed: ${PASSED}/${#EXPERIMENTS[@]} ==="
echo "=== Failed: ${FAILED}/${#EXPERIMENTS[@]} ==="
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "=== Failed: ${FAILED_LIST[*]} ==="
fi
echo "=== Finished at $(date) ==="
echo "=== Log: ${LOG_FILE} ==="
echo "=============================================="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
