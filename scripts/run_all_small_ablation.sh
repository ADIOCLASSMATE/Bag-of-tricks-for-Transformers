#!/usr/bin/env bash
# One-click launch for ALL small-model (9L/512d/8h) ablation experiments.
# After this batch completes, we proceed to larger-model ablation.
#
# Usage:
#   ./scripts/run_all_small_ablation.sh              # launch all
#   ./scripts/run_all_small_ablation.sh --dry-run    # validate only
#   NPROC=8 ./scripts/run_all_small_ablation.sh      # override GPU count

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/small_ablation_${TIMESTAMP}.log"

exec &> >(tee -a "$LOG_FILE")

# Each entry is "display_name|manifest_path"
# attention-residuals has two variants — both must run.
EXPERIMENTS=(
    "baseline|exp/baseline/baseline.json"
    "partial-key-offset|exp/partial-key-offset/partial-key-offset.json"
    "sparse-attn-gate|exp/sparse-attn-gate/sparse-attn-gate.json"
    "paired-head-attention|exp/paired-head-attention/paired-head-attention.json"
    "xsa|exp/xsa/xsa.json"
    "yarn|exp/yarn/yarn.json"
    "factored-embedding|exp/factored-embedding/factored-embedding.json"
    "untie-embed|exp/untie-embed/untie-embed.json"
    "loop-share-mid|exp/loop-share-mid/loop-share-mid.json"
    "loop-share-first|exp/loop-share-first/loop-share-first.json"
    "loop-share-all|exp/loop-share-all/loop-share-all.json"
    "kv-sharing-dwide-mlp|exp/kv-sharing-dwide-mlp/kv-sharing-dwide-mlp.json"
    "resid-mix|exp/resid-mix/resid-mix.json"
    "residual-lambda-init-11|exp/residual-lambda-init-11/residual-lambda-init-11.json"
    "sandwich-norm|exp/sandwich-norm/sandwich-norm.json"
    "scale-residual|exp/scale-residual/scale-residual.json"
    "unet-skip|exp/unet-skip/unet-skip.json"
    "attention-residuals (9 blocks)|exp/attention-residuals/attention-residuals.json"
    "attention-residuals (3 blocks)|exp/attention-residuals/attention-residuals-4gpu.json"
    "swiglu|exp/swiglu/swiglu.json"
    "geglu|exp/geglu/geglu.json"
    "leaky-relu-squared|exp/leaky-relu-squared/leaky-relu-squared.json"
    "relu-sq|exp/relu-sq/relu-sq.json"
    "logit-softcap|exp/logit-softcap/logit-softcap.json"
    "q-gain|exp/q-gain/q-gain.json"
    "attention-head-gating|exp/attention-head-gating/attention-head-gating.json"
    "drop-attn|exp/drop-attn/drop-attn.json"
    "per-layer-residual-input|exp/per-layer-residual-input/per-layer-residual-input.json"
    "smear-gate|exp/smear-gate/smear-gate.json"
    "engram-core|exp/engram-core/engram-core.json"
    "engram-compressed|exp/engram-compressed/engram-compressed.json"
)

cd "$PROJECT_DIR"

NPROC=${NPROC:-4}
WALLCLOCK_SECONDS=${WALLCLOCK_SECONDS:-1200}
DRY_RUN=""

# wandb grouping — all small-model runs in one group, tag for filtering
export WANDB_GROUP="${WANDB_GROUP:-small-ablation}"
export WANDB_TAGS="${WANDB_TAGS:-small-model,9L-512d}"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "=============================================="
echo "=== Small-Model Ablation Suite ==="
echo "=== Model: 9L / 512d / 8 heads ==="
echo "=== GPUs: ${NPROC} | Wallclock: ${WALLCLOCK_SECONDS}s ==="
echo "=== Experiments: ${#EXPERIMENTS[@]} ==="
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
    if uv run python exp/run_experiments.py "$MANIFEST" --dry-run --nproc-per-node "$NPROC" --wallclock-seconds "$WALLCLOCK_SECONDS"; then
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
    if uv run python exp/run_experiments.py "$MANIFEST" --nproc-per-node "$NPROC" --wallclock-seconds "$WALLCLOCK_SECONDS"; then
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
echo "=== Small-Model Ablation Complete ==="
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
