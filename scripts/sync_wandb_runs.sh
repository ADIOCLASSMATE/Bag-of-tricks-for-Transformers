#!/bin/bash
# Sync offline wandb runs to wandb.ai
# Usage: bash scripts/sync_wandb_runs.sh
# Prerequisites: WANDB_API_KEY must be set in environment

set -euo pipefail

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set."
    echo "Run: export WANDB_API_KEY=<your-key>"
    exit 1
fi

WANDB_BASE="/inspire/hdd/global_user/lipengfei-253107010003/my-proj/bags_tricks/Bag-of-tricks-for-Transformers/wandb"

SYNC_RUNS=(
    # --- leaky-relu-squared ---
    "leaky-relu-squared-fixed_time_20min/wandb/offline-run-20260513_070908-tx7ph9qn"
    "leaky-relu-squared-fixed_tokens_10b/wandb/offline-run-20260513_073243-lnwp527n"
    "leaky-relu-squared-medium-fixed_time_20min/wandb/offline-run-20260513_080309-7mkbxmzt"
    "leaky-relu-squared-medium-fixed_tokens_10b/wandb/offline-run-20260513_082639-ofri6f0b"

    # --- scale-residual ---
    "scale-residual-fixed_time_20min/wandb/offline-run-20260512_215355-9xz2q8s3"
    "scale-residual-fixed_tokens_10b/wandb/offline-run-20260512_221553-yygkbaqr"
    "scale-residual-medium-fixed_time_20min/wandb/offline-run-20260512_224447-eg6pgv16"
    "scale-residual-medium-fixed_tokens_10b/wandb/offline-run-20260512_230730-pyw02vtg"

    # --- unet-skip ---
    "unet-skip-fixed_time_20min/wandb/offline-run-20260513_013523-kl5u8qh1"
    "unet-skip-fixed_tokens_10b/wandb/offline-run-20260513_015723-fc058qyf"
    "unet-skip-medium-fixed_time_20min/wandb/offline-run-20260513_022548-ls9n4wzi"
    "unet-skip-medium-fixed_tokens_10b/wandb/offline-run-20260513_024906-jn3608hi"

    # --- attention-residuals (small only) ---
    "attention-residuals-fixed_time_20min/wandb/offline-run-20260513_152750-5vkwkz6x"
    "attention-residuals-fixed_tokens_10b/wandb/offline-run-20260513_155156-js8povtd"

    # --- geglu (small only) ---
    "geglu-fixed_time_20min/wandb/offline-run-20260513_061104-nrka7ko1"
    "geglu-fixed_tokens_10b/wandb/offline-run-20260513_063459-f5zgaan4"

    # --- swiglu (small only) ---
    "swiglu-fixed_time_20min/wandb/offline-run-20260513_051622-w8z0dvsa"
    "swiglu-fixed_tokens_10b/wandb/offline-run-20260513_053829-m7wbh8j9"
)

total=${#SYNC_RUNS[@]}
success=0
fail=0

echo "============================================"
echo "Syncing $total wandb offline runs"
echo "============================================"
echo

for run in "${SYNC_RUNS[@]}"; do
    full_path="$WANDB_BASE/$run"
    run_name=$(echo "$run" | cut -d'/' -f1)

    if [ ! -d "$full_path" ]; then
        echo "[SKIP] $run_name — directory not found: $full_path"
        fail=$((fail + 1))
        continue
    fi

    echo -n "[${success}✓ / ${fail}✗] $run_name ... "
    if wandb sync "$full_path" 2>&1; then
        echo " done."
        success=$((success + 1))
    else
        echo " FAILED."
        fail=$((fail + 1))
    fi
done

echo
echo "============================================"
echo "Sync complete: $success succeeded, $fail failed (out of $total)"
echo "============================================"
