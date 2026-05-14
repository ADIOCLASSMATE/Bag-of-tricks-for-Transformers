#!/usr/bin/env bash
# Run all 8 refactored experiments (baseline + 7 ablation experiments)
# with both model sizes (small, medium) × both control modes (fixed_compute, fixed_tokens).
#
# Total: 8 experiments × 2 sizes × 2 modes = 32 training runs
#
# Usage:
#   bash scripts/run_all_refactored.sh                    # Run all 32 sequentially
#   bash scripts/run_all_refactored.sh --dry-run           # Print what would run
#   bash scripts/run_all_refactored.sh --experiment resid-mix  # Run only one experiment
#   bash scripts/run_all_refactored.sh --size small        # Run only small models
#   bash scripts/run_all_refactored.sh --mode fixed_compute # Run only fixed_compute
#
# Env vars (optional):
#   GPUS=8                   Number of GPUs (default: 4)
#   SKIP_BASELINE=1          Skip baseline (already verified aligned)
#   WANDB_MODE=offline        Wandb mode (default: offline for this batch)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}"

# --- Defaults ---
GPUS="${GPUS:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-bag-of-tricks-for-transformers}"
WANDB_ENTITY="${WANDB_ENTITY:-Bag-of-Tricks}"
DRY_RUN=false
FILTER_EXPERIMENT=""
FILTER_SIZE=""
FILTER_MODE=""
SKIP_BASELINE="${SKIP_BASELINE:-0}"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --experiment) FILTER_EXPERIMENT="$2"; shift 2 ;;
        --size) FILTER_SIZE="$2"; shift 2 ;;
        --mode) FILTER_MODE="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Experiment list ---
# format: "exp_dir:trainer_file:experiment_name"
EXPERIMENTS=(
    "baseline:train_gpt_refactor.py:baseline"
    "tie-embed:train_gpt_refactor.py:tie-embed"
    "loop-share-mid:train_gpt_refactor.py:loop-share-mid"
    "loop-share-first:train_gpt_refactor.py:loop-share-first"
    "loop-share-all:train_gpt_refactor.py:loop-share-all"
    "kv-sharing-dwide-mlp:train_gpt_refactor.py:kv-sharing-dwide-mlp"
    "resid-mix:train_gpt_refactor.py:resid-mix"
    "residual-lambda-init-11:train_gpt_refactor.py:residual-lambda-init-11"
)

# --- Model sizes ---
# format: "suffix:wandb_group:num_layers:model_dim:num_heads:num_kv_heads:warmdown_iters"
MODEL_SIZES=(
    "small:small-ablation:9:512:8:4:1500"
    "medium:medium-ablation:18:1024:16:8:800"
)

# --- Control modes ---
# format: "mode_suffix:control_mode:wallclock_or_tokens:iterations_cap"
# For fixed_compute: wallclock_seconds
# For fixed_tokens: target_train_tokens
CONTROL_MODES=(
    "fixed_time_20min:fixed_compute:1200:1000000"
    "fixed_tokens_10b:fixed_tokens:10000000000:0"
)

TOTAL_RUNS=0
RUNS_TO_EXECUTE=()

echo "============================================"
echo " Planning refactored experiment runs"
echo "============================================"
echo " GPUs: $GPUS"
echo " Wandb: mode=$WANDB_MODE project=$WANDB_PROJECT"
echo ""

# Build temporary manifest for each run
TEMP_DIR="${PROJECT_ROOT}/.tmp/run_all_refactored"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_dir trainer_file exp_name <<< "$exp_entry"

    if [[ -n "$FILTER_EXPERIMENT" && "$exp_name" != "$FILTER_EXPERIMENT" ]]; then
        continue
    fi
    if [[ "$SKIP_BASELINE" == "1" && "$exp_name" == "baseline" ]]; then
        echo "  Skipping baseline (SKIP_BASELINE=1)"
        continue
    fi

    for size_entry in "${MODEL_SIZES[@]}"; do
        IFS=':' read -r size_suffix wandb_group n_layers model_dim n_heads n_kv_heads warmdown <<< "$size_entry"

        if [[ -n "$FILTER_SIZE" && "$size_suffix" != "$FILTER_SIZE" ]]; then
            continue
        fi

        for ctrl_entry in "${CONTROL_MODES[@]}"; do
            IFS=':' read -r mode_suffix ctrl_mode wallclock_or_tokens iter_cap <<< "$ctrl_entry"

            if [[ -n "$FILTER_MODE" && "$ctrl_mode" != "$FILTER_MODE" ]]; then
                continue
            fi

            run_name="${exp_name}-${size_suffix}-${mode_suffix}"
            trainer_path="exp/${exp_dir}/${trainer_file}"

            if [[ ! -f "$trainer_path" ]]; then
                echo "  WARNING: $trainer_path not found, skipping"
                continue
            fi

            TOTAL_RUNS=$((TOTAL_RUNS + 1))

            # Build control section
            if [[ "$ctrl_mode" == "fixed_compute" ]]; then
                control_json=$(cat <<EOF
{
  "mode": "fixed_compute",
  "target_wallclock_seconds": $wallclock_or_tokens,
  "iterations_cap": $iter_cap
}
EOF
)
            else
                control_json=$(cat <<EOF
{
  "mode": "fixed_tokens",
  "target_train_tokens": $wallclock_or_tokens
}
EOF
)
            fi

            # Build full manifest
            manifest=$(cat <<EOF
{
  "version": 1,
  "trainer_path": "$trainer_path",
  "launcher": {
    "nproc_per_node": $GPUS,
    "master_port_base": 29500
  },
  "defaults": {
    "data_path": "./data/datasets/fineweb10B_sp1024",
    "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
    "vocab_size": 1024,
    "train_batch_tokens": 524288,
    "train_seq_len": 1024,
    "warmup_steps": 50,
    "warmdown_iters": $warmdown,
    "enable_wandb": 1,
    "wandb_mode": "$WANDB_MODE",
    "wandb_project": "$WANDB_PROJECT",
    "wandb_entity": "$WANDB_ENTITY",
    "wandb_group": "$wandb_group",
    "num_layers": $n_layers,
    "model_dim": $model_dim,
    "num_heads": $n_heads,
    "num_kv_heads": $n_kv_heads,
    "mlp_mult": 2,
    "tied_embed_lr": 0.05,
    "embed_lr": 0.6,
    "head_lr": 0.008,
    "matrix_lr": 0.04,
    "scalar_lr": 0.04,
    "muon_momentum": 0.95,
    "muon_backend_steps": 5,
    "muon_momentum_warmup_start": 0.85,
    "muon_momentum_warmup_steps": 800,
    "grad_clip_norm": 0.0
  },
  "experiments": [
    {
      "name": "$run_name",
      "trainer_path": "$trainer_path",
      "control": $control_json
    }
  ]
}
EOF
)
            manifest_file="${TEMP_DIR}/${run_name}.json"
            echo "$manifest" > "$manifest_file"
            RUNS_TO_EXECUTE+=("$manifest_file:$run_name")
        done
    done
done

echo " Total runs planned: $TOTAL_RUNS"
echo " Temp manifests in: $TEMP_DIR"
echo "============================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "--- DRY RUN (command display) ---"
    for entry in "${RUNS_TO_EXECUTE[@]}"; do
        IFS=':' read -r manifest_file run_name <<< "$entry"
        echo ""
        echo "  [$run_name]"
        echo "    manifest: $manifest_file"
        echo "    command:  uv run python exp/run_experiments.py $manifest_file"
    done
    echo ""
    echo "Dry run complete. Remove --dry-run to execute."
    exit 0
fi

# --- Execute ---
echo "Starting training runs..."
echo ""

FAILED_RUNS=()
for entry in "${RUNS_TO_EXECUTE[@]}"; do
    IFS=':' read -r manifest_file run_name <<< "$entry"

    echo "============================================"
    echo " RUN: $run_name"
    echo "============================================"
    echo "  manifest: $manifest_file"
    echo "  started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    if uv run python exp/run_experiments.py "$manifest_file"; then
        echo "  [$run_name] SUCCEEDED"
    else
        echo "  [$run_name] FAILED (exit code: $?)"
        FAILED_RUNS+=("$run_name")
    fi
    echo ""
done

# --- Summary ---
echo "============================================"
echo " SUMMARY"
echo "============================================"
echo " Total:  $TOTAL_RUNS"
echo " Failed: ${#FAILED_RUNS[@]}"
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo " Failed runs:"
    for name in "${FAILED_RUNS[@]}"; do
        echo "   - $name"
    done
    exit 1
else
    echo " All runs succeeded!"
fi

# Cleanup
rm -rf "$TEMP_DIR"
