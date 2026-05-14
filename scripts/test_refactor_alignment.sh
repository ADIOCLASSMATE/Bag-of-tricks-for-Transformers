#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full training alignment test: original baseline vs refactored baseline.
#
# Runs both trainers with the SAME configs as production ablation:
#   - fixed_compute  20min (1200s wallclock, warmdown 1500)
#   - fixed_tokens   10B  (10,000,000,000 tokens, warmdown 1500)
#
# Each mode runs BOTH trainers with identical seed/env, then diffs result.json.
#
# Usage:
#   bash scripts/test_refactor_alignment.sh                              # both modes
#   bash scripts/test_refactor_alignment.sh --mode fixed_compute         # 20min only
#   bash scripts/test_refactor_alignment.sh --mode fixed_tokens          # 10B only
#   bash scripts/test_refactor_alignment.sh --quick-only                 # model tests only
#   bash scripts/test_refactor_alignment.sh --skip-train                 # diff existing results
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT_ROOT"

# ---- tunables ----
GPUS=${GPUS:-4}           # match baseline.json nproc_per_node
SEED=${SEED:-1337}
LOG_BASE="${PROJECT_ROOT}/exp/baseline/logs/alignment-test"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
info()  { echo -e "${YELLOW}[INFO]${NC} $*"; }

# ---- env: shared across all runs ---------------------------------------
_base_env() {
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export DATA_PATH="./data/datasets/fineweb10B_sp1024"
    export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
    export VOCAB_SIZE=1024
    export SEED="${SEED}"
    export WARMUP_STEPS=50
    export ENABLE_WANDB=0
    export WANDB_MODE="disabled"
    export NUM_LAYERS=9
    export MODEL_DIM=512
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=2
    export TIE_EMBEDDINGS=0
    export TRAIN_BATCH_TOKENS=524288
    export TRAIN_SEQ_LEN=1024
    export GRAD_ACCUM_STEPS=1
    export EMBED_LR=0.6
    export HEAD_LR=0.008
    export MATRIX_LR=0.04
    export SCALAR_LR=0.04
    export MUON_MOMENTUM=0.95
    export MUON_BACKEND_STEPS=5
    export MUON_MOMENTUM_WARMUP_START=0.85
    export MUON_MOMENTUM_WARMUP_STEPS=800
    export BETA1=0.9
    export BETA2=0.95
    export WEIGHT_DECAY=0.1
    export RUN_ID="alignment-test"
    export EXPERIMENT_NAME="alignment-test"
}

# ---- env: fixed_compute 20min -------------------------------------------
env_fixed_compute() {
    local outdir="$1"
    _base_env
    export CONTROL_MODE="fixed_compute"
    export MAX_WALLCLOCK_SECONDS=1200.0
    export ITERATIONS=1000000           # effectively unlimited; wallclock decides stop
    export WARMDOWN_ITERS=1500
    export OUTPUT_DIR="${outdir}"
}

# ---- env: fixed_tokens 10B ----------------------------------------------
env_fixed_tokens() {
    local outdir="$1"
    _base_env
    export CONTROL_MODE="fixed_tokens"
    export TARGET_TRAIN_TOKENS=10000000000
    # iterations = ceil(10B / 524288) = 19074
    export ITERATIONS=19074
    export WARMDOWN_ITERS=1500
    export MAX_WALLCLOCK_SECONDS=0.0
    export OUTPUT_DIR="${outdir}"
}

# ---- run one trainer ----------------------------------------------------
run_trainer() {
    local trainer="$1"
    local outdir="$2"

    info "Running: ${trainer}"
    mkdir -p "${outdir}/log"

    if [[ "${GPUS}" -gt 1 ]]; then
        uv run torchrun --standalone --nproc_per_node="${GPUS}" --master_port=29599 \
            "${trainer}"
    else
        uv run python "${trainer}"
    fi
    info "Finished (exit=$?) → ${outdir}/result.json"
}

# ---- diff two result.jsons ---------------------------------------------
diff_results() {
    local mode="$1"
    local left_label="$2"  left_json="$3"
    local right_label="$4" right_json="$5"

    if [[ ! -f "${left_json}" ]]; then
        fail "${left_label} result.json not found: ${left_json}"
        return 1
    fi
    if [[ ! -f "${right_json}" ]]; then
        fail "${right_label} result.json not found: ${right_json}"
        return 1
    fi

    info "Comparing ${left_label} vs ${right_label}  (mode=${mode})"

    local left_params right_params left_loss right_loss left_bpb right_bpb \
          left_step right_step left_tokens right_tokens left_time right_time

    read left_params left_loss left_bpb left_step left_tokens left_time < <(python3 -c "
import json
d=json.load(open('${left_json}'))
print(d['model']['model_params'],
      d['metrics']['final_val_loss'],
      d['metrics']['final_val_bpb'],
      d['training']['final_step'],
      d['control']['actual_train_tokens'],
      d['training']['training_time_ms'])")

    read right_params right_loss right_bpb right_step right_tokens right_time < <(python3 -c "
import json
d=json.load(open('${right_json}'))
print(d['model']['model_params'],
      d['metrics']['final_val_loss'],
      d['metrics']['final_val_bpb'],
      d['training']['final_step'],
      d['control']['actual_train_tokens'],
      d['training']['training_time_ms'])")

    local all_ok=0

    echo ""
    echo "  ----------------------------------------------------------------------"
    printf "  %-25s %-22s %-22s\n" "metric" "${left_label}" "${right_label}"
    echo "  ----------------------------------------------------------------------"
    printf "  %-25s %-22s %-22s\n" "model_params" "${left_params}" "${right_params}"
    printf "  %-25s %-22s %-22s\n" "final_step" "${left_step}" "${right_step}"
    printf "  %-25s %-22s %-22s\n" "actual_train_tokens" "${left_tokens}" "${right_tokens}"
    printf "  %-25s %-22s %-22s\n" "training_time_ms" "${left_time}" "${right_time}"
    printf "  %-25s %-22s %-22s\n" "final_val_loss" "${left_loss}" "${right_loss}"
    printf "  %-25s %-22s %-22s\n" "final_val_bpb" "${left_bpb}" "${right_bpb}"
    echo "  ----------------------------------------------------------------------"

    if [[ "${left_params}" != "${right_params}" ]]; then
        fail "model_params differ: ${left_params} vs ${right_params}"
        all_ok=1
    else
        pass "model_params match: ${left_params}"
    fi

    # fixed_tokens: deterministic step count → tight tolerance.
    # fixed_compute: wallclock timing noise → steps may differ ±0.5%, loss tol relaxed.
    if [[ "${mode}" == "fixed_compute" ]]; then
        local loss_tol=0.001      # 0.1% for wallclock-bound runs
        local step_tol=0.01       # 1% step-count drift is timing noise
    else
        local loss_tol=0.0002     # 0.02% for deterministic-step runs
        local step_tol=0          # steps must match exactly
    fi

    python3 -c "
left=${left_loss}
right=${right_loss}
tol=${loss_tol}
diff_frac = abs(left - right) / max(abs(left), abs(right), 1e-9)
if left == right:
    print('[PASS] val_loss exact match')
elif diff_frac < tol:
    print(f'[PASS] val_loss within {tol*100:.2f}%: {left:.8f} vs {right:.8f} (diff={diff_frac:.6%})')
else:
    print(f'[FAIL] val_loss differs: {left:.8f} vs {right:.8f} (diff={diff_frac:.6%} > {tol*100:.2f}%)')
    exit(1)
" || all_ok=1

    python3 -c "
left=${left_bpb}
right=${right_bpb}
tol=${loss_tol}
diff_frac = abs(left - right) / max(abs(left), abs(right), 1e-9)
if left == right:
    print('[PASS] val_bpb exact match')
elif diff_frac < tol:
    print(f'[PASS] val_bpb within {tol*100:.2f}%: {left:.8f} vs {right:.8f} (diff={diff_frac:.6%})')
else:
    print(f'[FAIL] val_bpb differs: {left:.8f} vs {right:.8f} (diff={diff_frac:.6%} > {tol*100:.2f}%)')
    exit(1)
" || all_ok=1

    local step_diff=$(( left_step > right_step ? left_step - right_step : right_step - left_step ))
    local step_max=$(( left_step > right_step ? left_step : right_step ))
    if [[ "${step_tol}" == "0" ]]; then
        if [[ "${left_step}" != "${right_step}" ]]; then
            fail "final_step differs: ${left_step} vs ${right_step} — must match exactly for ${mode}"
            all_ok=1
        else
            pass "final_step match: ${left_step}"
        fi
    else
        local step_diff_frac
        step_diff_frac=$(python3 -c "print(${step_diff} / ${step_max})")
        if python3 -c "exit(0 if ${step_diff} / ${step_max} < ${step_tol} else 1)"; then
            pass "final_step within tolerance: ${left_step} vs ${right_step} (diff=${step_diff}, ${step_diff_frac} < ${step_tol})"
        else
            fail "final_step drift too large: ${left_step} vs ${right_step} (diff=${step_diff}, ${step_diff_frac} >= ${step_tol})"
            all_ok=1
        fi
    fi

    return "${all_ok}"
}

# ---- run one full mode (original + refactored, then diff) ---------------
run_mode() {
    local mode="$1"
    local mode_label="$2"
    local orig_out="${LOG_BASE}/${mode_label}/original"
    local refac_out="${LOG_BASE}/${mode_label}/refactored"

    echo ""
    echo "============================================================"
    echo " Mode: ${mode_label}"
    echo "============================================================"

    # Original
    case "${mode}" in
        fixed_compute) env_fixed_compute "${orig_out}" ;;
        fixed_tokens)  env_fixed_tokens "${orig_out}" ;;
    esac
    run_trainer "${PROJECT_ROOT}/exp/baseline/train_gpt.py" "${orig_out}"

    # Refactored
    case "${mode}" in
        fixed_compute) env_fixed_compute "${refac_out}" ;;
        fixed_tokens)  env_fixed_tokens "${refac_out}" ;;
    esac
    run_trainer "${PROJECT_ROOT}/exp/baseline/train_gpt_refactor.py" "${refac_out}"

    echo ""
    echo "============================================================"
    echo " Diff: ${mode_label}"
    echo "============================================================"
    diff_results "${mode}" "original" "${orig_out}/result.json" "refactored" "${refac_out}/result.json" \
        && pass "${mode_label}: ALIGNMENT VERIFIED" \
        || { fail "${mode_label}: ALIGNMENT CHECK FAILED"; return 1; }
}

# ---- main ---------------------------------------------------------------
main() {
    local skip_train=0
    local quick_only=0
    local mode=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-train) skip_train=1 ;;
            --quick-only) quick_only=1 ;;
            --mode) mode="$2"; shift ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
        shift
    done

    # --- Phase 1: Quick model test ---
    echo ""
    echo "============================================================"
    echo " Phase 1: Quick model-level tests (no data needed)"
    echo "============================================================"
    uv run python exp/baseline/test_refactor_alignment.py || {
        fail "Quick tests FAILED — fix issues before full training"
        exit 1
    }

    if [[ "${quick_only}" -eq 1 ]]; then
        echo ""
        pass "Quick tests passed. Skipping full training (--quick-only)."
        exit 0
    fi

    if [[ "${skip_train}" -eq 1 ]]; then
        info "Skipping training (--skip-train). Diffing existing results..."

        if [[ -z "${mode}" || "${mode}" == "fixed_compute" ]]; then
            diff_results "fixed_compute" "original" "${LOG_BASE}/fixed_compute/original/result.json" \
                         "refactored" "${LOG_BASE}/fixed_compute/refactored/result.json" || true
        fi
        if [[ -z "${mode}" || "${mode}" == "fixed_tokens" ]]; then
            diff_results "fixed_tokens" "original" "${LOG_BASE}/fixed_tokens/original/result.json" \
                         "refactored" "${LOG_BASE}/fixed_tokens/refactored/result.json" || true
        fi
        exit 0
    fi

    # --- Phase 2+3: Full training per mode ---
    local overall_ok=0

    if [[ -z "${mode}" || "${mode}" == "fixed_compute" ]]; then
        run_mode "fixed_compute" "fixed_compute" || overall_ok=1
    fi

    if [[ -z "${mode}" || "${mode}" == "fixed_tokens" ]]; then
        run_mode "fixed_tokens" "fixed_tokens" || overall_ok=1
    fi

    echo ""
    if [[ "${overall_ok}" -eq 0 ]]; then
        pass "ALL ALIGNMENT TESTS PASSED"
    else
        fail "SOME ALIGNMENT CHECKS FAILED — see above"
        exit 1
    fi
}

main "$@"
