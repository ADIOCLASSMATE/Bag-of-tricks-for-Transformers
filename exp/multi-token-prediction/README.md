# Multi-Token Prediction — weighted CE loss over next k tokens

## Method

trick: multi-token-prediction — Uses the same logits to predict multiple future tokens
with a weighted cross-entropy loss, annealed in 3 phases:

```
# Phase 1 (first 1/3): [1.0, 0.5, 0.25→0] — 3-token prediction
# Phase 2 (middle 1/3): [1.0, 0.5→0]       — 2-token prediction
# Phase 3 (final 1/3): [1.0]               — standard next-token
```

The loss for each position i computes:
```
CE_i = -log_softmax(logits[i])[target_i]  # standard next-token
     + w_1 * (-log_softmax(logits[i])[target_{i+1}])  # 2nd token
     + w_2 * (-log_softmax(logits[i])[target_{i+2}])  # 3rd token
```

Sequence tails are masked to avoid out-of-bounds targets. No additional parameters,
no separate prediction heads — the same logits supervise all positions.

## Single-axis experiment (the 2 runs)

| Experiment | `mtp_enabled` | Control |
|---|---|---|
| `multi-token-prediction-fixed_time_10min` | 1 | fixed_compute 600 s |
| `multi-token-prediction-fixed_tokens_10b` | 1 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | Multi-Token Prediction |
|---|---|---|
| Loss function | CE(next token) | Weighted CE(next 1-3 tokens) |
| Annealing schedule | None | 3-phase: 3→2→1 token prediction |
| Parameters | 17.04M | **Identical** (zero parameter increase) |

## Origin

- Source record: modded-nanogpt `2025-12-22_MultiTokenPrediction`
- README: `records/track_1_short/2025-12-22_MultiTokenPrediction/README.md`

## Impact on training

- **Memory.** Negligible — one additional gather + logsumexp per step.
- **Compute.** Minor — extra gather/logsumexp ops offset by better convergence.
- **Convergence.** Multi-token supervision provides richer gradients early, potentially
  accelerating convergence. The annealing schedule ensures final fine-tuning uses
  standard next-token prediction.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | Multi-Token Prediction | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | Multi-Token Prediction | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with multi-token prediction (look for `# trick: multi-token-prediction`)
- `multi-token-prediction.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
python exp/run_experiments.py exp/multi-token-prediction/multi-token-prediction.json --dry-run
python exp/run_experiments.py exp/multi-token-prediction/multi-token-prediction.json
```
