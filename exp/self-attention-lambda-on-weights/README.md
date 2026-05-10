# SA Lambda on V Weights — per-layer learnable scalar on V projection

## Method Overview

trick: self-attention-lambda-on-weights — Add a per-layer learnable scalar $\lambda$ that
scales the value projection weight $W_v$ before the linear transform:

```
V = x @ (5 · λ · W_v).T
```

The factor-5 multiplier provides ~5× effective learning rate (gradient scaling) while
keeping the effective initial value at `5 × 0.2 = 1.0`, so training starts identically to
the baseline. The scalar is initialized to `0.2` and stored in fp32.

Only V is modulated; Q and K are unchanged since QK-Norm absorbs any scalar scaling.
$\lambda$ is placed in the Adam optimizer group (scalar group) along with other
low-dimensional parameters.

## Single-axis experiment (the 2 runs)

| Experiment | `sa_lambda` init | Control |
|---|---|---|
| `self-attention-lambda-on-weights-fixed_time_10min` | 0.2 (per layer) | fixed_compute 600 s |
| `self-attention-lambda-on-weights-fixed_tokens_10b` | 0.2 (per layer) | fixed_tokens 10 B |

### Origin

- Source record: modded-nanogpt `2025-12-10_SALambdaOnWeight`
- The original approach multiplied $\lambda$ on the merged QKV weight matrix.
  This version applies it only to $W_v$ since Q and K are immediately RMS-normed,
  avoiding unnecessary computation on Q/K.

## Impact on training

- **Memory.** Negligible — 9 additional scalar parameters per model.
- **Compute.** Negligible — one scalar-vector multiply per layer's V projection weight.
- **Convergence.** The learnable scalar allows the model to adjust the magnitude of
  the V projection independently per layer. This is complementary to the existing
  `lambdas` (block-level skip weights) and provides finer control over value
  representations.

## Key differences from baseline

| Parameter | Baseline | SA Lambda on V Weights |
|---|---|---|
| V projection | `v = x @ W_v.T` | `v = x @ (5 · λ · W_v).T` |
| New parameters | 0 | +`num_layers` scalars (e.g., +9) |
| λ init | — | 0.2 → effective init 1.0 |
| λ optimizer | — | Adam (scalar group) |
| λ lr scale | — | 5× (built into forward multiplier) |
| Parameters | 17.04M | ~17.04M (+9, negligible) |

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | SA Lambda on V Weights | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | SA Lambda on V Weights | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with SA lambda on V weights (look for `# trick: self-attention-lambda-on-weights`)
- `draft.py` — the original `2025-12-10_SALambdaOnWeight` code
- `self-attention-lambda-on-weights.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
# Dry-run the manifest
python exp/run_experiments.py exp/self-attention-lambda-on-weights/self-attention-lambda-on-weights.json --dry-run

# Launch both experiments
python exp/run_experiments.py exp/self-attention-lambda-on-weights/self-attention-lambda-on-weights.json
```
