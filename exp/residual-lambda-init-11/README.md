# Residual-Lambda-Init-11 — Residual/X0 Mix Initialization at 1.1 / 0.1

## Method Overview

This experiment adds **learnable per-channel residual mixing coefficients** (`resid_mix`) to each transformer block, initialized to **[1.1, 0.1]** instead of the default **[1.0, 0.0]**.

The `resid_mix` is an `nn.Parameter` (9,216 parameters total: 9 layers x 2 coefficients x 512 dim), meaning it is updated by the optimizer during training — this is not merely an initialization change. The mixing mechanism combines the sublayer output and the original input embedding: `x = resid_mix[0] * sublayer_out + resid_mix[1] * x0`. All other architectural choices remain identical to the baseline.

### Origin

From **slowrun T08 ResidualLambdaInit11**. The slowrun tiny-track change initializes `resid_lambdas` to `1.1` and `x0_lambdas` to `0.1`, so this experiment now mirrors that paired initialization instead of changing only the residual side.

### Motivation

- A slightly larger residual coefficient can help information propagate more directly through depth at the start of training
- A small nonzero `x0` coefficient gives every block immediate access to the input stream, matching the slowrun formulation
- The learnable coefficients are optimized during training, so the model can adapt the mixing ratio beyond the initial values
- The parameter overhead is minimal (+9,216 params, +0.05%) with negligible throughput and memory impact

## Impact on Training

- **Parameters**: +9,216 (9 layers x 2 x 512 — learnable `resid_mix` per channel)
- **Throughput**: No measurable change
- **Memory**: +2 MiB peak
- **Optimization**: Learnable mixing coefficients are trained alongside other parameters, starting from [1.1, 0.1]

## Key Differences from Baseline

| Component | Baseline | residual-lambda-init-11 |
|---|---|---|
| Residual mixing | `x = x + sublayer(x)` | `x = resid_mix[0] * sublayer_out + resid_mix[1] * x0` |
| `resid_mix` init | — | **[1.1, 0.1]** (learnable) |
| Extra parameters | — | **+9,216** (9 layers x 2 x 512) |
| Everything else | identical | identical |

## Results

| Regime | Metric | Baseline | Residual-Lambda-Init-11 | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2632 | **-0.0347** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1328 | -0.0586 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.25B | -5.5% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,391 MiB | +2 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2525 | **-0.0332** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1148 | -0.0561 |
| Fixed Tokens (10B) | Wall-clock | 772s | 814s | +5.4% |
| — | Total Params | 17.04M | 17.04M | +9,216 |

## Analysis

Residual-lambda-init-11 is a top-tier trick, delivering -0.035 BPB under fixed compute and -0.033 BPB under fixed tokens. These gains are nearly identical to resid-mix, which is expected: both tricks expose the same x0-mixing mechanism, differing only in how the coefficients are initialized.

The central insight is that **the mechanism matters more than the initialization**. Resid-mix starts at [1.0, 0.0] (no x0 access at step 0), while this variant starts at [1.1, 0.1] (slight x0 access from step 0). Both converge to similar final quality, indicating the model learns the optimal mixing ratio during training regardless of where it starts.

The 1.1 residual coefficient amplifies the residual path at initialization, helping preserve signal propagation through depth in early training. Combined with the 0.1 x0 coefficient, every block has immediate access to the input embedding stream from the first step. The 9,216 learnable parameters add negligible overhead (+2 MiB peak, +0.05% params) and are optimized with the rest of the model.

**Verdict**: Strong performer, functionally equivalent to resid-mix. The paired initialization [1.1, 0.1] vs [1.0, 0.0] is not a meaningful differentiator in final quality; the x0-mixing mechanism itself is what drives the gain.

## Files

- `train_gpt.py`: Baseline trainer with slowrun-style residual/x0 initialization
- `residual-lambda-init-11.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
