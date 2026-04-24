# ReLU^2 — Squared ReLU Activation Ablation

## Method Overview

This experiment replaces the baseline's GELU activation with **ReLU^2** (squared ReLU) in the MLP. The forward pass becomes `proj(relu(fc(x)).square())` instead of `proj(gelu(fc(x)))`.

### Motivation

- ReLU^2 is a cheaper alternative to GELU with competitive performance, popular in the modded-nanogpt speedrun community
- The squared ReLU provides sparse, non-negative gating with quadratic growth for positive activations
- Unlike GELU, ReLU has a hard threshold at zero, creating exact sparsity that can act as an implicit regularizer

## What This Ablation Tests

| Component | Baseline | ReLU^2 |
|---|---|---|
| MLP activation | GELU | `relu(x).square()` |
| Sparsity | Dense (smooth activation) | Exact zeros for negative inputs |
| Gradient | Smooth everywhere | Discontinuous at zero |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| MLP activation | GELU | ReLU^2 |
| Code change | — | MLP.forward: `relu(fc(x)).square()` instead of `gelu(fc(x))` |

## Results

| Regime | Metric | Baseline | ReLU^2 | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2902 | **-0.0077** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1785 | -0.0129 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.65B | -0.3% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Compute (10 min) | Wall-clock | 600s | 600s | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2793 | **-0.0064** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1600 | -0.0109 |
| Fixed Tokens (10B) | Wall-clock | 772s | 771s | -0.1% |
| — | Total Params | 17.04M | 17.04M | 0 |

## Analysis

ReLU^2 improves over GELU by **0.006-0.008 BPB** in both evaluation regimes with no increase in parameters or memory.

The mechanism is twofold. First, ReLU introduces exact sparsity by zeroing negative pre-activations, which acts as an implicit regularizer -- GELU's smooth taper preserves small negative contributions that add noise. Second, the squaring operation amplifies large positive activations quadratically while suppressing values near zero, creating a high-contrast representation that concentrates signal into fewer active units. The combination yields sparser, more informative MLP outputs for the downstream residual path.

Computationally, ReLU^2 is strictly cheaper than GELU per step (hard threshold + element-wise square vs. erf-based approximation), but this does not translate into a wall-clock advantage in the fixed-compute regime where both models train for the same 600s time budget. Instead, ReLU^2 achieves lower validation BPB (1.2902 vs 1.2979) within that identical time budget -- a quality improvement, not a speed improvement. The two models process nearly the same volume of tokens (7.65B vs 7.67B, -0.3%) because both hit the 600s limit. The per-step speedup from ReLU^2's cheaper activation is negligible relative to the full training step.

ReLU^2 is a drop-in replacement that trades GELU's smooth gradients for exact sparsity and quadratic gain. It achieves better BPB than GELU within the same compute budget at no extra parameter or memory cost, though gated activations (GeGLU, SwiGLU) remain stronger at the expense of additional parameters.

## Files

- `train_gpt.py`: Training script (baseline + ReLU^2 modification)
- `relu-sq.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
