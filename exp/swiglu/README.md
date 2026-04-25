# SwiGLU — Gated Swish Activation Ablation

## Motivation

The choice of activation function in the feed-forward network (FFN) is a fundamental architectural decision that affects the model's ability to approximate nonlinear transformations. The baseline uses **GELU**, a smooth activation that provides dense, differentiable outputs.

SwiGLU (Gated Swish/SiLU) is the most widely adopted gated activation in modern LLMs, used in LLaMA, Mistral, and other production models. It combines a Swish/SiLU gate with a multiplicative gating mechanism: one projection provides the gate signal while another provides the value, and their element-wise product forms the gated output.

## What This Ablation Tests

This experiment replaces the baseline MLP with a SwiGLU MLP, keeping all other architectural choices identical to the baseline. The hidden dimension remains 2x model_dim for both variants.

| Component | Baseline (GELU) | SwiGLU |
|---|---|---|
| Forward | `down_proj(gelu(fc(x)))` | `down_proj(silu(gate_proj(x)) * up_proj(x))` |
| Projection count | 2 (fc, down_proj) | 3 (gate_proj, up_proj, down_proj) |
| Activation | GELU (smooth, differentiable) | Swish/SiLU (smooth, non-monotonic) |
| Gating | None | Explicit (gate x value) |

Note: SwiGLU uses 3 projection matrices (gate, up, down) vs. the baseline's 2 (fc, down), so the MLP parameter count increases. With `mlp_mult=2` and hidden=2×dim, the baseline has `2 × dim × 2×dim = 4 x dim^2` MLP parameters, while SwiGLU has `3 × dim × 2×dim = 6 x dim^2` -- a 50% increase in MLP parameters.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| MLP activation | GELU | SwiGLU (SiLU gating) |
| MLP projections | fc (dim->2xdim), down_proj | gate_proj, up_proj (dim->2xdim each), down_proj |
| MLP parameter count | 4 x dim^2 | 6 x dim^2 (+50%) |

## Results

| Regime | Metric | Baseline | SwiGLU | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2774 | **-0.0205** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1569 | -0.0345 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 6.80B | -11.3% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 9,908 MiB | +1,519 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2589 | **-0.0268** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1256 | -0.0453 |
| Fixed Tokens (10B) | Wall-clock | 772s | 874s | +13.2% |
| -- | Total Params | 17.04M | 21.76M (+4.72M) | +27.7% |

## Analysis

SwiGLU improves over the GELU baseline by **-0.021 BPB** under fixed compute and **-0.027 BPB** under fixed tokens. The gains are consistent across both evaluation regimes.

**Gating is the primary factor, but the gate nonlinearity matters.** GeGLU outperforms SwiGLU in this ablation suite: -0.024 vs -0.021 FC BPB, -0.040 vs -0.027 FT BPB. Both are gated activations with the same parameter count (3 projections), differing only in gate nonlinearity (GeLU vs SiLU). The gap is most pronounced under fixed tokens, where GeGLU's advantage nearly doubles (0.040 vs 0.027). This suggests the multiplicative gating mechanism is the primary driver, but the gate nonlinearity is not irrelevant — GeLU's smoother gradients and non-monotonic shape appear to be a better gate signal than SiLU at this scale.

**Per-token efficiency improves despite fewer tokens.** Under fixed compute, SwiGLU processes 11.3% fewer tokens (6.80B vs. 7.67B) due to the heavier three-projection forward pass, yet still achieves lower BPB. The architecture learns more per token, offsetting the throughput reduction. Under fixed tokens, the 13.2% wall-clock overhead is proportionally smaller than the 27.7% parameter increase, reflecting the fact that the third projection shares the same hidden dimension rather than expanding it.

**Memory cost is moderate.** Peak usage rises by 1.5 GiB (+18%), from three projection matrices and their activation checkpoints. For the BPB improvement offered, this is a favorable trade-off at this scale.

**Verdict: adopt gating, consider GeGLU instead.** SwiGLU delivers meaningful gains over the non-gated baseline, but GeGLU outperforms it in both evaluation regimes. If choosing a gated activation for this architecture, GeGLU appears to be the better option.

## Files

- `train_gpt.py`: Training script (baseline + SwiGLU modification)
- `swiglu.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
