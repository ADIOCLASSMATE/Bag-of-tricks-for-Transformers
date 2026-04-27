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

Note: SwiGLU uses 3 projection matrices (gate, up, down) vs. the baseline's 2 (fc, down), so the MLP parameter count increases. With `mlp_mult=2` and hidden=2xdim, the baseline has `2 x dim x 2xdim = 4 x dim^2` MLP parameters, while SwiGLU has `3 x dim x 2xdim = 6 x dim^2` -- a 50% increase in MLP parameters.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| MLP activation | GELU | SwiGLU (SiLU gating) |
| MLP projections | fc (dim->2xdim), down_proj | gate_proj, up_proj (dim->2xdim each), down_proj |
| MLP parameter count | 4 x dim^2 | 6 x dim^2 (+50%) |

## Results

| Regime | Metric | Baseline | SwiGLU | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2622 | **-0.0316** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1312 | -0.0533 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 6.78B | -11.1% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 9,908 MiB | +1,519 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2506 | **-0.0341** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1116 | -0.0576 |
| Fixed Tokens (10B) | Wall-clock | 771s | 870s | +12.8% |
| -- | Total Params | 17.04M | 21.76M (+4.72M) | +27.7% |

## Analysis

SwiGLU improves over the GELU baseline by **-0.0316 BPB** under fixed compute and **-0.0341 BPB** under fixed tokens. The gains are consistent and substantial across both evaluation regimes.

**SwiGLU outperforms GeGLU in this ablation suite.** Under fixed compute, SwiGLU's -0.0316 BPB improvement clearly exceeds GeGLU's -0.0183 -- a 73% larger gain. Under fixed tokens, SwiGLU holds a narrower but real edge (-0.0341 vs -0.0323). Both are gated activations with the same parameter count (3 projections, 21.76M params), differing only in gate nonlinearity (SiLU vs GeLU). The gap is most pronounced under fixed compute, where SwiGLU's advantage is decisive. This suggests that SiLU's smooth, unbounded-above, bounded-below gating signal provides better learned feature selection than GeLU's smooth taper at this scale.

**Per-token efficiency improves despite fewer tokens.** Under fixed compute, SwiGLU processes 11.1% fewer tokens (6.78B vs. 7.63B) due to the heavier three-projection forward pass, yet still achieves substantially lower BPB. The architecture learns more per token, offsetting the throughput reduction. Under fixed tokens, the 12.8% wall-clock overhead is proportionally smaller than the 27.7% parameter increase, reflecting the fact that the third projection shares the same hidden dimension rather than expanding it.

**Memory cost is moderate.** Peak usage rises by 1.5 GiB (+18%), from three projection matrices and their activation checkpoints. For the BPB improvement offered, this is a favorable trade-off at this scale.

**Verdict: adopt gating, prefer SwiGLU over GeGLU.** SwiGLU delivers the strongest gated activation results in this ablation suite, outperforming GeGLU in both evaluation regimes. If choosing a gated activation for this architecture, SwiGLU is the clear winner.

## Files

- `train_gpt.py`: Training script (baseline + SwiGLU modification)
- `swiglu.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
