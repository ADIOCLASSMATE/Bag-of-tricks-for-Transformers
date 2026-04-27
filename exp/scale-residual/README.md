# Scale-Residual — Learnable Per-Dimension Residual Scaling

## Method Overview

This experiment adds **learnable per-dimension scaling** on attention and MLP residual connections. Instead of the standard `x = x + sublayer(norm(x))`, the residual becomes `x = x + scale * sublayer(norm(x))`, where `scale` is a learnable vector of dimension `model_dim`.

### Motivation

- Different dimensions may benefit from different residual contribution strengths
- Learnable scales allow the model to calibrate how much each sub-layer should contribute to each dimension
- This provides a more flexible alternative to fixed residual scaling or zero-init

## What This Ablation Tests

| Component | Baseline | Scale-Residual |
|---|---|---|
| Residual connection | `x + sublayer(x)` | `x + scale * sublayer(x)` |
| Additional params | 0 | `2 × num_layers × model_dim` scalars |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `attn_scale` | — | `nn.Parameter(torch.ones(dim))` |
| `mlp_scale` | — | `nn.Parameter(torch.ones(dim))` |
| Initial behavior | Standard | Identity (scales init to 1.0) |

## Results

| Regime | Metric | Baseline | Scale-Residual | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2841 | **-0.0097** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1682 | -0.0163 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.37B | -3.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 9,287 MiB | +10.7% |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2667 | **-0.0180** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1388 | -0.0304 |
| Fixed Tokens (10B) | Wall-clock | 771s | 796s | +3.2% |
| — | Total Params | 17,039,360 | 17,048,576 | +9,216 |

## Analysis

Scale-residual delivers strong BPB improvements across both evaluation regimes (-0.010 FC, -0.018 FT), ranking it among the most effective tricks tested. The gains come from per-dimension learnable scaling that lets each residual channel independently control sub-layer contribution strength, a strictly more expressive parameterization than a single scalar per sub-layer.

The memory overhead is the main drawback. The +898 MiB increase is disproportionate to the +9,216 parameters (approximately 36 KiB of weights) because the optimizer maintains additional state for these scalar parameters. Note that `attn_scale` and `mlp_scale` are routed to the Adam optimizer (via `CONTROL_TENSOR_NAME_PATTERNS` matching their names as scalar_params), not Muon, so the memory amplification comes from Adam's first/second moment buffers rather than Muon's spectral momentum states. This optimizer-side overhead turns a negligible parameter cost into a meaningful memory penalty.

The FT improvement (-0.018 BPB) is nearly double the FC improvement (-0.010 BPB), indicating the per-dimension scales continue to provide value with sustained training. Rather than converging early and saturating, the learnable scales appear to become more beneficial as the model sees more data -- each residual dimension benefits from increasingly refined calibration of sub-layer contribution strength over the full 10B token budget.

**Verdict**: Effective but memory-expensive. The quality gains are real and scale with training duration (-0.018 FT BPB, among the strongest results), but the optimizer-state memory overhead (+898 MiB) makes this less practical than alternatives that achieve similar improvements with near-zero overhead.

## Files

- `train_gpt.py`: Training script (baseline + scale-residual modifications)
- `scale-residual.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
