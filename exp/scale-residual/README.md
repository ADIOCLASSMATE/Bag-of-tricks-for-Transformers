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
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2808 | **-0.0171** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1626 | -0.0288 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.35B | -4.2% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 9,287 MiB | +10.7% |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2734 | **-0.0123** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1501 | -0.0208 |
| Fixed Tokens (10B) | Wall-clock | 772s | 803s | +4.0% |
| — | Total Params | 17,039,360 | 17,048,576 | +9,216 |

## Analysis

Scale-residual delivers consistent BPB improvements across both evaluation regimes (-0.017 FC, -0.012 FT), ranking it among the stronger tricks tested. The gains come from per-dimension learnable scaling that lets each residual channel independently control sub-layer contribution strength, a strictly more expressive parameterization than a single scalar per sub-layer.

The memory overhead is the main drawback. The +898 MiB increase is disproportionate to the +9,216 parameters (approximately 36 KiB of weights) because Muon treats these as scalar parameters and maintains spectral momentum states that are far larger than the parameters themselves. This optimizer-side amplification turns a negligible parameter cost into a meaningful memory penalty.

The FC improvement (-0.017) exceeds FT (-0.012), suggesting the per-dimension scales are most impactful during early training when the model is calibrating the residual-sublayer balance. By the time 10B tokens are consumed, this advantage has partially compressed, indicating the scales converge toward stable values relatively early.

**Verdict**: Effective but memory-expensive. The quality gains are real and consistent, but the Muon-driven memory overhead makes this less practical than alternatives (e.g., U-Net skip connections) that achieve similar improvements with near-zero overhead.

## Files

- `train_gpt.py`: Training script (baseline + scale-residual modifications)
- `scale-residual.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
