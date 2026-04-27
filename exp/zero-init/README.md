# Zero-Init — Zero-Initialized Output Projections Ablation

## Method Overview

This experiment **zero-initializes the output projection layers** (attn.proj, mlp.proj) so each transformer block starts as an identity mapping: `x + 0 = x`. This stabilizes early training by preserving the signal from lower layers.

### Motivation

- Standard initialization gives output projections random weights, causing early blocks to produce noisy residual contributions
- Zero init makes each block initially a no-op, so the model starts as a simple embedding → unembedding pass
- Gradient flow is preserved through the residual stream, and blocks gradually learn to contribute
- This technique is used in GPT-2 and many subsequent architectures

## What This Ablation Tests

| Component | Baseline | Zero-Init |
|---|---|---|
| Output proj init | Default (Kaiming) | `nn.init.zeros_()` |
| Initial block behavior | Random contribution | Identity (no-op) |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `attn.proj` init | Default | Zero |
| `mlp.proj` init | Default | Zero |

## Results

| Regime | Metric | Baseline | Zero-Init | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2905 | **-0.0033** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1790 | -0.0055 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.65B | +0.3% |
| Fixed Compute (10 min) | Wall-clock | — | 600s | — |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2883 | **+0.0036** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1753 | +0.0061 |
| Fixed Tokens (10B) | Wall-clock | 771s | 771s | 0 |
| — | Total Params | 17,039,360 | 17,039,360 | 0 |

## Analysis

Zero-init yields a modest improvement under fixed-compute (-0.0033 BPB) but a slight regression under fixed-tokens (+0.0036 BPB). The effect is regime-dependent rather than uniformly neutral.

The mechanism: zero-initialized output projections make each transformer block start as an identity, suppressing noisy residual contributions during early optimization. This initial stabilization provides a transient benefit -- under fixed-compute, where training is cut off at 10 minutes, the model benefits from this smoother start (-0.0033 BPB). However, given enough data (fixed-tokens, 10B tokens), the early advantage fades and the initialization restriction slightly constrains final convergence (+0.0036 BPB).

This training regime combines QK-Norm and the Muon optimizer, both of which provide strong gradient normalization. Muon's spectral normalization and momentum warmup handle much of the gradient-scale instability that zero-init is designed to address. The small FC benefit suggests zero-init provides marginal complementary stabilization during the initial optimization phase, but the FT regression indicates it is slightly over-constraining in the long run.

**Verdict**: Regime-dependent. Zero-init provides a small early-training benefit (-0.0033 FC BPB) but slightly hurts final convergence (+0.0036 FT BPB). Alongside QK-Norm + Muon, the trick is of limited value. It may carry more weight with less stable optimizers (e.g., vanilla Adam) or without QK-Norm.

## Files

- `train_gpt.py`: Training script (baseline + zero-init modification)
- `zero-init.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
