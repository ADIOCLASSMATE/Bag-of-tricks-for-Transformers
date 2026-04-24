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
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2977 | -0.0002 |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1911 | -0.0003 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.65B | -0.3% |
| Fixed Compute (10 min) | Wall-clock | — | 600s | — |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2828 | -0.0029 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1659 | -0.0050 |
| Fixed Tokens (10B) | Wall-clock | 772s | 775s | +0.4% |
| — | Total Params | 17,039,360 | 17,039,360 | 0 |

## Analysis

Zero-init yields a negligible -0.0002 BPB under fixed-compute and a modest -0.0029 BPB under fixed-tokens. The effect is nearly neutral in both regimes.

The mechanism is straightforward: zero-initialized output projections make each transformer block start as an identity, suppressing noisy residual contributions during early optimization. However, this training regime already combines QK-Norm and the Muon optimizer, both of which provide strong gradient normalization. Muon's spectral normalization and momentum warmup handle the gradient-scale instability that zero-init is designed to address, rendering the intervention largely redundant.

The fixed-tokens improvement (-0.0029 BPB) slightly exceeds the fixed-compute improvement (-0.0002 BPB). This gap suggests that zero-init may confer a small convergence benefit when the model trains on more data, but the magnitude is too small to be practically meaningful.

**Verdict**: Near-neutral. Zero-init is superfluous alongside QK-Norm + Muon, which already stabilize gradient flow. It may carry more weight with less stable optimizers (e.g., vanilla Adam) or without QK-Norm.

## Files

- `train_gpt.py`: Training script (baseline + zero-init modification)
- `zero-init.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
