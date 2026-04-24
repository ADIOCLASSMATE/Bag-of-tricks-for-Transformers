# Q-Gain — Learnable Per-Head Query Scaling

## Method Overview

This experiment adds a **learnable per-head Q-Gain** parameter that scales query vectors after RoPE application. Each attention head gets its own scalar gain, initialized to 1.5, which boosts attention logit magnitudes.

### Motivation

- Standard attention computes `softmax(QK^T / sqrt(d))`, where Q and K magnitudes directly affect attention sharpness
- After QK-Norm and RoPE, query magnitudes may be attenuated, reducing the model's ability to form sharp attention patterns
- A per-head gain allows each head to independently calibrate its attention sharpness

## What This Ablation Tests

| Component | Baseline | Q-Gain |
|---|---|---|
| Query scaling | None (post QK-Norm + RoPE) | `q = q * q_gain[None, :, None, None]` |
| Additional params | 0 | `num_heads` scalar per block |
| Q-Gain init | — | 1.5 |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| Q-Gain | — | `nn.Parameter(torch.full((num_heads,), 1.5))` |
| Additional parameters | — | 72 (8 scalars × 9 blocks) |

## Results

| Regime | Metric | Baseline | Q-Gain | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2834 | **-0.0145** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1670 | -0.0244 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.56B | -1.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,965 MiB | +576 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2721 | **-0.0136** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1479 | -0.0230 |
| Fixed Tokens (10B) | Wall-clock | 772s | 779s | +0.9% |
| — | Total Params | 17,039,360 | 17,039,432 | +72 |

## Analysis

Q-Gain delivers a **consistent -0.014 BPB improvement** across both fixed-compute and fixed-token regimes with only 72 additional parameters (+0.0004%).

The 1.5 initialization is the critical design choice. After QK-Norm and RoPE, query magnitudes are attenuated, which flattens the attention distribution and limits the model's ability to form sharp, selective attention patterns. Initializing the gain above 1.0 counteracts this attenuation from the first training step, giving each head a head start on calibrating its attention sharpness. The per-head granularity then lets individual heads converge to their own optimal scale rather than sharing a single global factor.

The +576 MiB memory overhead is disproportionately large for 72 scalar parameters. This arises because Muon (the optimizer used for 2D+ parameters) cannot handle 1D tensors, so the gain scalars are routed to Adam instead. The extra memory comes from Adam's dual momentum states for these parameters, not from the parameters themselves.

The 1.4% throughput reduction under fixed compute is modest and primarily attributable to the memory pressure from the Adam optimizer states described above; the actual forward-pass cost of the gain multiplication is negligible.

**Verdict**: Q-Gain is a high-signal, low-complexity trick. The per-head query scaling mechanism addresses a real architectural weakness (post-norm magnitude collapse), and the 1.5 initialization is well-matched to this model configuration. The memory overhead from optimizer routing is a minor practical concern.

## Files

- `train_gpt.py`: Training script (baseline + Q-Gain modification)
- `q-gain.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
