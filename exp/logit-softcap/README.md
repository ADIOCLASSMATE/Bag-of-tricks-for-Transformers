# Logit-Softcap — Logit Magnitude Capping Ablation

## Method Overview

This experiment adds **logit softcapping**: `logits = softcap * tanh(logits / softcap)`, which bounds logit magnitudes via the tanh function. The default softcap value is 30.0.

### Motivation

- Extreme logit values can cause numerical instability and sharp gradient spikes during training
- Softcapping via tanh smoothly constrains logits to the range `[-softcap, +softcap]`
- This technique is used in Gemma 2/4 and other production models
- Unlike gradient clipping, softcapping prevents extreme logits from forming in the first place

## What This Ablation Tests

| Component | Baseline | Logit-Softcap |
|---|---|---|
| Logit processing | Raw (unbounded) | `softcap * tanh(logits / softcap)` |
| Logit range | Unbounded | `[-30.0, +30.0]` |
| Additional params | 0 | 0 (softcap is a hyperparameter, not learned) |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `logit_softcap` | — | 30.0 |
| Code change | — | Apply `softcap * tanh(logits / softcap)` before cross-entropy |

## Results

| Regime | Metric | Baseline | Logit-Softcap | Delta |
|---|---|---|---|---|
| Fixed Compute | Val BPB | 1.2979 | 1.2583 | **-0.0396** |
| Fixed Compute | Val Loss | 2.1914 | 2.1247 | -0.0667 |
| Fixed Compute | Train Tokens | 7.67B | 7.62B | -0.7% |
| Fixed Compute | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Compute | Wall-clock | 600s | 600s | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2526 | **-0.0331** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1150 | -0.0559 |
| Fixed Tokens (10B) | Wall-clock | 772s | 777s | +0.6% |
| — | Total Params | 17.04M | 17.04M | 0 |

## Analysis

Logit-softcap applies `softcap * tanh(logits / softcap)` before cross-entropy, bounding logit magnitudes to `[-30, +30]` without adding parameters or changing memory. The effect is consistent across both evaluation regimes: **-0.040 BPB** under fixed compute and **-0.033 BPB** under fixed tokens, making it the strongest zero-cost trick in this ablation suite.

The mechanism is straightforward. Without capping, the model can drive correct-token logits to extreme values, which lowers training loss but reflects overconfidence rather than genuine generalization. The tanh saturation regime compresses large logit magnitudes, forcing the model to redistribute probability mass across more tokens rather than concentrating it on a single prediction. This yields better-calibrated outputs that transfer more effectively to validation data.

The softcap value of 30.0 provides a useful operating range: logits below ~15 pass through nearly linearly (tanh(0.5) ~ 0.46), while only the tail of the distribution encounters meaningful compression. This means the cap intervenes selectively — it shapes extreme predictions without distorting the typical logit landscape.

**Verdict**: Unambiguous win. Substantial quality gain at zero parameter, memory, or compute cost. Should be adopted as a default training technique.

## Files

- `train_gpt.py`: Training script (baseline + logit-softcap modification)
- `logit-softcap.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
