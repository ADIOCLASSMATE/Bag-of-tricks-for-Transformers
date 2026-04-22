# Warmdown-2400 — Doubled Learning Rate Cooldown Period

## Method Overview

This experiment doubles the learning rate warmdown (cooldown) period from the baseline's 1200 iterations to **2400 iterations**. This is a manifest-only change — the training script is identical to baseline.

The warmdown phase is the final portion of training where the learning rate decays from its peak value down to zero (or near-zero). By extending this phase, the optimizer has more iterations at progressively lower learning rates, allowing it to settle into a flatter, more generalizable minimum.

### Origin

From **parameter-golf T02** and **nanogpt 2025-03-06_LongerCooldown**. The nanogpt speedrun community found that longer cooldown periods consistently improve final validation loss, especially when training on large token budgets. The parameter-golf SOTA configuration used `warmdown_iters=3500`, suggesting the baseline's 1200 is suboptimal.

### Motivation

- Longer warmdown gives the optimizer more time to fine-tune weights at low learning rates
- Reduces the risk of "overshooting" good minima during the final training phase
- A simple hyperparameter change with zero compute overhead per iteration

## Impact on Training

- **Parameters**: No change
- **Throughput**: No change (identical training script)
- **Memory**: No change
- **Training duration**: With fixed-token budget, more iterations are spent in the warmdown phase; with fixed-compute, the warmdown occupies a larger fraction of wall-clock time

## Key Differences from Baseline

| Component | baseline-sp1024 | warmdown-2400 |
|---|---|---|
| `warmdown_iters` | 1200 | **2400** |
| Code change | — | **None (manifest only)** |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | warmdown-2400 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | warmdown-2400 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The baseline's 1200-iteration warmdown was too aggressive, causing the optimizer to overshoot good minima. The gentler 2400-iteration decay allows better convergence.
- **If BPB is unchanged**: At this scale, the warmdown duration may not be a bottleneck — 1200 iterations already provides sufficient cooldown.
- **If BPB worsens**: Spending too many iterations at very low learning rates may waste compute that could have been spent at higher learning rates making meaningful progress.

## Files

- `train_gpt.py`: Identical copy of baseline training script (no modifications)
- `warmdown-2400.json`: Experiment manifest (`warmdown_iters: 2400`)
- `logs/`: Training output (automatically generated)
