# Warmdown-3600 — Tripled Learning Rate Cooldown Period

## Method Overview

This experiment triples the learning rate warmdown (cooldown) period from the baseline's 1200 iterations to **3600 iterations**, placing it close to the parameter-golf SOTA value of 3500. This is a manifest-only change — the training script is identical to baseline.

With 3600 warmdown iterations out of ~20,000 total iterations (at the 10B token budget), approximately **18% of training time** is spent in the cooldown phase. This aggressive warmdown schedule allocates a substantial fraction of compute to low-learning-rate fine-tuning.

### Origin

From **parameter-golf T02** and **nanogpt 2025-03-06_LongerCooldown**. The parameter-golf SOTA configuration used `warmdown_iters=3500`, demonstrating that very long cooldown periods are beneficial. This experiment tests 3600, which is slightly beyond the SOTA value, to probe the upper end of the warmdown sweep.

### Motivation

- The parameter-golf SOTA (warmdown_iters=3500) suggests the optimum is far above baseline's 1200
- 3600 iterations gives the optimizer extensive time to polish weights at progressively lower learning rates
- Tests whether the benefit continues to scale beyond the known-good 3500 value

## Impact on Training

- **Parameters**: No change
- **Throughput**: No change (identical training script)
- **Memory**: No change
- **Training duration**: ~18% of total iterations spent in warmdown phase (vs ~6% at baseline)

## Key Differences from Baseline

| Component | baseline-sp1024 | warmdown-3600 |
|---|---|---|
| `warmdown_iters` | 1200 | **3600** |
| Code change | — | **None (manifest only)** |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | warmdown-3600 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | warmdown-3600 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The optimal warmdown length is indeed much longer than baseline, consistent with parameter-golf findings. The extra low-LR iterations help the model find a flatter minimum.
- **If BPB is unchanged**: There may be diminishing returns beyond some threshold — warmdown-2400 may capture most of the benefit.
- **If BPB worsens**: 3600 iterations of warmdown may be overshooting the sweet spot, spending too much compute at learning rates too low to make meaningful progress. The optimal value likely lies between 1200 and 3600.

## Files

- `train_gpt.py`: Identical copy of baseline training script (no modifications)
- `warmdown-3600.json`: Experiment manifest (`warmdown_iters: 3600`)
- `logs/`: Training output (automatically generated)
