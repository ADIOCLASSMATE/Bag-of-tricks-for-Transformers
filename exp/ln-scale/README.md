# LN-Scale — Depth-Dependent RMSNorm Output Scaling

## Method Overview

This experiment multiplies each block's normalized activations by a depth-dependent factor:

`1 / sqrt(layer_idx + 1)`

The scaling is applied **after** `RMSNorm` and **before** the attention and MLP sublayers. The `RMSNorm` implementation itself is unchanged. A `layer_idx` argument is added to `Block` so each layer can compute its own fixed scaling factor.

### Origin

From **parameter-golf A09 LNScale**. The reported idea is to damp activation magnitude in deeper layers without changing the overall block topology.

### Motivation

- Deeper layers can accumulate larger activation magnitudes even when each sublayer is normalized
- A simple `1 / sqrt(l+1)` factor provides stronger damping in later blocks
- Applying the scale outside `RMSNorm` keeps the trick isolated and easy to reason about

## Impact on Training

- **Parameters**: No change
- **Throughput**: Negligible impact from one extra scalar multiply per normalized branch
- **Memory**: No meaningful change
- **Optimization**: Later layers receive smaller normalized inputs than earlier layers

## Key Differences from Baseline

| Component | baseline-sp1024 | ln-scale |
|---|---|---|
| Norm output scale | 1.0 | **`1 / sqrt(layer_idx + 1)`** |
| `Block` signature | no layer index | **adds `layer_idx`** |
| `RMSNorm` class | unchanged | unchanged |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | ln-scale | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | ln-scale | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The depth-dependent scaling reduced instability or over-amplification in later layers, making optimization cleaner.
- **If BPB is unchanged**: Baseline normalization was already sufficient at 9 layers, so the extra damping does not materially change training.
- **If BPB worsens**: The deeper layers may be over-damped, reducing their effective capacity and slowing useful feature formation.

## Files

- `train_gpt.py`: Baseline trainer with per-layer post-norm scaling in `Block.forward`
- `ln-scale.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
