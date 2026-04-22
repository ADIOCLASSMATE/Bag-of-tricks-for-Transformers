# muP-Scaled-Output-Projections — Depth-Scaled Projection Initialization

## Method Overview

This experiment applies a muP-style output-projection scaling after initialization: for each
non-zero-init projection weight, it multiplies the weight by `1 / sqrt(2 * L)`, where `L` is the
number of transformer blocks.

Concretely, the trainer keeps the baseline zero-initialization for layers marked with
`_zero_init=True`, and only rescales projection layers whose names match `.proj`.

### Origin

From later **parameter-golf** combo records that use **OrthoInit + muP-scaled output projections**.
This experiment isolates the projection-scaling part from the orthogonal-init part.

### Motivation

- Smaller output-projection scale can reduce early residual-branch amplitude
- The trick is initialization-only, so it does not change the runtime graph
- Isolating it from OrthoInit makes the interaction testable

## Impact on Training

- **Parameters**: No change
- **Throughput**: No measurable runtime impact after initialization
- **Memory**: No change
- **Optimization**: Only non-zero-init `.proj` weights are rescaled; `_zero_init=True` output projections remain unchanged

## Important Note

In the current SP1024 baseline, the main transformer output projections are marked `_zero_init=True`.
Those layers stay zero-initialized here, so this isolated experiment may have little or no effect
unless the model includes non-zero-init `.proj` layers.

## Key Differences from Baseline

| Component | baseline-sp1024 | mup-scaled-output-projections |
|---|---|---|
| Non-zero-init `.proj` weights | default init scale | **multiplied by `1 / sqrt(2L)`** |
| `_zero_init=True` projections | zero init | zero init |
| Runtime graph | unchanged | unchanged |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | mup-scaled-output-projections | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | mup-scaled-output-projections | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The smaller non-zero-init projection scale likely reduced early residual-branch noise and made optimization cleaner.
- **If BPB is unchanged**: The current baseline may expose too few non-zero-init `.proj` weights for this trick to matter in isolation.
- **If BPB worsens**: The reduced projection scale may over-dampen useful residual updates when applied without the rest of the muP-style initialization stack.

## Files

- `train_gpt.py`: Baseline trainer with isolated muP-style output projection scaling
- `mup-scaled-output-projections.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
