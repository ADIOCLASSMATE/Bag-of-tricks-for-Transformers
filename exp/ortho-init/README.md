# Ortho-Init — Orthogonal Initialization for All Non-Zero-Init Linear Layers

## Method Overview

This experiment applies `nn.init.orthogonal_` to **every non-zero-init linear weight matrix** while preserving the baseline zero-initialization of layers marked with `_zero_init=True`.

In this codebase, that means the attention output projection and MLP output projection remain zero-initialized exactly as in the baseline, while the other linear maps switch from the default PyTorch init to orthogonal init. This matches the earlier standalone parameter-golf OrthoInit formulation more closely than later stacked variants that also rescaled projection weights.

### Origin

From **parameter-golf T07 OrthoInit**. The source-faithful standalone formulation initializes all non-zero-init linear layers with `nn.init.orthogonal_()` and leaves `_zero_init` layers unchanged.

### Motivation

- Orthogonal weights preserve directional structure better than generic random initializations
- The change targets only linear layer initialization, not runtime computation
- Preserving `_zero_init` output projections avoids conflating orthogonal init with a separate residual-output initialization ablation

## Impact on Training

- **Parameters**: No change
- **Throughput**: No measurable runtime impact after initialization
- **Memory**: No change
- **Optimization**: All non-zero-init linear maps start orthogonal, while residual output projections preserve baseline zero-init behavior

## Key Differences from Baseline

| Component | baseline-sp1024 | ortho-init |
|---|---|---|
| Non-zero-init linear init | default PyTorch init | **orthogonal** |
| `_zero_init=True` projections | zero init | zero init |
| Runtime graph | unchanged | unchanged |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | ortho-init | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | ortho-init | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The orthogonal weights likely gave the network a better-conditioned starting point for learning useful representations.
- **If BPB is unchanged**: Initialization geometry may not be a major bottleneck at this scale, especially with the baseline's existing normalization and optimizer setup.
- **If BPB worsens**: Orthogonal init may be mismatched with the rest of the baseline recipe, or the retained zero-init output projections may dominate the effective early dynamics.

## Files

- `train_gpt.py`: Baseline trainer with source-faithful orthogonal init for all non-`_zero_init` linear layers
- `ortho-init.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
