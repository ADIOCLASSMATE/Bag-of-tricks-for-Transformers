# Residual-Lambda-Init-11 — Residual/X0 Mix Initialization at 1.1 / 0.1

## Method Overview

This experiment changes the initial residual-path coefficient in each transformer block from **1.0** to **1.1** and also initializes the `x0` mix coefficient from **0.0** to **0.1**.

This is a minimal code change that only affects initialization. The forward structure, optimizer split, training schedule, tokenizer, and model size remain identical to `baseline-sp1024`.

### Origin

From **slowrun T08 ResidualLambdaInit11**. The slowrun tiny-track change initializes `resid_lambdas` to `1.1` and `x0_lambdas` to `0.1`, so this experiment now mirrors that paired initialization instead of changing only the residual side.

### Motivation

- A slightly larger residual coefficient can help information propagate more directly through depth at the start of training
- A small nonzero `x0` coefficient gives every block immediate access to the input stream, matching the slowrun formulation
- The change is isolated to initialization, so it is easy to attribute any BPB shift
- It is effectively zero-cost in throughput and memory

## Impact on Training

- **Parameters**: No change
- **Throughput**: No measurable change
- **Memory**: No change
- **Optimization**: Early updates start from a slightly stronger residual backbone

## Key Differences from Baseline

| Component | baseline-sp1024 | residual-lambda-init-11 |
|---|---|---|
| `resid_mix[0]` init | 1.0 | **1.1** |
| `resid_mix[1]` init | 0.0 | **0.1** |
| Code change | — | **paired init change** |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | residual-lambda-init-11 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | residual-lambda-init-11 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The baseline residual coefficient was slightly too conservative, and the stronger residual path helped preserve useful signal through the stack.
- **If BPB is unchanged**: This initialization perturbation is too small to matter at this model scale, or training quickly washes it out.
- **If BPB worsens**: The larger residual coefficient may make early activations too dominant, reducing the benefit of learned transformations in the block.

## Files

- `train_gpt.py`: Baseline trainer with slowrun-style residual/x0 initialization
- `residual-lambda-init-11.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
