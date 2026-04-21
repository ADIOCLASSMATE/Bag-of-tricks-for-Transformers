# Smear-Gate — Parameter-Golf-Style Per-Dimension Smear Gate

## Method Overview

This experiment implements the **parameter-golf SmearGate** as a learned **per-dimension** gate vector. Token embeddings are mixed with the previous position before entering the transformer stack:

`x[t] = gate * x[t] + (1 - gate) * x[t-1]`

where `gate = sigmoid(smear_gate)` is learned independently for each embedding dimension. The gate is initialized at `3.0`, so `sigmoid(3.0) ≈ 0.95`: the model starts near identity and learns how much previous-token blending is useful per dimension.

The first position uses a zero previous embedding, matching the standalone parameter-golf implementation more closely than an unchanged first token.

### Origin

From **parameter-golf A04 SmearGate**. In the source implementation, the gate is per-dimension rather than a single global scalar, and it is initialized near identity rather than at 0.5 mixing.

### Motivation

- A single global mixing ratio is often too coarse
- A per-dimension gate lets training choose which embedding features should carry previous-token context
- Near-identity initialization avoids forcing heavy mixing before training has evidence it helps

## Impact on Training

- **Parameters**: Adds one learned gate value per embedding dimension
- **Throughput**: Negligible overhead from one sigmoid and a few elementwise ops
- **Memory**: Negligible increase
- **Optimization**: The model can learn whether local embedding mixing should be stronger, weaker, or near zero

## Key Differences from Baseline

| Component | baseline-sp1024 | smear-gate |
|---|---|---|
| Embedding preprocessing | none | **learned previous-token mix** |
| Mixing coefficient | — | **per-dimension `sigmoid(smear_gate[d])`** |
| Initial coefficient | — | **~0.95 on current token / ~0.05 on previous token** |
| Optimizer routing | baseline only | **explicitly adds `smear_gate` to scalar optimizer** |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | smear-gate | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | smear-gate | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The model benefits from learning its own local mixing strength rather than using a fixed `alpha=0.5`.
- **If BPB is unchanged**: The learned gate may converge near 0.5, suggesting the fixed smear already captured the useful operating point.
- **If BPB worsens**: The extra flexibility may destabilize the embedding stream or make optimization noisier than the fixed-smear variant.

## Files

- `train_gpt.py`: Baseline trainer with source-faithful parameter-golf-style per-dimension smear gate
- `smear-gate.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
