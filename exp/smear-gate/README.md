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

| Component | Baseline | smear-gate |
|---|---|---|
| Embedding preprocessing | none | **learned previous-token mix** |
| Mixing coefficient | — | **per-dimension `sigmoid(smear_gate[d])`** |
| Initial coefficient | — | **~0.95 on current token / ~0.05 on previous token** |
| Optimizer routing | baseline only | **explicitly adds `smear_gate` to scalar optimizer** |

## Results

| Regime | Metric | Baseline | Smear-Gate | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.3016 | +0.0037 |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1977 | +0.0063 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.56B | -1.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,454 MiB | +65 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2873 | +0.0016 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1735 | +0.0026 |
| Fixed Tokens (10B) | Wall-clock | 772s | 781s | +1.2% |
| — | Total Params | 17,039,360 | 17,039,872 | +512 |

## Analysis

Smear-gate regresses BPB by +0.0016 (fixed-tokens) to +0.0037 (fixed-compute). Per-dimension gating provides no benefit over the simpler global smear, confirming that the problem is not expressiveness but redundancy with self-attention.

Self-attention already mixes information across positions. Pre-mixing embeddings before they enter the attention stack is at best redundant and at worst interferes with the patterns attention needs to learn. The per-dimension gate gives the model the flexibility to choose a useful operating point per feature, but it never finds one: the near-identity initialization (sigmoid(3) ~ 0.95) keeps the model close to baseline behavior, and gradient signal does not pull the gate toward a beneficial mixing ratio.

The added parameter budget (+512) and memory (+65 MiB) produce no return. The throughput penalty under fixed-compute (-1.4% tokens) accounts for roughly half the FC BPB gap; the remainder is genuine quality degradation.

**Verdict**: Slightly harmful. Per-dimension gating does not rescue embedding-level pre-mixing because the obstacle is not insufficient flexibility but structural redundancy with attention.

## Files

- `train_gpt.py`: Baseline trainer with source-faithful parameter-golf-style per-dimension smear gate
- `smear-gate.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
