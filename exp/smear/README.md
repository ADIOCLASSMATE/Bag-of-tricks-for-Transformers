# Smear — Modded-NanoGPT-Style Learned Smear Injection

## Method Overview

This experiment applies the **smear** mechanism used in the modded-nanogpt record: each token can receive an additive contribution from the previous token embedding, with the strength controlled by a learned content-dependent gate and a learned scalar `smear_lambda`.

For positions `t > 0`, the update is:

`x[t] = x[t] + smear_lambda * sigmoid(W_gate x[t][:12]) * x[t-1]`

Both `smear_lambda` and the gate weights are initialized to zero, so the model starts as an exact no-op and learns whether smear is useful.

### Origin

From **nanogpt-speedrun track_1_short / 2025-09-18_Smear**. The original record did not use a fixed 0.5 average; it used a learned additive previous-token injection that starts as a no-op.

### Motivation

- Neighbor mixing can provide a cheap inductive bias for local token continuity
- The trick acts before the first transformer block, so every later layer sees the mixed representation
- Zero initialization keeps the ablation clean by matching baseline behavior at step 0

## Impact on Training

- **Parameters**: Adds one scalar plus a tiny `12 x 1` gate matrix
- **Throughput**: Negligible overhead from one small linear gate and a few elementwise ops
- **Memory**: Negligible increase
- **Optimization**: Starts as a no-op and learns whether to inject previous-token context

## Key Differences from Baseline

| Component | baseline-sp1024 | smear |
|---|---|---|
| Embedding preprocessing | none | **learned additive previous-token injection** |
| Gate | — | **`sigmoid(W_gate x[:12])`** |
| Initial behavior | baseline | **exact no-op** |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | smear | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | smear | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The fixed local-context bias gave the network a better starting representation for short-range dependencies.
- **If BPB is unchanged**: Attention already learns this local mixing well enough, so explicit smear adds little.
- **If BPB worsens**: The forced local blend may wash out useful token identity information too early.

## Files

- `train_gpt.py`: Baseline trainer with source-faithful modded-nanogpt smear
- `smear.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
