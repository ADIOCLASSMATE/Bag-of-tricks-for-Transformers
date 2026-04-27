# Resid-Mix — Learnable Embedding Mixing Ablation

## Method Overview

This experiment adds **learnable per-dimension residual mixing** with the original embedding (x0). Each block takes an additional input — the raw embedding output — and mixes it with the current hidden state before sublayers:

```
mix = resid_mix  # shape [2, dim], init [1, 0]
x = mix[0] * x + mix[1] * x0
```

### Motivation

- In deep Transformers, the original embedding signal can become diluted through many residual connections
- Mixing in the raw embedding at every layer provides a persistent "anchor" to the input representation
- The learnable mixing weights allow each dimension to independently control how much embedding signal to retain
- Initialization of `[1, 0]` means the model starts with standard behavior (no embedding mixing) and can learn to incorporate it

## What This Ablation Tests

| Component | Baseline | Resid-Mix |
|---|---|---|
| Block input | `x` (residual stream only) | `mix[0]*x + mix[1]*x0` |
| Additional input | — | Raw embedding x0 passed to every block |
| Additional params | — | `2 × dim` per block |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `resid_mix` init | — | `[1, 0]` (standard behavior at start) |
| Block forward | `x = x + sublayer(norm(x))` | `x = mix*x + mix*x0; x = x + sublayer(norm(x))` |

## Results

| Regime | Metric | Baseline | Resid-Mix | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2625 | **-0.0313** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1317 | -0.0528 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.20B | -5.6% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,391 MiB | +2 MiB |
| Fixed Compute (10 min) | Wall-clock | — | 600s | — |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2527 | **-0.0320** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1152 | -0.0540 |
| Fixed Tokens (10B) | Wall-clock | 771s | 817s | +6.0% |
| — | Total Params | 17,039,360 | 17,048,576 | +9,216 |

## Analysis

Resid-mix delivers a consistent -0.031 BPB improvement across both fixed-compute and fixed-token regimes, making it one of the strongest tricks in the ablation suite.

The mechanism addresses a specific structural weakness: as the residual stream passes through successive blocks, the original embedding signal -- carrying token identity -- becomes progressively diluted. Re-injecting the raw embedding at every layer gives each block direct access to "what token is this?" alongside the increasingly abstract contextual representation. The per-dimension learnable mixing weights let the model calibrate how much identity signal to retain at each position, rather than forcing a fixed blend.

The `[1, 0]` initialization is a sensible default: the model begins with standard residual-stream behavior and can gradually learn to incorporate embedding information only where it helps. This avoids destabilizing early training.

Overhead is minimal. The mixing parameters add only 9,216 scalars (2 x dim x n_layer), resulting in negligible memory increase (+2 MiB). The 6.0% wall-clock slowdown under fixed tokens comes from the extra per-block mixing operation, which is a modest cost given the quality gain.

**Verdict**: Resid-mix is highly effective and efficient -- it provides a substantial quality improvement at near-zero memory cost by ensuring every layer retains access to the original token identity signal.

## Files

- `train_gpt.py`: Training script (baseline + resid-mix modifications)
- `resid-mix.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
