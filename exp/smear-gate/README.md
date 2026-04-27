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
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2965 | +0.0027 |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1890 | +0.0045 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.53B | -1.3% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,454 MiB | +65 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2904 | +0.0057 |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1788 | +0.0096 |
| Fixed Tokens (10B) | Wall-clock | 771s | 781s | +1.3% |
| — | Total Params | 17,039,360 | 17,039,872 | +512 |

## Analysis

Smear-gate regresses BPB by +0.0027 (fixed-compute) to +0.0057 (fixed-tokens). Per-dimension gating provides no benefit at this training scale, and the reasons are more nuanced than simple redundancy with self-attention.

**Why "redundancy with attention" is the wrong diagnosis.** Smear-gate operates on raw token embeddings before any transformer layer: the smeared embeddings become the input to the entire 9-layer stack. Every subsequent attention layer, MLP, and residual connection sees only the already-mixed representation. Later self-attention in principle cannot "undo" the initial blending to recover the original per-token embeddings — the information set entering the first block is already altered. Pre-mixing embeddings is a genuine architectural change to the model's input, not a redundant preface to attention.

**The more plausible mechanism: near-identity initialization with weak gradient signal.** The gate vector is initialized at 3.0, producing sigmoid(3.0) ≈ 0.95, meaning the model starts at 95% current-token / 5% previous-token — nearly indistinguishable from baseline identity. The gradient through the sigmoid at value 3.0 has a multiplier of sigmoid(3) × (1 − sigmoid(3)) ≈ 0.045. Combined with a scalar optimizer learning rate of 0.04, the effective per-step movement of the raw gate parameter is tiny. Over ~14,368 steps (~7.5B tokens under fixed-compute), the gate has limited opportunity to drift meaningfully away from its initialization. The signal-to-move ratio is simply too low at this training budget.

**Why per-dimension flexibility doesn't help here.** The motivation for 512 independent gate values was to let training assign stronger or weaker mixing to different embedding dimensions — e.g., syntactic features might benefit from more previous-token context while semantic features prefer sharper per-token identity. But with initialization near identity and weak gradients, the per-dimension structure adds 512 parameters without the gate ever exploring the space where those degrees of freedom matter. The gate effectively functions as a single near-frozen scalar.

**The fixed 5% noise floor.** Even at initialization, every token embedding contains 5% of the previous token's content. This introduces a small but unconditional perturbation to every input vector that the transformer must compensate for. Across 9 layers, this slight embedding distortion may compound into harder-to-recover modeling errors.

The added parameter budget (+512) and memory (+65 MiB) produce no return. The throughput penalty under fixed-compute (-1.3% tokens) accounts for roughly half the FC BPB gap; the remainder is genuine quality degradation from the initial mixing noise rather than a fundamental architectural flaw.

**Verdict**: Slightly harmful at this scale. The sigmoid(3.0) initialization places the gate at a near-identity operating point with very weak gradient signal, preventing the model from discovering whether per-dimension embedding blending is useful. The trick could conceivably benefit from (a) a more balanced initialization (e.g., 0.0 giving gate = 0.5), (b) a higher learning rate specifically for the gate parameters, or (c) longer training to accumulate sufficient gradient — but none of those were explored here.

## Files

- `train_gpt.py`: Baseline trainer with source-faithful parameter-golf-style per-dimension smear gate
- `smear-gate.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
