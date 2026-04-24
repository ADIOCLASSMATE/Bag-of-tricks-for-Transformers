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

| Component | Baseline | ln-scale |
|---|---|---|
| Norm output scale | 1.0 | **`1 / sqrt(layer_idx + 1)`** |
| `Block` signature | no layer index | **adds `layer_idx`** |
| `RMSNorm` class | unchanged | unchanged |
| Everything else | identical | identical |

## Results

| Regime | Metric | Baseline | LN-Scale | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.3118 | +0.0139 |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.2149 | +0.0235 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.58B | -1.2% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2923 | +0.0066 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1821 | +0.0112 |
| Fixed Tokens (10B) | Wall-clock | 772s | 776s | +0.5% |
| -- | Total Params | 17.04M | 17.04M | 0 |

## Analysis

LN-Scale degrades quality across both evaluation regimes (+0.014 BPB fixed-compute, +0.007 BPB fixed-tokens). The `1/sqrt(l+1)` damping factor is counterproductive at this depth.

The root cause is over-damping. At the deepest layer (index 8), the scale drops to `1/sqrt(9) ≈ 0.333`, attenuating normalized activations by roughly 67%. Residual contributions from later blocks shrink proportionally, effectively reducing the model's functional depth. Earlier layers are also attenuated but less severely (e.g., layer 0 retains full scale, layer 4 gets `1/sqrt(5) ≈ 0.447`).

A 9-layer baseline does not exhibit the activation-magnitude growth that LN-Scale is designed to correct. The fixed schedule therefore constrains the model without addressing any real pathology. Whether the trick helps in deeper architectures where activation growth is genuine remains an open question.

**Verdict**: Harmful at this scale. Depth-dependent damping is too aggressive for shallow models and provides no compensating benefit.

## Files

- `train_gpt.py`: Baseline trainer with per-layer post-norm scaling in `Block.forward`
- `ln-scale.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
