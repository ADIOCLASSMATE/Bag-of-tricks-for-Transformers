# Input RMSNorm — Embedding Normalization Ablation

## Method Overview

This experiment adds **RMSNorm on the embedding output** before feeding it into the first transformer block. The raw token embedding activations are normalized to unit variance before entering the residual stream.

### Motivation

- The raw embedding output can have arbitrary scale depending on initialization, which affects all downstream layers
- Normalizing the input to the transformer blocks provides a consistent starting point for the residual stream
- This matches the design principle of pre-norm Transformers where each sub-layer receives normalized input

## What This Ablation Tests

| Component | Baseline | Input RMSNorm |
|---|---|---|
| Embedding output | Raw (unscaled) | `F.rms_norm(tok_emb(x), ...)` |
| Additional parameters | 0 | 0 (RMSNorm without learnable weight) |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| Input normalization | None | RMSNorm on embedding output |
| Code change | — | Added `F.rms_norm` after embedding lookup |

## Results

| Regime | Metric | Baseline | Input RMSNorm | Delta |
|---|---|---|---|---|
| Fixed Compute (600s) | Val BPB | 1.2979 | 1.2937 | -0.0042 |
| Fixed Compute (600s) | Val Loss | 2.1914 | 2.1843 | -0.0071 |
| Fixed Compute (600s) | Train Tokens | 7.67B | 7.63B | -0.5% |
| Fixed Compute (600s) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2851 | -0.0006 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1698 | -0.0011 |
| Fixed Tokens (10B) | Wall-clock | 772s | 777s | +0.6% |
| — | Total Params | 17,039,360 | 17,039,360 | 0 |

## Analysis

Input RMSNorm normalizes the embedding output to unit variance before it enters the residual stream, adding zero parameters and zero memory overhead.

The fixed-compute gain of -0.004 BPB shrinks to -0.0006 BPB under fixed-tokens, indicating the advantage is an early-convergence effect rather than a representational improvement. With more training, the model compensates for embedding scale through its own learned parameters, making the explicit normalization redundant.

The 0.6% wall-clock overhead from the extra normalization op is negligible, and the technique is strictly non-harmful. However, the vanishing benefit under equal training renders it inconsequential for well-tuned baselines.

**Verdict**: Near-neutral. Input RMSNorm is a free but negligible improvement at this scale. It may justify itself in deeper architectures or poorly-initialized embedding layers where the raw scale discrepancy is larger.

## Files

- `train_gpt.py`: Training script (baseline + input RMSNorm modification)
- `input-rmsnorm.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
