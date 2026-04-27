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
| Fixed Compute (600s) | Val BPB | 1.2938 | 1.2965 | +0.0027 |
| Fixed Compute (600s) | Val Loss | 2.1845 | 2.1890 | +0.0045 |
| Fixed Compute (600s) | Train Tokens | 7.63B | 7.65B | +0.3% |
| Fixed Compute (600s) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2871 | +0.0024 |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1733 | +0.0041 |
| Fixed Tokens (10B) | Wall-clock | 771s | 770s | -0.1% |
| — | Total Params | 17,039,360 | 17,039,360 | 0 |

## Analysis

Input RMSNorm applies `F.rms_norm` to the embedding output before it enters the first transformer block, adding zero parameters and negligible compute overhead.

The trick is **mechanically redundant** in this architecture, and the redundancy is the root cause of the regression. Three specific mechanisms are at play:

**1. Redundancy with block-level pre-norm.** Every `Block.forward` begins with `self.attn_norm(x)` -- an `RMSNorm` on the residual stream before the attention sub-layer. Block 0's first operation on the embedding output is therefore already `F.rms_norm`. Since `rms_norm(rms_norm(x))` is mathematically idempotent (the RMS of a unit-variance vector is 1, so the second normalization is the identity), the added input RMSNorm is a near-identity transformation for the attention computation. The attention sub-layer sees effectively the same activations with or without the trick.

**2. Loss of embedding magnitude from the residual stream.** The `attn_norm` in block 0 normalizes only the attention *branch*; the skip connection in the residual `x = x + attn(attn_norm(x))` adds the raw embedding back, preserving token-level magnitude information in the residual stream. Input RMSNorm normalizes the entire stream before block 0, so the skip connection now carries a unit-variance embedding instead of the raw one. Token embeddings naturally learn magnitude differences during training (common tokens tend to have larger L2 norms than rare tokens, creating a soft frequency prior), and stripping this signal from the residual path removes information the model can exploit.

**3. Insufficient depth to need stream normalization.** At 9 layers, the residual stream is short enough that embedding scale does not drift or accumulate significantly across residual additions. Block-level pre-attention RMSNorm in every layer already provides all the normalization this architecture needs. In very deep networks (100+ layers) where activation variance can compound across many skip connections, explicit input normalization could stabilize training. At shallow-to-moderate depths, the built-in per-block normalization is adequate.

**Verdict**: Slightly harmful and architecturally redundant. Input RMSNorm duplicates block 0's existing pre-attention RMSNorm -- a no-op for the attention branch -- while stripping potentially useful embedding magnitude information from the residual stream's skip connection. The observed regression (+0.0027 BPB under fixed compute, +0.0024 BPB under fixed tokens) is consistent across both regimes. Input RMSNorm should not be applied to architectures that already use block-level pre-normalization, especially at shallow-to-moderate depths.

## Files

- `train_gpt.py`: Training script (baseline + input RMSNorm modification)
- `input-rmsnorm.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
