# Drop-Attn — Remove Attention from the First Transformer Block

## Method Overview

This experiment disables the attention sublayer in the **first transformer block only**. All later blocks keep the standard attention path.

The implementation uses a static `layer_idx == 0` branch inside `Block.forward`, so the first block returns a zero attention contribution and only keeps the MLP path. To keep the ablation clean, the first block's entire attention module (`attn`) is frozen and excluded from optimizer groups. (RMSNorm has no learnable weight, so `attn_norm` requires no freezing.)

### Origin

From **nanogpt-speedrun track_1_short / 2025-09-21_DropAttn**. The reported idea is that the earliest layer may not need a full attention computation, and removing it can simplify the shallow part of the network.

### Motivation

- The first layer operates directly on token embeddings, where local mixing may already be sufficient
- Removing one attention sublayer is a clean architectural ablation with minimal code change
- The condition is static per layer and compile-friendly

## Impact on Training

- **Parameters**: No change
- **Throughput**: Slightly better, because one block skips its attention path
- **Memory**: Slightly lower attention activation cost in layer 0
- **Optimization**: The first block becomes MLP-only, while deeper blocks remain unchanged

## Key Differences from Baseline

| Component | Baseline | drop-attn |
|---|---|---|
| Block 0 attention | enabled | **disabled** |
| Blocks 1-8 attention | enabled | enabled |
| Code change | — | adds `layer_idx`, static skip in block 0, and freezes/excludes `blocks.0.attn.*` |
| Everything else | identical | identical |

## Results

| Regime | Metric | Baseline | drop-attn | Delta |
|---|---|---|---|---|
| Fixed Compute (600s) | **Val BPB** | 1.2979 | 1.2894 | -0.0085 |
| | Val Loss | 2.1914 | 2.1772 | -0.0142 |
| | Tokens | 7.67B | 8.21B | +7.0% |
| | Peak Memory | 8,389 MiB | 7,901 MiB | -488 MiB |
| Fixed Tokens (10B) | **Val BPB** | 1.2857 | 1.2840 | -0.0017 |
| | Val Loss | 2.1709 | 2.1679 | -0.0030 |
| | Wall-clock | 772s | 718s | -7.0% |
| | Peak Memory | 8,389 MiB | 7,901 MiB | -488 MiB |
| | Total Params | 17.04M | 17.04M | 0 |

## Analysis

Drop-attn removes the attention sublayer from the first transformer block, leaving it as an MLP-only layer. The intuition is that the first block operates on raw token embeddings where positional mixing is less critical than feature projection; deeper layers are better positioned to perform contextual aggregation.

The method delivers a modest but consistent improvement across both evaluation regimes. Under fixed compute, the gain (-0.009 BPB) is a compound effect: skipping attention in block 0 frees compute that is reallocated to more tokens (+7%), and the remaining layers compensate for the missing attention with no per-token quality degradation. Under fixed tokens, the improvement shrinks to -0.002 BPB, indicating that the first attention layer does contribute marginal representational value when data is abundant, but that contribution is small relative to its cost.

The 488 MiB memory reduction comes entirely from eliminating the QKV and output activation tensors in block 0. The 7% wall-clock speedup reflects the saved attention matmuls and reduced memory pressure from smaller activations.

**Key insight**: The first attention layer is redundant enough that removing it is a net win under compute constraints, and at worst neutral under data-abundant conditions. This suggests that shallow layers in small transformers rely more on per-position feature transformation than on cross-position mixing, which aligns with the observation that early representations are still close to the embedding space and lack the contextual depth needed for meaningful attention patterns.

## Files

- `train_gpt.py`: Baseline trainer with a static first-layer attention skip
- `drop-attn.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
