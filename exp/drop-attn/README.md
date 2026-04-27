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
| Fixed Compute (600s) | **Val BPB** | 1.2938 | 1.2946 | +0.0008 |
| | Val Loss | 2.1845 | 2.1859 | +0.0014 |
| | Tokens | 7.63B | 8.27B | +8.4% |
| | Peak Memory | 8,389 MiB | 7,900 MiB | -489 MiB |
| Fixed Tokens (10B) | **Val BPB** | 1.2847 | 1.2890 | +0.0043 |
| | Val Loss | 2.1692 | 2.1764 | +0.0072 |
| | Wall-clock | 771s | 712s | -7.7% |
| | Peak Memory | 8,389 MiB | 7,900 MiB | -489 MiB |
| | Total Params | 17.04M | 17.04M | 0 |

## Analysis

Drop-attn removes the attention sublayer from the first transformer block, leaving it as an MLP-only layer. The intuition is that the first block operates on raw token embeddings where positional mixing is less critical than feature projection; deeper layers are better positioned to perform contextual aggregation.

The method causes a **small quality regression** in both evaluation regimes (+0.0008 FC, +0.0043 FT BPB). Removing attention from block 0 eliminates meaningful cross-position mixing that even the shallowest layer provides. Under fixed-compute, the extra throughput (+8.4% more tokens) nearly offsets the per-token quality loss, resulting in only a marginal regression. Under fixed-tokens, where the token budget is equal, the quality cost is more visible (+0.0043 BPB) -- the first attention layer does provide genuine representational value.

The memory reduction (-489 MiB) comes from eliminating the QKV and output activation tensors in block 0. The ~8% wall-clock speedup reflects the saved attention matmuls and reduced memory pressure from smaller activations.

**Key revision**: Contrary to the earlier hypothesis that the first attention layer is redundant, these corrected results show it makes a meaningful quality contribution. Removing it trades a small but real per-token quality loss for throughput and memory savings. The speedup and memory savings are genuine, but there is a detectable quality cost. For memory-constrained or throughput-critical deployments, the trade-off may still be acceptable, but it is not a free lunch.

## Files

- `train_gpt.py`: Baseline trainer with a static first-layer attention skip
- `drop-attn.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
