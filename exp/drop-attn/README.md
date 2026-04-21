# Drop-Attn — Remove Attention from the First Transformer Block

## Method Overview

This experiment disables the attention sublayer in the **first transformer block only**. All later blocks keep the standard attention path.

The implementation uses a static `layer_idx == 0` branch inside `Block.forward`, so the first block returns a zero attention contribution and only keeps the MLP path. To keep the ablation clean, the first block's entire attention branch (`attn_norm` + `attn`) is frozen and excluded from optimizer groups.

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

| Component | baseline-sp1024 | drop-attn |
|---|---|---|
| Block 0 attention | enabled | **disabled** |
| Blocks 1-8 attention | enabled | enabled |
| Code change | — | adds `layer_idx`, static skip in block 0, and freezes/excludes `blocks.0.attn_norm.*` plus `blocks.0.attn.*` |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | drop-attn | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | drop-attn | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: The first layer attention was unnecessary or even noisy, and removing it made optimization cleaner.
- **If BPB is unchanged**: The first layer attention is redundant at this scale, but not harmful enough to matter.
- **If BPB worsens**: The first attention layer is still useful for early token-to-token routing, and removing it cuts too much capacity.

## Files

- `train_gpt.py`: Baseline trainer with a static first-layer attention skip
- `drop-attn.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
