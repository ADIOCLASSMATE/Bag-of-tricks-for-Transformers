# loop-share-first: First N Blocks Weight-Shared via Looping

## Summary

This experiment weight-shares the first 4 transformer blocks by repeating them 3 times, then passes through 5 unique remaining blocks. The shared blocks increase effective depth from 9 to 17 at the same parameter count (17.04M), trading FLOPs for depth. No skip connections are used.

## Motivation

Early layers in transformers typically learn low-level features -- token-level patterns, local syntax, and surface-form regularities. These low-level features may be more amenable to weight sharing than mid- or late-layer representations: the same basic operations (attend to neighboring tokens, detect frequent n-gram patterns) apply at multiple depths.

This experiment tests the **"repeated refinement" hypothesis**: can the same early-layer parameters, applied multiple times, provide better representations than unique parameters? By sharing the first 4 blocks and repeating them 3 times, the model gains 8 extra effective layers (9 to 17) at zero parameter cost -- each token receives more computation. The trade-off is throughput: deeper effective depth means fewer tokens processed per second under a fixed-time budget.

The experiment also serves as an ablation contrast to loop-share-mid (which shares middle layers). Early layers handle raw token embeddings and must learn input-specific transformations. Sharing those layers forces the same parameters to serve both the initial embedding-to-hidden projection and subsequent refinement, which may degrade input representation quality. Middle layers, by contrast, operate on already-structured hidden states and may be more tolerant of repeated parameters. Comparing loop-share-first against loop-share-mid isolates this input-specialization effect.

## Ablation Story

This is part of a study on **which layers should be weight-shared (looped) for better effective depth**. The three variants test different sharing strategies:

| Variant | Block Definitions | Shared Pattern | Effective Depth | Params |
|---------|-------------------|----------------|-----------------|--------|
| baseline | 9 unique | none (all unique) | 9 | 17.04M |
| loop-share-all | 3 unique | all 3 blocks x 3 repeats | 9 | 6.03M |
| **loop-share-first** | **9 defs (5 unique + 4 shared)** | **first 4 blocks x 3 repeats** | **17** | **17.04M** |
| loop-share-mid | 9 defs (6 unique + 3 shared) | middle 3 blocks x 3 repeats | 15 | 17.04M |

loop-share-first tests whether sharing the first blocks (which typically learn low-level features) and repeating them is more effective than the baseline's unique-block approach. The hypothesis is that early-layer features may be more amenable to weight sharing.

> **Note on fairness:** loop-share-first has the same param count as baseline (17.04M) but ~1.89x the FLOPs due to higher effective depth (17 vs 9). In fixed-compute mode, it processes fewer tokens per step -- this tests whether the depth benefit outweighs the throughput cost within a fixed time budget. In fixed-tokens mode, the comparison is at equal data with more compute per token.

## Architecture

| Property | Baseline | loop-share-first |
|----------|----------|-----------------|
| Block definitions | 9 unique | 9 defs (5 unique + 4 shared) |
| Effective depth | 9 | 17 (4 x 3 + 5) |
| Skip connections | No | No |
| Parameter count | 17.04M | 17.04M |
| FLOPs ratio vs baseline | 1.0x | ~1.89x |

The forward pass applies the first 4 shared blocks 3 times, then 5 unique blocks:

```
input -> [Block0 -> Block1 -> Block2 -> Block3] x 3 -> Block4 -> Block5 -> Block6 -> Block7 -> Block8 -> output
```

The 4 shared blocks (indices 0-3) are applied 3 times with weight sharing. The 5 remaining blocks (indices 4-8) are unique.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LOOP_LAYERS` | 4 | Number of shared (looped) blocks at the start |
| `NUM_LOOP_REPEATS` | 3 | Number of times the shared blocks are repeated |

All other hyperparameters (model_dim, num_heads, learning rates, etc.) are identical to the baseline.

## Experiments

| Name | Control | Description |
|------|---------|-------------|
| `loop-share-first-fixed_time_10min` | Fixed compute, 600s wall-clock | Compare throughput x quality under equal compute |
| `loop-share-first-fixed_tokens_10b` | Fixed tokens, 10B training tokens | Compare convergence under equal data |

## Results

| Regime | Metric | Baseline | loop-share-first | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.3062 | +0.0124 |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.2054 | +0.0209 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 4.25B | -44.3% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 15,373 MiB | +6,984 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2682 | -0.0165 |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1413 | -0.0279 |
| Fixed Tokens (10B) | Wall-clock | 771s | 1,394s | +80.8% |
| -- | Total Params | 17.04M | 17.04M | 0 |

## Analysis

loop-share-first shares the first 4 blocks via 3x looping, increasing effective depth from 9 to 17 at the same parameter count. The results reveal a clear compute-quality tradeoff:

### Fixed-compute: depth doesn't justify the throughput cost

Under 10-minute fixed-compute, loop-share-first processes 44.3% fewer tokens (4.25B vs 7.63B) due to the ~1.89x higher FLOPs per step. The per-token quality improvement is insufficient to compensate -- the final BPB is slightly worse (+0.012). The depth benefit is real but modest, and cannot overcome the throughput loss within a fixed time budget.

### Fixed-tokens: depth wins with sufficient data

When given equal training tokens (10B), loop-share-first achieves -0.017 BPB improvement. The extra effective depth (17 vs 9) helps when throughput is not the bottleneck -- each token gets more computation, and the model learns better per-token representations.

### Memory cost

The 15,373 MiB peak memory is 6,984 MiB higher than baseline, reflecting the deeper effective computation graph. This limits scalability at larger model sizes.

### Sharing the first layers works, but is not optimal

Comparing with loop-share-mid (+0.001 FC BPB, -0.023 FT BPB), sharing the middle layers while keeping first and last unique is more effective than sharing the first layers. This suggests that early layers benefit more from having unique parameters (specialized input processing), while middle layers are more amenable to iterative refinement via weight sharing. The input embedding space demands input-specific transformation that weight sharing cannot provide, since the first shared block must both project raw token embeddings into hidden space and contribute to subsequent layer refinement -- two distinct roles that benefit from separate parameters.
