# loop-share-mid: Middle Blocks Weight-Shared via Looping

## Summary

This experiment keeps 1 unique first block, weight-shares 3 middle blocks by repeating them 3 times, then passes through 5 unique remaining blocks. The shared middle blocks increase effective depth from 9 to 15 at the same parameter count (17.04M), trading FLOPs for depth. No skip connections are used.

## Ablation Story

This is part of a study on **which layers should be weight-shared (looped) for better effective depth**. The three variants test different sharing strategies:

| Variant | Block Definitions | Shared Pattern | Effective Depth | Params |
|---------|-------------------|----------------|-----------------|--------|
| baseline | 9 unique | none (all unique) | 9 | 17.04M |
| loop-share-all | 3 unique | all 3 blocks × 3 repeats | 9 | 6.03M |
| loop-share-first | 9 defs (5 unique + 4 shared) | first 4 blocks × 3 repeats | 17 | 17.04M |
| **loop-share-mid** | **9 defs (6 unique + 3 shared)** | **middle 3 blocks × 3 repeats** | **15** | **17.04M** |

loop-share-mid tests whether sharing the middle blocks (which typically learn mid-level abstractions) is more effective than sharing the first blocks. The hypothesis is that middle layers may benefit more from iterative refinement, while keeping first and last layers unique preserves input representation and output projection quality.

> **Note on fairness:** loop-share-mid has the same param count as baseline (17.04M) but ~1.67× the FLOPs due to higher effective depth (15 vs 9). In fixed-compute mode, it processes fewer tokens per step — this tests whether the depth benefit outweighs the throughput cost within a fixed time budget. In fixed-tokens mode, the comparison is at equal data with more compute per token.

## Architecture

| Property | Baseline | loop-share-mid |
|----------|----------|---------------|
| Block definitions | 9 unique | 6 unique + 3 shared via loop (1 enc + 3 loop + 5 dec) |
| Effective depth | 9 | 15 (1 + 3 × 3 + 5) |
| Skip connections | No | No |
| Parameter count | 17.04M | 17.04M |
| FLOPs ratio vs baseline | 1.0× | ~1.67× |

The forward pass applies 1 unique first block, then 3 shared middle blocks repeated 3 times, then 5 unique remaining blocks:

```
input -> Block0 -> [Block1 -> Block2 -> Block3] × 3 -> Block4 -> Block5 -> Block6 -> Block7 -> Block8 -> output
```

The unique first block (index 0) provides a stable input transformation. The 3 shared middle blocks (indices 1-3) are applied 3 times with weight sharing for iterative refinement. The 5 remaining blocks (indices 4-8) are unique.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_UNIQUE_ENCODER` | 1 | Number of unique first blocks (before the loop) |
| `NUM_LOOP_LAYERS` | 3 | Number of shared (looped) middle blocks |
| `NUM_LOOP_REPEATS` | 3 | Number of times the shared blocks are repeated |

All other hyperparameters (model_dim, num_heads, learning rates, etc.) are identical to the baseline.

## Experiments

| Name | Control | Description |
|------|---------|-------------|
| `loop-share-mid-fixed_time_10min` | Fixed compute, 600s wall-clock | Compare throughput × quality under equal compute |
| `loop-share-mid-fixed_tokens_10b` | Fixed tokens, 10B training tokens | Compare convergence under equal data |

## Results

| Regime | Metric | Baseline | loop-share-mid | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2938 | -0.0041 |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1846 | -0.0068 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 4.78B | -37.7% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 13,623 MiB | +5,234 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2662 | -0.0195 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1379 | -0.0330 |
| Fixed Tokens (10B) | Wall-clock | 772s | 1,242s | +60.9% |
| — | Total Params | 17.04M | 17.04M | 0 |

## Analysis

loop-share-mid shares 3 middle blocks via 3× looping (with 1 unique first block and 5 unique last blocks), increasing effective depth from 9 to 15 at the same parameter count. This is the best-performing loop variant in the ablation suite.

### Fixed-compute: modest improvement despite throughput loss

Under 10-minute fixed-compute, loop-share-mid processes 37.7% fewer tokens (4.78B vs 7.67B) due to the ~1.67× higher FLOPs per step. Despite this, it still achieves -0.004 BPB improvement — the only loop variant that wins under fixed-compute. The depth benefit outweighs the throughput cost, though the margin is thin.

### Fixed-tokens: strongest loop improvement

When given equal training tokens (10B), loop-share-mid achieves -0.020 BPB improvement — the best fixed-tokens result among all loop variants. The extra effective depth (15 vs 9) provides a clear quality gain when throughput is not the constraint.

### Why the middle?

Comparing with loop-share-first (+0.009 FC BPB, -0.015 FT BPB), sharing the middle layers is more effective than sharing the first layers. This suggests:

1. **First layers need uniqueness.** The initial block (index 0) handles raw token embeddings and learns specialized input representations that benefit from unique parameters.
2. **Middle layers benefit from iterative refinement.** The 3 shared middle blocks (indices 1-3) can improve representations through repeated application, acting as a refinement loop.
3. **Last layers need uniqueness.** The 5 final blocks (indices 4-8) project refined representations to output logits, requiring diverse learned transformations.

This "anchor-refine-project" pattern — unique first, shared middle, unique last — is the most effective weight-sharing strategy at this scale.

### Memory cost

The 13,623 MiB peak memory is 5,234 MiB higher than baseline but 1,750 MiB lower than loop-share-first (15,373 MiB). The lower effective depth (15 vs 17) translates to both better memory efficiency and better throughput.

## Modifications from Baseline

All changes are marked with `# trick: loop-share-mid` comments in `train_gpt.py`:

1. **Hyperparameters**: Added `num_unique_encoder`, `num_loop_layers`, and `num_loop_repeats`
2. **GPT.__init__**: Split blocks into `self.encoder_blocks` (1 unique), `self.loop_blocks` (3 shared), and `self.decoder_blocks` (5 unique)
3. **GPT.forward**: Unique first blocks, then looped iteration over shared blocks, then sequential iteration over remaining blocks
4. **Optimizer setup**: Collect params from all three block groups
5. **Logging**: Added loop configuration log line

## File Layout

```
exp/loop-share-mid/
  train_gpt.py          -- Trainer script with loop-share-mid modifications
  loop-share-mid.json   -- Experiment configuration
  README.md             -- This file
```
