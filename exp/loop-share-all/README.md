# loop-share-all: All Blocks Weight-Shared via Looping

## Summary

This experiment weight-shares ALL transformer blocks by replacing the baseline's 9 unique blocks with 3 shared blocks repeated 3 times. It is the most aggressive form of weight sharing in the loop ablation family: every block application reuses the same parameters, and there are no skip connections.

## Ablation Story

This is part of a study on **which layers should be weight-shared (looped) for better effective depth**. The three variants test different sharing strategies:

| Variant | Block Definitions | Shared Pattern | Effective Depth | Params |
|---------|-------------------|----------------|-----------------|--------|
| baseline | 9 unique | none (all unique) | 9 | 17.04M |
| **loop-share-all** | **3 unique** | **all 3 blocks × 3 repeats** | **9** | **6.03M** |
| loop-share-first | 9 defs (5 unique + 4 shared) | first 4 blocks × 3 repeats | 17 | 17.04M |
| loop-share-mid | 9 defs (6 unique + 3 shared) | middle 3 blocks × 3 repeats | 15 | 17.04M |

loop-share-all tests the extreme case: same effective depth as baseline but with only 1/3 the parameters. This isolates whether weight sharing can compensate for reduced capacity through repeated application.

> **Note on fairness:** loop-share-all has 6.03M params vs baseline's 17.04M. Under fixed-compute mode, it processes roughly the same number of tokens per step (same effective depth, shared params are cheaper). Under fixed-tokens mode, the comparison is at equal data but vastly different capacity.

## Architecture

| Property | Baseline | loop-share-all |
|----------|----------|----------------|
| Block definitions | 9 unique | 3 unique |
| Effective depth | 9 | 9 (3 × 3) |
| Skip connections | No | No |
| Parameter count | 17.04M | 6.03M |
| FLOPs ratio vs baseline | 1.0× | ~1.0× |

The forward pass iterates through the same 3 blocks three times:

```
input -> [Block0 -> Block1 -> Block2] × 3 -> output
```

Each iteration applies the same weights, so the model must learn to perform different computations at each depth using the same parameters plus the evolving hidden state.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LOOP_LAYERS` | 3 | Number of unique layers in the shared block |
| `NUM_LOOP_REPEATS` | 3 | Number of times the block is repeated |

All other hyperparameters (model_dim, num_heads, learning rates, etc.) are identical to the baseline.

## Experiments

| Name | Control | Description |
|------|---------|-------------|
| `loop-share-all-fixed_time_10min` | Fixed compute, 600s wall-clock | Compare throughput × quality under equal compute |
| `loop-share-all-fixed_tokens_10b` | Fixed tokens, 10B training tokens | Compare convergence under equal data |

## Results

| Regime | Metric | Baseline | loop-share-all | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.3685 | +0.0706 |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.3107 | +0.1193 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.86B | +2.5% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,294 MiB | -95 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.3609 | +0.0752 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.2978 | +0.1269 |
| Fixed Tokens (10B) | Wall-clock | 772s | 750s | -2.9% |
| — | Total Params | 17.04M | 6.03M | -64.6% |

## Analysis

loop-share-all replaces 9 unique blocks with 3 shared blocks iterated 3 times, cutting parameters by 64.6%. The result is the largest regression in the loop ablation suite (+0.071 BPB fixed-compute, +0.075 BPB fixed-tokens).

### Capacity collapse

Three shared blocks (6.03M params) must serve the role of nine unique blocks (17.04M params). Weight sharing means the same transformation is applied at every depth — the model cannot learn depth-specific features, only depth-dependent behavior through the evolving hidden state. This capacity bottleneck is the dominant failure mode.

### Throughput gain does not compensate

loop-share-all processes slightly more tokens under fixed-compute (7.86B vs 7.67B) and finishes faster under fixed-tokens (750s vs 772s) because the 3 unique blocks require less computation per forward pass. This throughput advantage is irrelevant — per-token quality is poor enough that more tokens cannot close the gap.

### Implications

Weight sharing all blocks is not viable at this scale. The 64.6% parameter reduction fundamentally limits what the model can learn, regardless of how many times the shared blocks are iterated. Partial sharing strategies (loop-share-first, loop-share-mid) that maintain the full parameter budget fare much better.

## Modifications from Baseline

All changes are marked with `# trick: loop-share-all` comments in `train_gpt.py`:

1. **Hyperparameters**: Added `num_loop_layers` and `num_loop_repeats`
2. **GPT.__init__**: Replaced `self.blocks` (9 layers) with `self.loop_blocks` (3 layers)
3. **GPT.forward**: Replaced sequential block iteration with looped iteration
4. **Optimizer setup**: Collect params from `loop_blocks` instead of `blocks`
5. **Logging**: Added loop configuration log line

## File Layout

```
exp/loop-share-all/
  train_gpt.py          -- Trainer script with loop-share-all modifications
  loop-share-all.json   -- Experiment configuration
  README.md             -- This file
```
