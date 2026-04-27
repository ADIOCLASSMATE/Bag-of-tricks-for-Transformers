# loop-share-all: All Blocks Weight-Shared via Looping

## Summary

This experiment weight-shares ALL transformer blocks by replacing the baseline's 9 unique blocks with 3 shared blocks repeated 3 times. It is the most aggressive form of weight sharing in the loop ablation family: every block application reuses the same parameters, and there are no skip connections.

## Motivation

### Why weight sharing via block looping?

In a standard transformer, each block contains its own parameters (attention weights, MLP weights, normalization parameters). The forward pass applies these blocks sequentially: input flows through block 0, then block 1, then block 2, and so on. Each block learns different features at different depths.

Block looping (also called weight tying across depth) changes this: instead of defining N unique blocks, define K shared blocks (K < N) and apply them repeatedly. The same attention and MLP weights are reused at multiple depths, each time operating on a different hidden state:

```
input -> [Block0 -> Block1 -> Block2] -> [Block0 -> Block1 -> Block2] -> [Block0 -> Block1 -> Block2] -> output
```

The key insight is that each application of the same block receives a *different* hidden state, so the same parameters can produce different outputs at different depths. The model must learn a single set of weights that works well across all depths, relying on the evolving hidden state (not unique parameters) to encode depth-specific information.

### The hypothesis: iterative refinement

The hypothesis behind weight sharing is iterative refinement: each pass through the shared blocks acts as a refinement step, progressively improving the representation. Rather than having dedicated early/middle/late blocks with specialized roles, the same computation is applied repeatedly like an unrolled recurrence. The hidden state evolves with each pass, so the model can perform different "operations" at different depths even with the same weights.

If this hypothesis holds, weight sharing could dramatically reduce parameter count without sacrificing quality. A model with K shared blocks repeated R times (K x R effective depth) might match a model with K x R unique blocks.

### Parameter efficiency

Weight sharing directly reduces the number of unique parameters. In this experiment, 9 unique blocks (17.04M parameters) are replaced by 3 shared blocks (6.03M parameters) -- a 64.6% reduction. Fewer unique parameters means:

- **Less GPU memory** for parameter storage
- **Potentially faster training** because the same blocks are computed repeatedly and their weights may stay in cache
- **Smaller model checkpoints**

The tradeoff is capacity: with only 3 blocks' worth of parameters, the model must encode all its knowledge into a much smaller parameter space.

### The extreme case: sharing ALL blocks

While loop-share-first and loop-share-mid share a *subset* of blocks (keeping some unique first/last blocks), loop-share-all shares **every** block. This is the most extreme form of weight sharing: there are no unique blocks at any depth -- every block application reuses the same small set of parameters.

The question this experiment asks is: what happens when capacity is *severely* constrained? Can iterative refinement compensate for having only 1/3 the parameters? Or does the capacity bottleneck dominate any potential gain from repeated application?

### The null hypothesis of the loop family

This experiment serves as the **null hypothesis** for the loop ablation family. If sharing ALL blocks produces no quality loss compared to the baseline, then depth is essentially "free": you can always trade unique parameters for repeated computation without harm. The three loop variants form a spectrum:

- **loop-share-all** tests the extreme (all blocks shared, severe capacity reduction)
- **loop-share-mid** tests moderate sharing (middle blocks shared, some unique blocks preserved)
- **loop-share-first** tests early sharing (first blocks shared, later blocks unique)

If loop-share-all fails (quality degrades sharply) while partial-sharing variants succeed, it tells us that *some* unique blocks are necessary -- weight sharing works as a complement to unique parameters, not a wholesale replacement.

## Ablation Story

This is part of a study on **which layers should be weight-shared (looped) for better effective depth**. The three variants test different sharing strategies:

| Variant | Block Definitions | Shared Pattern | Effective Depth | Params |
|---------|-------------------|----------------|-----------------|--------|
| baseline | 9 unique | none (all unique) | 9 | 17.04M |
| **loop-share-all** | **3 unique** | **all 3 blocks x 3 repeats** | **9** | **6.03M** |
| loop-share-first | 9 defs (5 unique + 4 shared) | first 4 blocks x 3 repeats | 17 | 17.04M |
| loop-share-mid | 9 defs (6 unique + 3 shared) | middle 3 blocks x 3 repeats | 15 | 17.04M |

loop-share-all tests the extreme case: same effective depth as baseline but with only 1/3 the parameters. This isolates whether weight sharing can compensate for reduced capacity through repeated application.

> **Note on fairness:** loop-share-all has 6.03M params vs baseline's 17.04M. Under fixed-compute mode, it processes roughly the same number of tokens per step (same effective depth, shared params are cheaper). Under fixed-tokens mode, the comparison is at equal data but vastly different capacity.
>
> **Note on parameter constraints:** The 64.6% parameter reduction (17.04M to 6.03M) makes this fundamentally a parameter-constrained comparison, not just a depth study. Unlike loop-share-first and loop-share-mid (which maintain the full 17.04M parameter budget and increase effective depth), loop-share-all trades parameters for repeated computation at the same effective depth. Any quality gap must be interpreted in light of this severe capacity reduction -- the model has fewer degrees of freedom to fit the training distribution.

## Architecture

| Property | Baseline | loop-share-all |
|----------|----------|----------------|
| Block definitions | 9 unique | 3 unique |
| Effective depth | 9 | 9 (3 x 3) |
| Skip connections | No | No |
| Parameter count | 17.04M | 6.03M |
| FLOPs ratio vs baseline | 1.0x | ~1.0x |

The forward pass iterates through the same 3 blocks three times:

```
input -> [Block0 -> Block1 -> Block2] x 3 -> output
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
| `loop-share-all-fixed_time_10min` | Fixed compute, 600s wall-clock | Compare throughput x quality under equal compute |
| `loop-share-all-fixed_tokens_10b` | Fixed tokens, 10B training tokens | Compare convergence under equal data |

## Results

| Regime | Metric | Baseline | loop-share-all | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.3687 | +0.0749 |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.3110 | +0.1265 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.89B | +3.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,294 MiB | -95 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.3612 | +0.0765 |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.2984 | +0.1292 |
| Fixed Tokens (10B) | Wall-clock | 771s | 747s | -3.1% |
| -- | Total Params | 17.04M | 6.03M | -64.6% |

## Analysis

loop-share-all replaces 9 unique blocks with 3 shared blocks iterated 3 times, cutting parameters by 64.6%. The result is the largest regression in the loop ablation suite (+0.075 BPB fixed-compute, +0.077 BPB fixed-tokens).

### Capacity collapse

Three shared blocks (6.03M params) must serve the role of nine unique blocks (17.04M params). Weight sharing means the same transformation is applied at every depth -- the model cannot learn depth-specific features, only depth-dependent behavior through the evolving hidden state. This capacity bottleneck is the dominant failure mode.

### Throughput gain does not compensate

loop-share-all processes slightly more tokens under fixed-compute (7.89B vs 7.63B) and finishes faster under fixed-tokens (747s vs 771s) because the 3 unique blocks require less computation per forward pass. This throughput advantage is irrelevant -- per-token quality is poor enough that more tokens cannot close the gap.

### Implications

Weight sharing all blocks is not viable at this scale. The 64.6% parameter reduction fundamentally limits what the model can learn, regardless of how many times the shared blocks are iterated. Partial sharing strategies (loop-share-first, loop-share-mid) that maintain the full parameter budget fare much better.
