# Loop Depth Sampling

## Overview

This experiment adds per-micro-batch depth sampling to the pure looped transformer. Instead of always looping exactly N times for all sequences, the number of loops is sampled per training step from a Poisson distribution, following recommendations from Parcae and Loop-Think-Generalize.

**Single change from loop-pure**: Sample loop depth per micro-batch from `clip(Poisson(lambda=3), min=1, max=3)` instead of using a fixed depth of 3.

## Mechanism

Each training step:
1. Sample `R ~ clip(Poisson(lambda=3.0), min=1, max=3)` once
2. All sequences in that micro-batch loop R times through the shared blocks
3. This models the expectation over variable depths during training

At evaluation: always use `num_loop_repeats` (max depth = 3) for consistent, comparable metrics.

With Poisson(lambda=3) clipped to [1, 3], the expected depth distribution is approximately:
- P(R=1) ~ 0.05 (Poisson(3) gives 0 or 1, clipped to 1)
- P(R=2) ~ 0.22
- P(R=3) ~ 0.73 (Poisson(3) gives 3+, clipped to 3)
- Expected depth ~ 2.68

## Why This Matters

Fixed-depth training optimizes for one specific depth. The model learns representations that work well at exactly N loops but may degrade at other depths. Variable-depth training forces the model to produce useful intermediate representations at every loop iteration, making the model:

1. **More robust**: Performance degrades gracefully if you use fewer or more loops
2. **Depth-extrapolation capable**: Can generalize to more loops at inference time (test-time compute scaling)
3. **Less prone to loss spikes**: Better modeling of the expectation over variable depths reduces training instability

## Context from Prior Work

### Parcae (arXiv 2604.12946)

Per-sequence depth sampling within a micro-batch reduces variance-induced loss spikes, especially for variable-depth training. The key insight is that when all sequences use the same depth, the gradient signal is biased toward that specific depth. Sampling depth per micro-batch averages over the depth distribution, producing gradients that optimize for expected performance across depths.

### Loop-Think-Generalize (arXiv 2604.07822)

Dynamic iteration via `R ~ clip(Poisson(lambda), R_min, R_max)` per batch is critical for depth extrapolation. Models trained with fixed depth cannot effectively use additional loops at inference -- they overfit to the training depth. Dynamic depth during training creates representations that improve monotonically with more compute, enabling flexible inference-time compute allocation.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LOOP_LAYERS` | 3 | Number of shared transformer blocks |
| `NUM_LOOP_REPEATS` | 3 | Maximum loop repeats (also eval depth) |
| `LOOP_DEPTH_SAMPLING` | 1 | Enable (1) or disable (0) depth sampling |
| `LOOP_DEPTH_LAMBDA` | 3.0 | Poisson lambda for depth sampling |
| `LOOP_DEPTH_MIN` | 1 | Minimum loop repeats (floor of the clip) |

## Experiments

| Name | Depth Sampling | Control Mode | Notes |
|------|---------------|--------------|-------|
| `loop-depth-sampling-on-fixed_time_10min` | ON (Poisson) | 10 min wall-clock | Primary experiment |
| `loop-depth-sampling-on-fixed_tokens_10b` | ON (Poisson) | 10B tokens | Token-controlled comparison |
| `loop-depth-sampling-off-fixed_time_10min` | OFF (fixed=3) | 10 min wall-clock | Control: equivalent to loop-pure |
| `loop-depth-sampling-off-fixed_tokens_10b` | OFF (fixed=3) | 10B tokens | Control: equivalent to loop-pure |

The OFF experiments serve as an ablation control -- they are functionally identical to loop-pure but run through the same code path with `LOOP_DEPTH_SAMPLING=0`.

## Fixed-Compute Consideration

With Poisson sampling and the default parameters (lambda=3, min=1, max=3), the expected depth is approximately 2.68 rather than 3. This means each training step uses roughly 11% less compute than fixed-depth-3 on average, allowing more steps per wall-clock interval. This is a feature, not a bug: the model gets more gradient updates for the same compute budget while also learning depth-robust representations.

For the fixed-tokens comparison, both ON and OFF experiments see the same number of tokens, so the ON experiment will finish faster (less compute per step on average).

## Hypothesis

Depth sampling should improve robustness and reduce loss spikes compared to fixed-depth training. The model learns to produce good outputs regardless of loop count, enabling flexible inference-time compute allocation. On fixed-compute budgets, the ON experiment may achieve lower loss due to both the regularization effect of variable depth and the higher step count per wall-clock time.

## Architecture

Same as loop-pure: 3 shared transformer blocks with residual connections to the initial embedding (`x0`), looped up to `num_loop_repeats` times. No U-Net skip connections.

```
Input -> TokEmb -> RMSNorm -> x0
                          -> [Block1 -> Block2 -> Block3] x R (sampled depth)
                          -> FinalNorm -> Logits
```

Each `Block` contains:
- Residual mix: `mix[0] * x + mix[1] * x0` (learned mixing with original input)
- Attention with QK-norm, GQA, and RoPE
- ReLU-squared MLP
- Learnable attention and MLP scaling factors
