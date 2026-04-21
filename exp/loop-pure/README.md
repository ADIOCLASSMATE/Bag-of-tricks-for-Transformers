# loop-pure: Pure Looped Transformer (No U-Net, No Skip Connections)

## Summary

This experiment replaces the entire 9-layer U-Net architecture with a 3-layer looped block repeated 3 times. It is the most aggressive architectural change in the looped transformer family: no U-Net skip connections, no encoder/decoder split, just a shared block of layers iterated multiple times.

## Architecture

| Property | Baseline | loop-pure |
|----------|----------|-----------|
| Unique blocks | 9 | 3 |
| Effective depth | 9 | 9 (3 x 3) |
| U-Net skip connections | Yes (4 skip pairs) | No |
| Encoder/decoder split | Yes (4 + 5) | No |
| Parameter count | ~17M | ~5.7M |
| FLOPs (forward) | Baseline | Same effective depth, shared parameters |

The forward pass iterates through the same 3 blocks three times:

```
input -> [Block0 -> Block1 -> Block2] x 3 -> output
```

Each iteration applies the same weights, so the model must learn to perform different computations at each depth using the same parameters plus the evolving hidden state.

## Hypothesis

Weight sharing through looping may compensate for reduced parameters by making the same knowledge accessible at every depth. However, losing U-Net skip connections may hurt gradient flow and representation quality. This is the most radical test of the looped transformer hypothesis -- if pure looping works, it suggests that depth repetition can substitute for both parameter count and skip connections.

## Context from Prior Work

- **Parcae (2604.12946)**: A 770M looped model matched a 1.3B vanilla transformer, demonstrating that looped models can achieve parameter efficiency competitive with much larger models.
- **Loop-Think-Generalize (2604.07822)**: Weight sharing enables compositional generalization that vanilla transformers lack, suggesting loops may provide implicit algorithmic depth.

## Role as Base Model

This experiment also serves as the base model for the four training method ablation experiments:

| Alation | What it adds to loop-pure |
|---------|---------------------------|
| `loop-stable-A` | Stable repeated forward passes (orthogonal initialization, progressive depth) |
| `loop-zero-init` | Zero-initialize later loop iterations so early training is shallower |
| `loop-depth-sampling` | Randomly sample loop count during training for variable-depth inference |
| `loop-prelude-norm` | Add a normalization layer between loop iterations |

Comparing these four against loop-pure isolates the contribution of each training method on top of the shared looped architecture.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LOOP_LAYERS` | 3 | Number of unique layers in the shared loop block |
| `NUM_LOOP_REPEATS` | 3 | Number of times the loop block is repeated |
| `NUM_LAYERS` | 9 | Kept for config compatibility (not used for block count) |

All other hyperparameters (model_dim, num_heads, learning rates, etc.) are identical to the baseline.

## Experiments

| Name | Control | Description |
|------|---------|-------------|
| `loop-pure-fixed_time_10min` | Fixed compute, 600s wall-clock | Compare parameter efficiency under equal compute |
| `loop-pure-fixed_tokens_10b` | Fixed tokens, 10B training tokens | Compare convergence under equal data |

## Results

| Regime | Metric | Baseline | Loop Pure | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2194 | 1.3252 | +0.1058 |
| Fixed Compute (10 min) | Train Tokens | 6.954B | 7.200B | +3.5% |
| Fixed Compute (10 min) | Peak Memory | 10,246 MiB | 10,149 MiB | -97 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2118 | 1.3192 | +0.1074 |
| Fixed Tokens (10B) | Wall-clock | 832.8s | 814.9s | -2.1% |
| — | Total Params | 17.06M | 6.04M | -64.6% |

## Analysis

### Catastrophic Degradation

Loop Pure is the worst-performing experiment in the entire ablation suite, with a +0.106 BPB regression under fixed-compute and +0.107 under fixed-tokens. To put this in perspective, the gap between loop-pure (1.325) and baseline (1.219) is larger than the combined improvement of all positive tricks (GeGLU -0.021, KV Sharing -0.017, Sandwich Norm -0.006, etc.).

### Why Pure Looping Fails So Badly

1. **Loss of U-Net skip connections is devastating**: The baseline's U-Net skip connections provide direct information highways from encoder to decoder, allowing the model to preserve and route low-level features alongside high-level abstractions. Removing them forces all information to flow through the residual stream across 9 sequential blocks, which severely degrades the model's ability to combine multi-scale features.

2. **Parameter count collapse**: The model has only 6.04M parameters (64.6% fewer than baseline's 17.06M). Three unique blocks must do the work of nine — this is a fundamental capacity bottleneck. The same weights are applied at every depth, so the model cannot learn depth-specific transformations.

3. **No gradient shortcuts**: Without skip connections, gradients must flow through all 9 effective layers (3 blocks x 3 repeats) via the residual stream alone. This creates a long gradient path that is prone to degradation, especially since the same parameters are applied repeatedly.

4. **Representation collapse from weight sharing**: When the same 3 blocks are applied 3 times, each application transforms the hidden state using identical weights. The representations at different loop iterations differ only because the input has changed, not because the transformation has. This limits the representational diversity that the model can express at each depth.

### Faster Per-Step, But Pointless

Interestingly, loop-pure processes slightly more tokens than baseline under fixed-compute (7.20B vs 6.95B) and finishes faster under fixed-tokens (815s vs 833s). This is because the 3 unique blocks (6.04M params) involve less computation per forward pass than 9 unique blocks (17.06M params), and the effective depth equals the baseline (9). But this throughput advantage is meaningless — the per-token quality is so poor that more tokens cannot compensate.

### Implications for Training Method Ablations

Loop-pure serves as the base model for four training method experiments (loop-stable-A, loop-zero-init, loop-depth-sampling, loop-prelude-norm). The catastrophic regression of loop-pure means these experiments are testing whether training methods can partially recover the quality lost from removing U-Net skips and reducing parameter count — they are not testing whether training methods can improve an already-functional architecture.

### Comparison with Loop U-Net Variants

| Variant | Params | Effective Depth | Has Skips | Fixed-Compute BPB | Delta |
|---------|--------|----------------|-----------|-------------------|-------|
| Baseline | 17.06M | 9 | Yes (U-Net) | 1.2194 | — |
| Loop Pure | 6.04M | 9 | No | 1.3252 | +0.1058 |
| Loop U-Net Mid | 17.06M | 15 | Yes | 1.2160 | -0.0034 |
| Loop U-Net Full | 17.06M | 17 | Yes | 1.2225 | +0.0031 |

The contrast is stark: looping without skip connections is catastrophic, but looping with skip connections is at worst neutral and at best beneficial. The U-Net skip pattern is the critical enabling factor for looped transformers, not a mere convenience.

### Conclusion

Pure looped transformers without skip connections are not viable at this scale. The combination of parameter reduction (64.6%), loss of multi-scale feature routing, and gradient degradation makes loop-pure fundamentally weaker than the baseline. This finding strongly suggests that any practical looped transformer design must retain some form of skip connection or information highway — the U-Net pattern in loop-unet-mid provides this effectively.

## File Layout

```
exp/loop-pure/
  train_gpt.py      -- Trainer script with loop-pure modifications
  loop-pure.json    -- Experiment configuration
  README.md         -- This file
```

## Modifications from Baseline

All changes are marked with `# trick: loop-pure` comments in `train_gpt.py`:

1. **Hyperparameters**: Added `num_loop_layers` and `num_loop_repeats`
2. **GPT.__init__**: Replaced `self.blocks` (9 layers) with `self.loop_blocks` (3 layers), removed U-Net skip weight parameters
3. **GPT.forward**: Replaced encoder/decoder U-Net forward with pure looped iteration
4. **Optimizer setup**: Collect params from `loop_blocks` instead of `blocks`, removed skip_weights from scalar params
5. **Logging**: Added loop configuration log line
