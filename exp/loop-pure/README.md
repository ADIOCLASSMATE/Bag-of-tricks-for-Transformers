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
