# Loop U-Net Mid — Looped Middle Encoder in U-Net Skip Architecture

## Motivation

The baseline U-Net architecture splits its 9 layers into 4 encoder + 5 decoder with skip connections between corresponding layers. This creates a fixed depth: each token passes through exactly 9 transformer blocks regardless of complexity. Looped transformers (weight sharing across repeated iterations) offer a way to increase effective depth without increasing parameter count, but prior experiments (loop-pure, loop-depth-sampling) apply looping uniformly. This experiment tests a hybrid: loop only the middle of the encoder, keeping the U-Net skip architecture intact.

The key question: Can shared weights in a looped middle region provide the benefits of deeper processing (richer representations, more sequential refinement) while the U-Net skip connections stabilize what would otherwise be a degenerate optimization landscape?

## What This Ablation Tests

This experiment places a looped block in the middle of the encoder half of the U-Net. The first encoder layer is unique (providing a stable skip), then a 3-layer block is repeated 3 times, creating effective depth of 10 encoder passes. The decoder remains unchanged with 5 unique layers.

### Architecture Comparison

| Component | Baseline | Loop U-Net Mid |
|---|---|---|
| Unique encoder layers | 4 | 1 (layer 0) |
| Looped encoder layers | — | 3 layers x 3 repeats = 9 passes |
| Total encoder passes | 4 | 1 + 9 = 10 |
| Decoder layers | 5 | 5 (unchanged) |
| Effective depth | 9 | 15 (1 + 3x3 + 5) |
| Unique layers | 9 | 9 (1 + 3 + 5, same param count) |
| Skip connections | 4 (encoder i -> decoder i) | 4 (1 from unique + 3 from loop iterations) |

### Skip Connection Design

The baseline has `num_skip_weights = min(4, 5) = 4`. This experiment preserves 4 skip connections:

1. After unique encoder layer 0
2. After loop iteration 1 (3 shared blocks)
3. After loop iteration 2 (3 shared blocks)
4. After loop iteration 3 (3 shared blocks)

Each skip connection carries a learned weight (the `skip_weights` parameter), same as baseline. The decoder layers consume skips in reverse order (most recent first), matching baseline U-Net semantics.

### Parameter Count Analysis

| Parameter Group | Baseline | Loop U-Net Mid |
|---|---|---|
| Encoder blocks | 4 x Block | 1 x Block (unique) |
| Loop blocks | — | 3 x Block (shared, repeated) |
| Decoder blocks | 5 x Block | 5 x Block (unchanged) |
| Skip weights | 4 x model_dim | 4 x model_dim |
| **Total unique blocks** | **9** | **9** |
| **Total params** | **~17.06M** | **~17.06M** |

Parameter count is identical to baseline because we have the same number of unique blocks (1 + 3 + 5 = 9).

### Effective Depth Calculation

```
effective_depth = num_unique_encoder + (num_loop_layers * num_loop_repeats) + num_decoder_layers
                = 1 + (3 * 3) + 5
                = 15
```

This is a 1.67x increase in effective depth (15/9), meaning each forward pass performs ~1.67x the FLOPs of the baseline.

## Expected Impact

### Potential Benefits

1. **Sample efficiency from weight sharing**: The same 3 looped blocks see each token 3 times, which may allow them to refine representations more thoroughly. Weight sharing acts as an implicit regularizer, potentially improving generalization per parameter.

2. **U-Net stabilization**: The skip connections from each loop iteration provide the decoder with multi-scale features from different depths of the looped region. This may help stabilize training in the looped section by providing gradient shortcuts through the skips.

3. **Deeper processing at same param budget**: The effective depth of 15 vs 9 means more sequential computation per token, which could improve the model's ability to learn complex transformations.

### Potential Risks

1. **Throughput reduction**: 1.67x the FLOPs per forward pass means ~1.67x slower training steps. Under fixed-compute, the model will see ~60% as many tokens as baseline, which could offset any per-token quality gains.

2. **Gradient degradation in loop**: Repeated application of the same 3 blocks may lead to rank collapse or gradient vanishing/exploding, especially without explicit normalization between iterations. The skip connections may partially mitigate this, but the loop interior has no residual path back to x0 between iterations.

3. **Limited representation diversity**: The 3 looped blocks must learn representations that work well at every iteration. This constraint may limit their capacity compared to 3 independent blocks that can specialize for a specific depth.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `num_unique_encoder` | — | 1 |
| `num_loop_layers` | — | 3 |
| `num_loop_repeats` | — | 3 |
| Block organization | Single `blocks` ModuleList (9) | `encoder_blocks` (1) + `loop_blocks` (3) + `decoder_blocks` (5) |
| Forward pass | Sequential through 9 blocks | Encoder -> 3x loop -> Decoder with skip injection |
| Skip source | After each of first 4 encoder blocks | After unique encoder + after each loop iteration |

## FLOPs Estimate

Baseline FLOPs per token (approximate): `9 * block_flops`
Loop U-Net Mid FLOPs per token: `15 * block_flops` (1.67x baseline)

Under fixed-compute (10 min), expect ~60% of baseline's token throughput.
Under fixed-tokens (10B), expect ~1.67x wall-clock time vs baseline.

## Hypothesis

Weight sharing through looping may improve per-token quality (deeper processing, implicit regularization), but the increased sequential compute will significantly reduce throughput. The net effect under fixed-compute depends on whether the quality-per-token gain exceeds the token-throughput loss. The U-Net skip connections from each loop iteration may help stabilize the looped region, but the loop interior lacks direct residual connections to x0, which could cause gradient flow issues at higher repeat counts.

This experiment isolates the effect of looping the middle encoder while preserving U-Net skip semantics, complementing loop-pure (full model loop) and loop-unet-full (full U-Net with loop) ablations.

## Files

- `train_gpt.py`: Training script (baseline + loop-unet-mid modification)
- `loop-unet-mid.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
