# Loop U-Net Full — Full Encoder Looping with U-Net Skip Connections

## Motivation

Universal Transformers introduced the idea of looping (repeating) transformer layers to increase effective depth without adding parameters. However, most looped architectures discard the U-Net skip connection pattern used in the baseline, which has proven effective for gradient flow and information routing. This experiment asks: **can looped computation coexist with U-Net skip connections?**

By looping the entire encoder half of the U-Net, we test whether the skip connections — which connect encoder intermediate states to decoder layers — remain beneficial when the encoder states come from different iterations of the same shared parameters rather than from independent layers. This is a stronger test of the U-Net pattern than partial looping, because every encoder pass reuses the same weights, making each skip connection carry a different "view" of the same computation.

## What This Ablation Tests

This experiment replaces the baseline's 4 independent encoder layers with 4 looped encoder layers repeated 3 times, while keeping all 5 decoder layers unique. The skip connections are stored after each complete loop iteration, giving 3 skip weights that connect to the first 3 decoder layers.

### Architecture Comparison

| Component | Baseline | Loop U-Net Full |
|---|---|---|
| Encoder layers | 4 independent | **4 looped x 3 repeats** (shared params) |
| Decoder layers | 5 independent | 5 independent (unchanged) |
| Effective depth | 4 + 5 = 9 | **4x3 + 5 = 17** |
| Unique block count | 9 | **9** (4 loop + 5 decoder) |
| Skip connections | 4 skip_weights (encoder-decoder pairs) | **3 skip_weights** (one per loop iteration) |
| Decoder layers with skips | All 4 get skips | **First 3 get skips, last 2 don't** |

### Skip Connection Design

In the baseline, each of the 4 encoder layers produces one skip to its paired decoder layer. In this experiment, the looped encoder produces one skip after each complete iteration (all 4 blocks pass):

- Loop iteration 0 output -> skip_weights[0] -> decoder layer 0
- Loop iteration 1 output -> skip_weights[1] -> decoder layer 1
- Loop iteration 2 output -> skip_weights[2] -> decoder layer 2
- Decoder layers 3 and 4 receive no skip connections

This asymmetry (3 skips for 5 decoder layers vs. 4 skips for 5 decoder layers in baseline) is a necessary consequence of the loop design. Each skip carries the output of the full 4-block encoder pass at a different loop iteration, providing progressively deeper representations.

## Expected Impact

1. **Increased effective depth at constant parameters**: The looped encoder provides 12 effective encoder passes (4 layers x 3 repeats) using only 4 unique blocks. This tests whether depth alone — without proportionally more parameters — improves model quality. The effective depth nearly doubles (17 vs. 9), so if depth is the primary bottleneck, we should see improvement.

2. **Gradient flow through looped computation**: Each loop iteration applies the same 4 blocks sequentially. Gradients must flow backward through all 3 iterations, which could lead to gradient degradation or instability. The skip connections from each iteration may help by providing shorter gradient paths, but they only connect to 3 of the 5 decoder layers.

3. **Skip connection quality from repeated computation**: In the baseline, each skip comes from a different set of learned parameters. Here, each skip comes from a different iteration of the same parameters applied to progressively transformed inputs. The later iterations' skips carry more processed information but may also carry more accumulated noise or saturation.

4. **Reduced skip coverage**: Only 3 of 5 decoder layers receive skip connections (vs. 4 of 5 in baseline). The last 2 decoder layers operate without the U-Net shortcut, which could reduce their effectiveness and make the model more reliant on the residual stream alone.

5. **Sequential compute overhead**: The ~1.89x increase in effective depth means ~1.89x more FLOPs per forward pass. Under fixed-compute, the model will see fewer training tokens per wall-clock second, so the per-token quality improvement must overcome this throughput disadvantage.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `num_loop_layers` | -- | **4** |
| `num_loop_repeats` | -- | **3** |
| Encoder architecture | 4 independent blocks | **4 shared blocks x 3 repeats** |
| `num_skip_weights` | 4 | **3** |
| Decoder layers with skips | 4 of 5 | **3 of 5** |
| FLOPs per forward pass | 1.0x (reference) | **~1.89x** |

## Parameter Count Analysis

The parameter count remains identical to the baseline:

- **Encoder blocks**: 4 unique blocks (same as baseline's 4 encoder blocks)
- **Decoder blocks**: 5 unique blocks (same as baseline's 5 decoder blocks)
- **Skip weights**: 3 vectors of size `model_dim=512` (vs. 4 in baseline) — negligible difference
- **Total unique blocks**: 9 (unchanged)
- **Total parameters**: ~17.06M (essentially identical to baseline; the 1 fewer skip_weight saves 512 params, which is <0.003%)

The key trade-off: same parameter budget, but 1.89x the compute per forward pass.

## FLOPs Comparison

For a single forward pass with sequence length S and model dimension D:

| Component | Baseline FLOPs | Loop U-Net Full FLOPs | Ratio |
|---|---|---|---|
| Encoder blocks | 4 x block_cost | 4 x 3 x block_cost | 3.0x |
| Decoder blocks | 5 x block_cost | 5 x block_cost | 1.0x |
| Total blocks | 9 x block_cost | 17 x block_cost | **1.89x** |
| Skip additions | 4 x S x D | 3 x S x D | 0.75x |

The ~1.89x FLOP increase means:
- **Fixed-compute regime**: ~47% fewer training tokens seen (9/17 of baseline throughput)
- **Fixed-tokens regime**: ~89% more wall-clock time per training step

## Hypothesis

Full encoder looping tests whether the U-Net skip pattern is compatible with looped computation. The skip connections from different loop iterations may help stabilize gradient flow through the repeated encoder, potentially making the 1.89x depth increase worthwhile. However, the reduced skip coverage (3 vs. 4 skip connections) and the fact that all encoder passes share parameters (limiting representational diversity) could offset the depth advantage. If this experiment underperforms, it suggests that the U-Net pattern relies on independent encoder representations, not just deeper computation.

## Files

- `train_gpt.py`: Training script (baseline + full encoder looping modification)
- `loop-unet-full.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
