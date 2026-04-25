# U-Net Skip — Learnable Skip Connections Ablation

## Method Overview

This experiment adds **U-Net skip connections** with learnable skip_weights. The first half of layers (encoder) store activations, and the second half (decoder) combines them in reverse order with per-skip learnable scaling.

### Motivation

- U-Net skip connections provide direct information highways from encoder to decoder
- They allow the model to preserve and route low-level features alongside high-level abstractions
- Learnable skip_weights let the model calibrate how much skip information to incorporate at each decoder layer
- This pattern is used in the parameter-golf speedrun baseline

## What This Ablation Tests

| Component | Baseline | U-Net Skip |
|---|---|---|
| Skip connections | None | Encoder→decoder with learnable weights |
| Additional params | 0 | `num_skips` per-dimension skip_weight vectors (shape: `num_skips × model_dim`) |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| Skip connections | None | U-Net pattern (4 skip pairs for 9 layers) |
| `skip_weight` init | — | 1.0 |
| Forward pass | Sequential blocks | Encoder stores, decoder retrieves with scaling |

## Results

| Regime | Metric | Baseline | U-Net Skip | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2803 | **-0.0176** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1618 | -0.0296 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.54B | -1.7% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2703 | **-0.0154** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1448 | -0.0261 |
| Fixed Tokens (10B) | Wall-clock | 772s | 781s | +1.2% |
| — | Total Params | 17,039,360 | 17,041,408 | +2,048 |

## Analysis

U-Net skip connections deliver a consistent -0.015 to -0.018 BPB improvement across both evaluation regimes, making this one of the strongest single tricks tested.

The mechanism is straightforward: each decoder layer receives a scaled copy of its paired encoder layer's output, creating direct information highways that bypass the intermediate residual stream. This addresses two weaknesses of sequential Transformers. First, fine-grained features from early layers are available to later layers without being diluted by successive residual updates. Second, gradient flow during backpropagation benefits from the shorter skip paths, improving optimization in the encoder half.

The overhead is negligible: 2,048 additional parameters (one per feature dimension per skip), zero memory increase, and only a 1.2% wall-clock slowdown. The slight throughput reduction (1.7% fewer tokens in fixed compute) is more than offset by the per-token quality gain, as evidenced by the lower BPB despite fewer tokens processed.

The learnable skip weights are initialized to 1.0, giving the model full skip throughput at the start of training and letting it dial back individual skips as needed. This means the architecture can gracefully regress toward baseline behavior for any skip that does not help, while retaining the ones that do.

**Verdict**: U-Net skip connections are a clear win. Multi-scale feature routing via encoder-decoder shortcuts provides meaningful quality gains with near-zero overhead.

## Files

- `train_gpt.py`: Training script (baseline + U-Net skip modifications)
- `unet-skip.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
