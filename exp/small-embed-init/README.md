# Small-Embed-Init — Reduced Embedding Initialization Scale

## Method Overview

This experiment initializes the tied embedding with a **small standard deviation (0.005)** instead of PyTorch's default `nn.Embedding` initialization. This keeps initial logit magnitudes small, preventing extreme gradients at the output layer.

### Motivation

- Default `nn.Embedding` init uses `nn.init.normal_(std=1.0)`, which can produce large embedding magnitudes, leading to extreme logits and high initial loss
- Small initialization ensures the model starts with modest logit magnitudes, producing smoother early training
- This technique is commonly used in large-scale language model training

## What This Ablation Tests

| Component | Baseline | Small-Embed-Init |
|---|---|---|
| Embedding init std | Default (std=1.0) | 0.005 |
| Initial logit scale | Potentially large | Small |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `tied_embed_init_std` | Default | 0.005 |

## Results

| Regime | Metric | Baseline | Small-Embed-Init | Delta |
|---|---|---|---|---|
| Fixed Compute (600s) | Val BPB | 1.2979 | 1.2915 | -0.0064 |
| Fixed Compute (600s) | Val Loss | 2.1914 | 2.1806 | -0.0108 |
| Fixed Compute (600s) | Train Tokens | 7.67B | 7.65B | -0.3% |
| Fixed Compute (600s) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2848 | -0.0009 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1693 | -0.0016 |
| Fixed Tokens (10B) | Wall-clock | 772s | 774s | +0.3% |
| — | Total Params | 17,039,360 | 17,039,360 | 0 |

## Analysis

Small-embed-init improves fixed-compute BPB by -0.0064 but only -0.0009 under fixed-tokens. The gap between the two regimes reveals that the benefit is concentrated in early training: initializing embeddings at std=0.005 (vs the default std=1.0, a 200x reduction) keeps initial logit magnitudes small, which reduces extreme gradients at the output layer and accelerates the first phase of convergence. Once the model has seen enough tokens (10B), the head start is largely absorbed and both models converge to nearly the same point.

The method adds zero cost — identical parameter count, memory, and wall-clock time. Under a compute budget, the faster early convergence translates directly into a better checkpoint at the deadline. Under a token budget, the advantage is negligible because the optimizer has sufficient steps to close the initialization gap.

**Verdict**: Worth adopting as a default. It is free, simple to implement, and provides a consistent (if small) improvement when training time is the binding constraint.

## Files

- `train_gpt.py`: Training script (baseline + small-embed-init modification)
- `small-embed-init.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
