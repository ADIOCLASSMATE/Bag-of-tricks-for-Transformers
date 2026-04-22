# Untie-Embed — Separate Embedding and LM-Head Weights

## Method Overview

This experiment **unties the embedding and language model head weights**, giving each its own independent parameter matrix and learning rate. In the baseline, a single weight matrix is shared between the token embedding layer and the output projection (lm_head). This experiment sets `tie_embeddings=0` and uses the original record's optimizer split: `embed_lr=0.3` for the embedding and `head_lr=0.002` for the lm_head.

By default, weight tying forces the embedding and lm_head to learn a single representation that must serve two purposes: (1) mapping token IDs to dense vectors, and (2) projecting hidden states back to logits. Untying allows each layer to specialize.

### Origin

From **nanogpt 2024-11-03_UntieEmbed**. The original record used `lr=0.3` for the token embedding and `lr=0.002` for the untied lm_head, so this experiment now matches that source configuration instead of using a retuned variant.

### Motivation

- The embedding layer and lm_head have fundamentally different roles — tying them is a parameter-efficiency hack that sacrifices expressiveness
- Per-component learning rates allow each layer to be optimized at its natural speed
- The large gap between `embed_lr=0.3` and `head_lr=0.002` suggests the embedding benefits from much faster updates while the head needs stability

## Impact on Training

- **Parameters**: Increases parameter count — the lm_head gets its own `vocab_size × model_dim` matrix instead of sharing with the embedding
- **Throughput**: Minimal impact — one extra matrix multiply, but vocab_size=1024 is small
- **Memory**: Slight increase due to the additional weight matrix and its optimizer states
- **Gradient flow**: Embedding and lm_head gradients are now independent, preventing conflicting gradient signals

## Key Differences from Baseline

| Component | baseline-sp1024 | untie-embed |
|---|---|---|
| `tie_embeddings` | 1 (default) | **0** |
| `embed_lr` | (default) | **0.3** |
| `head_lr` | (default) | **0.002** |
| Code change | — | **None (manifest only)** |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | untie-embed | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | untie-embed | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: Weight tying was constraining the model, and the specialized learning rates allow both the embedding and lm_head to converge to better optima. The high embed_lr (0.6) is critical for fast embedding adaptation.
- **If BPB is unchanged**: At vocab_size=1024, the embedding and lm_head matrices are small enough that tying doesn't meaningfully constrain the model. The benefit may only appear at larger vocab sizes.
- **If BPB worsens**: The additional parameters may lead to overfitting at this model scale, or the chosen learning rates (0.6 / 0.008) may not be optimal for this specific architecture configuration.

## Files

- `train_gpt.py`: Identical copy of baseline training script (no modifications)
- `untie-embed.json`: Experiment manifest (`tie_embeddings: 0, embed_lr: 0.3, head_lr: 0.002`)
- `logs/`: Training output (automatically generated)
