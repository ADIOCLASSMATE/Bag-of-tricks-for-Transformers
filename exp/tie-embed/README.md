# Tie-Embed — Weight Tying for Embedding and LM-Head

## Method Overview

This experiment **ties the embedding and language model head weights**, forcing them to share a single weight matrix. This is the baseline parameter-efficiency approach where `tie_embeddings=1`. The token embedding layer (`tok_emb`) and the output projection (`lm_head`) use the same weight matrix, reducing parameters while imposing a shared optimization constraint.

**This is the control/baseline configuration.** The `tie_embeddings` flag is set to 1, and the embedding is trained with `tied_embed_lr=0.05`. This configuration serves as the reference point against which other embedding variants are compared.

Weight tying forces the embedding and lm_head to share a single weight matrix, creating a unified optimization objective: the shared matrix must simultaneously serve as a good input representation and a good output classifier. This parameter-efficiency technique trades expressiveness for reduced model size and memory footprint.

### Motivation

- Weight tying reduces parameters by sharing the embedding and output projection matrices
- Provides a parameter-efficient baseline for comparison against separated embedding architectures
- Uses standard learning rate `tied_embed_lr=0.05` for the shared embedding matrix
- Serves as the control experiment for ablation studies on embedding strategies

## Key Configuration

| Component | Setting |
|---|---|
| `tie_embeddings` | 1 (shared tok_emb/lm_head) |
| tok_emb LR | 0.05 (`tied_embed_lr`) |
| lm_head | Tied (uses tok_emb weights) |
| Parameters | 17.04M (baseline) |
| Learning Rate Strategy | Single unified LR for tied embedding |
| Architecture | Identical to baseline (GQA, RoPE, RMSNorm, QK-Norm) |

## Configuration

| Regime | Metric | Value |
|---|---|---|
| Fixed Compute | Target | 20 min wall-clock |
| Fixed Tokens | Target | 10B tokens |
| Embedding Strategy | Type | Weight-tied |
| — | tied_embed_lr | 0.05 |
| — | Total Params | 17,039,360 (17.04M) |
| — | Peak Memory | ~8,389 MiB |

## Analysis

Weight tying is the standard parameter-efficiency baseline. By sharing a single weight matrix between input embedding and output projection, this configuration:

1. **Minimizes parameter count**: Uses the smallest possible embedding matrix, reducing overall model size
2. **Provides a control baseline**: Serves as the reference point against which other embedding strategies (e.g., untied embeddings) are compared
3. **Establishes unified learning dynamics**: The shared matrix trains under a single learning rate (`tied_embed_lr=0.05`), creating coupled optimization dynamics between input representation and output classification
4. **Uses stable, conservative LR**: The `tied_embed_lr=0.05` is a conservative setting that works well for the shared matrix

This configuration is typically used as the baseline/control experiment. Ablations can then compare:- **Untied embeddings**: separate matrices with higher per-matrix learning rates
- **Other embedding strategies**: custom initialization, positional encodings, etc.

The value of this tied-embed configuration is that it establishes the performance ceiling for parameter-efficient models and provides a clear comparison point for more expressive alternatives.

## Files

- `train_gpt.py`: Training script (`TIE_EMBEDDINGS=1` by default, weight-tied baseline)
- `tie-embed.json`: Experiment manifest (`tie_embeddings: 1`, tied_embed_lr=0.05)
- `tie-embed-medium.json`: Medium variant for faster iteration
- `logs/`: Training output (automatically generated)
