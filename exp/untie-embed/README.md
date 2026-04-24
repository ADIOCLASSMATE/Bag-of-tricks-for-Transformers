# Untie-Embed — Separate Embedding and LM-Head Weights

## Method Overview

This experiment **unties the embedding and language model head weights**, giving each its own independent parameter matrix. In the baseline, a single weight matrix is shared between the token embedding layer and the output projection (lm_head). This experiment sets `tie_embeddings=0` and changes nothing else — `embed_lr=0.6` and `head_lr=0.008` remain at their default values.

**Important: This is not a pure single-variable ablation.** While only the `tie_embeddings` flag changes (1→0), the baseline code's learning rate routing (`token_lr = tied_embed_lr if tie_embeddings else embed_lr`) means untying automatically switches the effective embedding learning rate from `tied_embed_lr=0.05` to `embed_lr=0.6` — a **12x increase**. The reported improvement conflates two effects: (1) the architectural benefit of separate matrices, and (2) the much larger embedding learning rate. These cannot be disentangled without an additional control experiment (untied with `embed_lr=0.05`).

Weight tying forces the embedding and lm_head to share a single weight matrix, creating a conflicting optimization objective: the embedding must simultaneously serve as a good input representation and a good output classifier. Untying resolves this conflict — the separate lm_head can specialize for the classification task (mapping hidden states to vocabulary logits) while tok_emb specializes for the input representation task (mapping tokens to hidden space).

### Motivation

- The embedding layer and lm_head have fundamentally different roles — tying them is a parameter-efficiency hack that sacrifices expressiveness
- Untying adds only +0.52M params (+3.1%) with negligible memory/runtime overhead
- Even with baseline learning rates, the expressiveness gain from two independent matrices is substantial
- Note: the effective embedding LR changes from 0.05 to 0.6 when untying (see below)

## Key Differences from Baseline

| Component | Baseline | untie-embed |
|---|---|---|
| `tie_embeddings` | 1 (shared tok_emb/lm_head) | **0** (independent matrices) |
| Effective tok_emb LR | 0.05 (`tied_embed_lr`) | **0.6** (`embed_lr`) — 12x increase |
| lm_head LR | — (tied) | **0.008** (`head_lr`) |
| Parameters | 17.04M | **17.56M** (+0.52M) |
| Everything else | identical | identical |

## Results

| Regime | Metric | Baseline | Untie-Embed | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2461 | **-0.0518** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1040 | -0.0874 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.56B | -1.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,398 MiB | +9 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2398 | **-0.0459** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.0934 | -0.0775 |
| Fixed Tokens (10B) | Wall-clock | 772s | 775s | +0.4% |
| Fixed Tokens (10B) | Peak Memory | 8,389 MiB | 8,398 MiB | +9 MiB |
| — | Total Params | 17,039,360 (17.04M) | 17,563,648 (17.56M) | +524,288 (+3.1%) |

## Analysis

Untying embeddings produces the largest improvement in the ablation suite: -0.052 FC BPB, -0.046 FT BPB with near-zero overhead. However, this improvement conflates two effects that cannot be disentangled from this experiment alone:

1. **Architectural expressiveness**: Tied embeddings force a single matrix to optimize for two conflicting objectives simultaneously. Untying lets each matrix specialize — the separate lm_head focuses on mapping hidden states to vocabulary logits (a classification task), while tok_emb focuses on mapping tokens into the representation space (an encoding task).

2. **12x embedding learning rate increase**: The baseline code routes `token_lr = tied_embed_lr if tie_embeddings else embed_lr`, so untying automatically changes the effective embedding LR from 0.05 to 0.6. A larger embedding LR means faster adaptation of the input representation, which may account for a significant portion of the observed improvement.

The relative contribution of each effect is unknown without a control experiment (untied with `embed_lr=0.05`). The 12x LR increase is not an arbitrary tuning choice — it is the baseline code's default behavior when `tie_embeddings=False`. This makes the experiment a valid test of "what happens when you untie embeddings using the baseline code," but the improvement cannot be attributed solely to architectural expressiveness.

Despite this caveat, the trick has an excellent quality-to-cost ratio: +0.52M params (+3.1%), +0.4% wall-clock, +9 MiB memory for a 4.0% BPB reduction under fixed compute.

## Files

- `train_gpt.py`: Training script (changed default `TIE_EMBEDDINGS` from 1 to 0; all other defaults unchanged)
- `untie-embed.json`: Experiment manifest (`tie_embeddings: 0`, baseline LRs)
- `logs/`: Training output (automatically generated)
