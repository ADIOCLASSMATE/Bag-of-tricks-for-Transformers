# Factored Embedding — smaller embed dim + linear projection to model_dim

## Method

trick: factored-embedding — Replace the full V × model_dim embedding table with:

```
x_embedded = tok_emb(input_ids)           # (B, T, embed_dim)
x = embed_proj(x_embedded)               # (B, T, model_dim)
```

Where `embed_dim < model_dim` (default: 256 vs 512). Total params:

| Component | Baseline (tied) | Factored |
|---|---|---|
| Token embedding | V × 512 = 524K | V × 256 = 262K |
| Embed projection | — | 256 × 512 = 131K |
| LM head | — (tied) | 512 × V = 524K |
| **Total** | **524K** | **917K** |

Factored embedding **requires untied lm_head** since `embed_dim != model_dim` breaks
the weight tie. The lm_head acts as an "unembedding" projection back to vocabulary space.

For small vocab sizes (1024), the parameter count increases. For large vocabs,
the reduction `V × (model_dim - embed_dim) - embed_dim × model_dim` outweighs the
lm_head cost.

## Single-axis experiment (the 2 runs)

| Experiment | `embed_dim` | Control |
|---|---|---|
| `factored-embedding-fixed_time_10min` | 256 | fixed_compute 600 s |
| `factored-embedding-fixed_tokens_10b` | 256 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | Factored Embedding |
|---|---|---|
| Token embedding | Embedding(1024, 512) | Embedding(1024, 256) |
| Embed projection | None | CastedLinear(256, 512) |
| LM head | Tied (none) | CastedLinear(512, 1024) |
| tie_embeddings | True | False |
| Parameters | 17.04M | ~17.43M (+393K) |

## Origin

- Source record: parameter-golf `2026-04-01_Vocab4096_MLPMult4_WD085`
  - A12, BPB=1.2393
  - Used embed_dim=320 with vocab_size=4096, model_dim=576

## Impact on training

- **Memory.** Slightly increased for small vocab (1024): +393K params. Much larger
  savings for large vocabs (e.g., 8192 BPE).
- **Compute.** One additional small matmul per forward pass (embed_dim → model_dim).
- **Convergence.** The embedding factorization + forced untied head changes the
  parameterization. The projection layer provides a learned "translation" between
  the compact embedding space and the full model dimension.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | Factored Embedding | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | Factored Embedding | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with factored embedding (look for `# trick: factored-embedding`)
- `factored-embedding.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
python exp/run_experiments.py exp/factored-embedding/factored-embedding.json --dry-run
python exp/run_experiments.py exp/factored-embedding/factored-embedding.json
```
