# tie-embed — Tied input/output embeddings

## Trick

Toggle `tie_embeddings = 1`: the token embedding matrix is reused as the output projection (no separate `lm_head`). Architecturally a single-variable change from baseline.

## How it differs from baseline

| | baseline | tie-embed |
|---|---|---|
| `tie_embeddings` | 0 | **1** |
| Embedding LR (AdamW) | `embed_lr=0.6` | `tied_embed_lr=0.05` |
| Weight decay on `tok_emb` | 0 | 0 |
| Separate `lm_head` | yes (lr=0.008, wd=0.1) | **none** |
| Param count vs baseline | — | −524 K (no `lm_head`) on small |

Everything else (Muon LR, scalar LR, Muon momentum warmup, warmdown_iters, batch tokens, sequence length, GQA, RoPE, RMSNorm, QK-Norm) is identical to baseline.

## Recipe

- Same as baseline.
- Weight decay = 0 on the tied embedding (matches baseline's `tok_emb`, eliminates the WD-coupling that earlier versions had).

## Files

- `train_gpt.py` — trainer (identical body to baseline; `Hyperparameters` overrides `tie_embeddings = 1`)
- `tie-embed.json` / `tie-embed-medium.json` — small / medium manifests
