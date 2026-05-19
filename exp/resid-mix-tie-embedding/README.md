# resid-mix-tie-embedding — resid-mix under tied embedding

## What this tests

Standard `resid-mix` uses separate input embedding and output projection (lm_head) matrices. This experiment tests whether resid-mix still helps when the embedding is **tied** — the lm_head weight is shared with the input embedding matrix.

The baseline is also a tie-embedding model, so any delta is attributable purely to resid-mix under the tie constraint.

## Trick

Each transformer block receives two inputs: the current hidden state `x` and the original token embedding `x0`. Before the attention sublayer, the block mixes them:

```
x = mix[0] * x + mix[1] * x0
```

where `mix` is a learnable `[2, dim]` parameter (one scalar per residual channel). The remaining residual structure (`x = x + attn(...)`, `x = x + mlp(...)`) is unchanged.

## Initialization (identity-equivalent)

`resid_mix` is initialized at `[1.0, 0.0]` per channel, so at step 0 the model is **bit-identical to its tie-embedding baseline**. Any improvement is therefore attributable to the learnable mixing alone.

## Optimizer routing

`resid_mix` is a 2D tensor (`shape [2, dim]`), so the default optimizer split would route it to **Muon** alongside the weight matrices. That is wrong: Muon performs spectral normalization for matrix updates, but `resid_mix` is a pair of per-channel scaling vectors stacked into a 2D shape.

Fix: this trainer sets `CONTROL_TENSOR_NAME_PATTERNS = "resid_mix,resid_mixes"` **before** importing `trainlib.optimizer`, so parameters whose names contain `resid_mix` are excluded from `matrix_params` and routed to the AdamW scalar group instead.

## Net parameter effect

Negligible: adds `num_layers × 2 × dim` params (small: 9 × 2 × 512 = 9 216 params, +0.05 %). Weight tying partially offsets this since the lm_head is shared with the embedding.

## Training recipe

Identical to the tie-embedding baseline.

## Files

- `train_gpt.py`
- `resid-mix-tie-embedding.json` / `resid-mix-tie-embedding-medium.json`
