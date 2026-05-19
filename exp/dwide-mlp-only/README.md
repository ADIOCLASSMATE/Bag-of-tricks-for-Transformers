# dwide-mlp-only — double-wide MLP in the last N layers, no KV sharing

## Trick

Last `N` layers of the model use an MLP with **doubled** hidden dim (`mlp_mult * 2`). All layers keep their own independent `c_k`/`c_v` (no cross-layer sharing). With `num_wide_mlp_layers=0.2222`:

- small (9 layers): MLP of the last 2 layers is widened from `mlp_mult=2` to `mlp_mult=4`
- medium (18 layers): MLP of the last 4 layers is widened from `mlp_mult=2` to `mlp_mult=4`

## Net parameter effect

For small: each widened layer adds `2 × mlp_mult × dim^2 = 2 × 2 × 512^2 ≈ 1.05 M` params. Two widened layers ⇒ +2.1 M params on top of baseline (17.56 M → ~19.66 M). Compute (FLOPs) is correspondingly higher — the wider MLP matmuls are larger.

## Companion to `kv-sharing-only` and `kv-sharing-dwide-mlp`

| Variant | KV shared | MLP × 2 | What it isolates |
|---|---|---|---|
| baseline | no | no | reference |
| kv-sharing-only | yes | no | the KV sharing axis |
| **dwide-mlp-only** | no | yes | the wider-MLP axis |
| kv-sharing-dwide-mlp | yes | yes | the bundled trick |

## Hyperparameters

| Name | Default | Meaning |
|---|---|---|
| `NUM_WIDE_MLP_LAYERS` | 0.2222 (× num_layers) | trailing layers whose MLP hidden dim is doubled |

## Training recipe

Identical to baseline. Trainer reuses `trainlib.transformer.CausalSelfAttention` (no KV-sharing logic), with a local `Block` / `GPT` that accepts a per-layer `mlp_mult_override` for the last `N` layers.

## Files

- `train_gpt.py`
- `dwide-mlp-only.json` / `dwide-mlp-only-medium.json`
