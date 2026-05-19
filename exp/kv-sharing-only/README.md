# kv-sharing-only — cross-layer KV sharing, no MLP changes

## Trick

Last `N` layers of the model drop their own `c_k`/`c_v` projections and reuse the K/V tensors from earlier source layers. With `num_kv_shared_layers=0.2222`:

- small (9 layers): layers 7–8 reuse K/V from layers 0–1
- medium (18 layers): layers 14–17 reuse K/V from layers 0–3

The MLP hidden dim is left at the baseline `mlp_mult` — this is the difference vs `kv-sharing-dwide-mlp`.

## Net parameter effect

For small: saves `c_k + c_v` on the 2 trailing layers ≈ 2 × 2 × 512 × 256 ≈ 524 K params. So small is roughly **17.04 M (baseline 17.56 M − 0.52 M)**. Compute (FLOPs) is essentially unchanged — `q` is still computed, attention is still done, only `c_k`/`c_v` matmuls are skipped on the shared layers.

## Companion to `dwide-mlp-only` and `kv-sharing-dwide-mlp`

| Variant | KV shared | MLP × 2 | What it isolates |
|---|---|---|---|
| baseline | no | no | reference |
| **kv-sharing-only** | yes | no | the KV sharing axis |
| dwide-mlp-only | no | yes | the wider-MLP axis |
| kv-sharing-dwide-mlp | yes | yes | the bundled trick |

## Hyperparameters

| Name | Default | Meaning |
|---|---|---|
| `NUM_KV_SHARED_LAYERS` | 0.2222 (× num_layers) | how many trailing layers reuse earlier K/V |
| `USE_DOUBLE_WIDE_MLP` | **0** | wide-MLP disabled (this is the difference vs `kv-sharing-dwide-mlp`) |

## Training recipe

Identical to baseline. Trainer code is the same file body as `kv-sharing-dwide-mlp`; the only knob flipped is `USE_DOUBLE_WIDE_MLP=0`.

## Files

- `train_gpt.py`
- `kv-sharing-only.json` / `kv-sharing-only-medium.json`
