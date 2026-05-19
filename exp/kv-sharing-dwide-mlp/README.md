# kv-sharing-dwide-mlp — bundled trick: cross-layer KV sharing + double-wide MLP

## Trick

Last `N` layers of the model:

1. **Drop their own `c_k`/`c_v`** and reuse the K/V tensors from earlier source layers. With `num_kv_shared_layers=0.2222`, small has the last 2 layers reusing layers 0/1; medium has the last 4 layers reusing layers 0/1/2/3.
2. **Double their MLP hidden dim** (`mlp_mult * 2`), to take advantage of the parameter budget freed by dropping `c_k`/`c_v`. Net effect is *not* parameter-neutral — for small, baseline 17.56 M → 19.14 M (+9.2 %).

This is the **bundled** Gemma-style trick: combining the two changes is what the paper reports.

## Why we also run two single-axis ablations

Because the trick is a bundle of two changes, "kv-sharing-dwide-mlp beats baseline" cannot be attributed to either component on its own. Two companion experiments isolate them:

| Variant | KV shared (last N) | MLP × 2 (last N) | Param delta vs baseline | What it tests |
|---|---|---|---|---|
| baseline | no | no | 0 | reference |
| **kv-sharing-only** | yes | no | small ↓ (saves `c_k+c_v`) | does KV sharing alone hurt? |
| **dwide-mlp-only** | no | yes | small ↑ (adds MLP params) | does extra MLP capacity alone explain the win? |
| **kv-sharing-dwide-mlp** | yes | yes | +9.2 % (small) | bundled — the original recipe |

If `kv-sharing-only` recovers most of the bundle's improvement, the KV sharing is doing the work; if `dwide-mlp-only` does, it's the extra MLP capacity; if neither alone matches the bundle, the two interact.

## KV sharing details

For each shared layer `i` ∈ [`first_kv_shared_layer`, `num_layers`):
```
source = (i − first_kv_shared_layer) mod first_kv_shared_layer
```
The destination layer's `q` is computed from its own current hidden state; `k`/`v` are taken from the source layer's already-RMS-normed-and-RoPE-applied tensors and used directly. RoPE's relative-phase property keeps the attention pattern well-defined despite the layer offset.

## Hyperparameters

| Name | Default | Meaning |
|---|---|---|
| `NUM_KV_SHARED_LAYERS` | 0.2222 (× num_layers) | how many trailing layers reuse earlier K/V |
| `USE_DOUBLE_WIDE_MLP` | 1 | doubles the MLP hidden dim on the same trailing layers |

## Training recipe

Identical to baseline.

## Files

- `train_gpt.py`
- `kv-sharing-dwide-mlp.json` / `kv-sharing-dwide-mlp-medium.json`
