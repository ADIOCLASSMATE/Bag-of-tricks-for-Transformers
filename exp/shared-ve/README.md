# Shared VE — inject token embedding into the V path of early attention layers

## Method

trick: shared-ve — Reuse the existing token embedding as an additional signal injected
into the value (V) of the first 3 attention layers:

```
token_embed = tok_emb(input_ids)
For layer 0, 1, 2:
    v = v + 5 * ve_lambda * token_embed
For layer 3..8:
    v unchanged
```

- `ve_lambda` is a per-layer learnable scalar initialized to `0.0` (no effect at step 0)
- The 5× multiplier provides ~5× effective LR via gradient scaling
- Zero extra parameters — reuses `tok_emb.weight` directly

The design hypothesis: raw token identity is most useful in shallow layers where
local features dominate; deeper layers work with abstractions that token identity
does not directly help.

## Differences from original ValueEmbed (modded-nanogpt `2024-12-04_ValueEmbed`)

| Aspect | Original ValueEmbed | Shared VE (this) |
|---|---|---|
| VE source | Separate `vte` table (`vocab × dim×12`) | Reuses `tok_emb` (0 params) |
| Injection formula | `v = (1-λ)·v + λ·ve` (convex combination) | `v = v + 5·λ·token_embed` (additive) |
| λ init | 0.5 | 0.0 |
| Layer coverage | All 12 layers (one VE slice per layer) | First 3 layers only |
| x0 residual | Present in each Block | Absent (standard residual) |

## Single-axis experiment (the 2 runs)

| Experiment | `ve_lambda` init | Control |
|---|---|---|
| `shared-ve-fixed_time_10min` | 0.0 (per layer, first 3) | fixed_compute 600 s |
| `shared-ve-fixed_tokens_10b` | 0.0 (per layer, first 3) | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | Shared VE |
|---|---|---|
| V injection | `v = x @ W_v.T` | `v = x @ W_v.T + 5·λ·token_embed` (layers 0-2) |
| New parameters | 0 | +3 scalars (`ve_lambda` per early layer) |
| Extra compute | — | One `view_as` + add per early attention layer |
| Parameters | 17.04M | ~17.04M (+3, negligible) |

## Origin

- Source record: modded-nanogpt `2024-12-04_ValueEmbed` by @KoszarskyB
- Inspired by @Grad62304977's value residual learning (arXiv 2410.17897)
- This version simplifies the original by reusing the token embedding instead of
  a separate VE table, and restricts injection to early layers only.

## Impact on training

- **Memory.** Negligible — 3 additional scalar parameters per model.
- **Compute.** Negligible — one `view_as` and add per early attention layer.
- **Convergence.** The token identity signal in early layers may help the model
  retain lexical information that QK-Norm would otherwise discard. The additive
  formula with zero-init ensures no disruption to baseline behavior at step 0.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | Shared VE | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | Shared VE | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with shared VE (look for `trick: shared-ve`)
- `shared-ve.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
# Dry-run the manifest
python exp/run_experiments.py exp/shared-ve/shared-ve.json --dry-run

# Launch both experiments
python exp/run_experiments.py exp/shared-ve/shared-ve.json
```
