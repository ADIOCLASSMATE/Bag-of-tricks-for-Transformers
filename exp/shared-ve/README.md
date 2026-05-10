# Shared VE — per-layer V modulation with token embedding injection in early layers

## Method

trick: shared-ve — Each attention layer has two learnable scalars `[λ_v, λ_ve]` that
modulate V and inject token identity into the value path:

```
token_embed = tok_emb(input_ids)

Layer 0-2:  v = 5 * λ_v * v + 5 * λ_ve * token_embed
Layer 3-8:  v unchanged
```

- `λ_v` init = 0.2 → effective 1.0 (V unchanged at step 0)
- `λ_ve` init = 0.0 → effective 0.0 (VE has no effect at step 0)
- The 5× multiplier provides ~5× effective LR via gradient scaling
- Zero extra parameters — `token_embed` reuses `tok_emb.weight` directly

## Single-axis experiment (the 2 runs)

| Experiment | lambdas init | Control |
|---|---|---|
| `shared-ve-fixed_time_10min` | [0.2, 0.0] per layer (first 3) | fixed_compute 600 s |
| `shared-ve-fixed_tokens_10b` | [0.2, 0.0] per layer (first 3) | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | Shared VE |
|---|---|---|
| V path | `v = x @ W_v.T` | `v = 5·λ_v·v + 5·λ_ve·token_embed` (layers 0-2) |
| New parameters | 0 | +18 scalars (2 per attention layer × 9 layers) |
| Extra compute | — | One `view_as` + scalar ops per VE-active attention layer |
| Parameters | 17.04M | ~17.04M (+18, negligible) |

## Differences from original ValueEmbed (modded-nanogpt `2024-12-04_ValueEmbed`)

| Aspect | Original ValueEmbed | Shared VE (this) |
|---|---|---|
| VE source | Separate `vte` table (`vocab × dim×12`) | Reuses `tok_emb` (0 params) |
| Injection formula | `v = (1-λ)·v + λ·ve` | `v = 5·λ_v·v + 5·λ_ve·token_embed` |
| λ init | 0.5 (single coefficient) | [0.2, 0.0] (independent V and VE) |
| Layer coverage | All 12 layers | First 3 layers only |
| x0 residual | Present in each Block | Absent |

## Origin

- Source record: modded-nanogpt `2024-12-04_ValueEmbed` by @KoszarskyB
- Inspired by @Grad62304977's value residual learning (arXiv 2410.17897)
- This version reuses the token embedding and adds independent per-layer V/VE coefficients.

## Impact on training

- **Memory.** Negligible — 18 additional scalar parameters per model (lambdas in all layers).
- **Compute.** Negligible — scalar-vector ops per VE-active attention layer.
- **Convergence.** λ_v allows per-layer V magnitude tuning; λ_ve controls token identity
  injection strength. Both init to no-op so training starts identically to baseline.

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
