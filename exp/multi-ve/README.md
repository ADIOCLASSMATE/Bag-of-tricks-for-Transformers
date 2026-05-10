# Multi VE — two independent VE tables with per-layer V/VE modulation

## Method

trick: multi-ve — Two independent Value Embedding tables with per-layer learnable
`[λ_v, λ_ve]` coefficients:

```
ve_shallow = VE_shallow(input_ids)   # table 1
ve_deep   = VE_deep(input_ids)       # table 2

Layer 0-2:  v = 5 * λ_v * v + 5 * λ_ve * ve_shallow
Layer 3-5:  v unchanged
Layer 6-8:  v = 5 * λ_v * v + 5 * λ_ve * ve_deep
```

- Two independent `nn.Embedding(vocab_size, dim)` tables, learned from scratch
- Each VE-active layer has two scalars: λ_v (V coefficient) and λ_ve (VE coefficient)
- `[λ_v, λ_ve]` init = [0.2, 0.0] → effective [1.0, 0.0], identical to baseline at step 0
- The 5× multiplier provides ~5× effective LR via gradient scaling
- Layers 3-5 (middle) receive no VE injection

## Single-axis experiment (the 2 runs)

| Experiment | VE tables | Control |
|---|---|---|
| `multi-ve-fixed_time_10min` | 2 (`ve_shallow` + `ve_deep`) | fixed_compute 600 s |
| `multi-ve-fixed_tokens_10b` | 2 (`ve_shallow` + `ve_deep`) | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | Multi VE |
|---|---|---|
| V path | `v = x @ W_v.T` | `v = 5·λ_v·v + 5·λ_ve·ve(input_ids)` (layers 0-2, 6-8) |
| New params (VE tables) | 0 | 2 × 1024 × 512 = 1,048,576 |
| New params (scalars) | 0 | +18 (2 per attention layer × 9 layers) |
| Total params | 17.04M | ~18.09M (+1,048,594) |

## Relation to shared-ve

| Aspect | shared-ve | multi-ve |
|---|---|---|
| VE source | Reuses `tok_emb` (0 params) | 2 independent `nn.Embedding` tables |
| Layer coverage | 0-2 only | 0-2 (shallow) + 6-8 (deep) |
| λ design | Same `[λ_v, λ_ve]` per layer | Same `[λ_v, λ_ve]` per layer |
| Extra params | +18 scalars | +1,048,576 (2 VE tables) + 18 scalars |

## Origin

- Source record: modded-nanogpt `2024-12-04_ValueEmbed` by @KoszarskyB
- Inspired by @Grad62304977's value residual learning (arXiv 2410.17897)
- This version strips the U-net dependency and uses two independent tables
  for shallow vs deep layers, each with per-layer V/VE coefficients.

## Impact on training

- **Memory.** +1,048,594 params (+1,048,576 for two embedding tables, +18 for lambdas).
- **Compute.** Scalar-vector ops per VE-active attention layer (6 of 9 layers).
- **Convergence.** Independent VE tables enable shallow/deep divergence; λ_v allows
  per-layer V tuning; λ_ve controls VE injection per layer.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | Multi VE | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | Multi VE | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with multi VE (look for `trick: multi-ve`)
- `multi-ve.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
# Dry-run the manifest
python exp/run_experiments.py exp/multi-ve/multi-ve.json --dry-run

# Launch both experiments
python exp/run_experiments.py exp/multi-ve/multi-ve.json
```
