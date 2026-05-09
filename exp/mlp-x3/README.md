# MLP-x3 — widen MLP hidden dimension from 2× to 3× model_dim

## Method

trick: mlp-x3 — Widen the MLP hidden layer from `2 × model_dim` to `3 × model_dim`:

```
# Baseline
hidden = 2 * dim = 1024
MLP: fc(dim → hidden) + gelu + proj(hidden → dim)

# MLP-x3
hidden = 3 * dim = 1536
MLP: fc(dim → hidden) + gelu + proj(hidden → dim)
```

This is a pure capacity expansion — no architectural change, just a wider hidden layer.
The MLP parameter count increases by ~50%, giving the FFN more representational capacity
within the same layer count.

## Single-axis experiment (the 2 runs)

| Experiment | `mlp_mult` | Control |
|---|---|---|
| `mlp-x3-fixed_time_10min` | 3 | fixed_compute 600 s |
| `mlp-x3-fixed_tokens_10b` | 3 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | MLP-x3 |
|---|---|---|
| mlp_mult | 2 | 3 |
| Hidden dim | 1024 | 1536 |
| MLP params/layer | 1.05M | 1.57M |
| Total MLP params (9L) | 9.44M | 14.16M |
| Total model params | 17.04M | 21.76M (+4.72M) |
| MLP FLOPs | ~1.0× | ~1.5× |

## Origin

- Source record: parameter-golf `A03 MLPx3`
- Widen MLP hidden dim to 3× for higher FFN capacity within the same architecture

## Impact on training

- **Memory.** +4.72M params (all in MLP matrices). Activations grow proportionally.
- **Compute.** ~50% more FLOPs in MLP sublayers. In fixed-compute mode, this means
  fewer tokens processed — the trade-off is more capacity vs fewer training tokens.
- **Convergence.** Wider MLP gives more per-layer capacity, potentially better
  loss at the same depth. The fixed-compute comparison reveals whether the capacity
  gain outweighs the throughput loss.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | MLP-x3 | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | MLP-x3 | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer (identical to baseline; `mlp_mult` overridden via env var)
- `mlp-x3.json` — 2-experiment manifest with `"mlp_mult": 3`
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
# Dry-run the manifest
python exp/run_experiments.py exp/mlp-x3/mlp-x3.json --dry-run

# Launch both experiments
python exp/run_experiments.py exp/mlp-x3/mlp-x3.json
```
