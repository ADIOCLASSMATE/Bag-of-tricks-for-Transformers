# XSA — Exclusive Self-Attention

## Method

trick: xsa — Subtract the self-value projection from attention output:

```
vn = F.normalize(v, dim=-1)
y_xsa = y - proj_{vn}(y) = y - (y · vn) * vn / ||vn||
```

Implemented as a GQA-aware operation that correctly handles grouped query attention:

```python
def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
    B, H, T, D = y.shape
    Hkv = v.size(1)
    group = H // Hkv
    y_g = y.reshape(B, Hkv, group, T, D)
    vn = F.normalize(v, dim=-1).unsqueeze(2)
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, H, T, D)
```

Applied to the last N layers (default: 4, i.e. layers 5-8 of a 9-layer model).
Set `xsa_last_n=-1` to apply to all layers.

## Single-axis experiment (the 2 runs)

| Experiment | `xsa_last_n` | Control |
|---|---|---|
| `xsa-fixed_time_10min` | 4 | fixed_compute 600 s |
| `xsa-fixed_tokens_10b` | 4 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | XSA |
|---|---|---|
| Attention output | y = sdpa(q, k, v) | y = sdpa(q, k, v) - proj_{vn}(sdpa(q, k, v)) |
| Modified layers | None | Last 4 layers |
| Parameters | 17.04M | **Identical** (zero parameter increase) |

## Origin

- Source record: parameter-golf `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - A06 (XSA4): last 4 layers, BPB=1.1248
  - A07 (XSA-all): all layers

## Impact on training

- **Memory.** Negligible — one additional normalize + element-wise ops per modified layer.
- **Compute.** Minimal — only F.normalize and a few element-wise ops per modified layer.
- **Convergence.** Removing self-value projection reduces a degenerate dimension in
  attention output, allowing heads to produce more diverse representations.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | XSA | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | XSA | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with XSA (look for `# trick: xsa`)
- `xsa.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
python exp/run_experiments.py exp/xsa/xsa.json --dry-run
python exp/run_experiments.py exp/xsa/xsa.json
```
