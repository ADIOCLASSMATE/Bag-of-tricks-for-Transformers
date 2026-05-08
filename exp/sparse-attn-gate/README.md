# Sparse Attention Gate — low-dimensional per-head gating on attention output

## Method

trick: sparse-attn-gate — A compact linear layer maps a small slice of the residual
stream (default 12 dims out of 512) to per-head sigmoid gates. The gate is applied
to the attention output before the output projection:

```
gate = sigmoid(Linear_{gate_input_dim → num_heads}(x[:, :gate_input_dim]))
y = attention(q, k, v) * gate
```

The extremely low-dimensional input (12/512 = 2.3%) forces the gate to be sparse
and context-selective, rather than learning trivial per-head scale factors.

Gate weights are zero-initialized so the trick starts as an identity function.

## Single-axis experiment (the 2 runs)

| Experiment | `attn_gate_dim` | Control |
|---|---|---|
| `sparse-attn-gate-fixed_time_10min` | 12 | fixed_compute 600 s |
| `sparse-attn-gate-fixed_tokens_10b` | 12 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | sparse-attn-gate |
|---|---|---|
| Attention gate | None | Learnable gate: Linear(12, 8) → sigmoid |
| Parameters | 17.04M | **+108** (dm × heads + heads params) |

## Origin

- Source record: modded-nanogpt `2025-08-23_SparseAttnGate`
- README: `records/track_1_short/2025-08-23_SparseAttnGate/README.md`

## Impact on training

- **Memory.** Negligible (~108 new params per layer; 9×108 ≈ 972 params total).
- **Compute.** One additional small matmul (dim → heads) per attention layer.
- **Convergence.** Sparse gating allows heads to be contextually dampened, potentially
  reducing interference between heads and improving signal quality.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | sparse-attn-gate | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | sparse-attn-gate | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with sparse attention gate (look for `# trick: sparse-attn-gate`)
- `sparse-attn-gate.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
python exp/run_experiments.py exp/sparse-attn-gate/sparse-attn-gate.json --dry-run
python exp/run_experiments.py exp/sparse-attn-gate/sparse-attn-gate.json
```
