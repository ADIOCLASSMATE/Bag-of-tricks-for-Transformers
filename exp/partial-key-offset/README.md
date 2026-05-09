# Partial Key Offset — shift stationary key dimensions forward

## Method

trick: partial-key-offset — For long-window attention layers, shift the key tensor's
stationary dimensions (those not rotated by RoPE) forward by 1 position. This allows
each query to attend to the *previous* token's stationary features, enabling single-layer
induction head formation without additional parameters.

trick: partial-key-offset dependency — Half-Truncated RoPE: only rotate the first
`head_dim//4` dimensions; the remaining `3*head_dim//4` dims are stationary.
The key offset then shifts only the stationary quarters:

```
k[:, 1:, :, head_dim//4 : head_dim//2]                  = k[:, :-1, :, head_dim//4 : head_dim//2]
k[:, 1:, :, head_dim//4 + head_dim//2 :]                = k[:, :-1, :, head_dim//4 + head_dim//2 :]
```

Key offset is applied per the `key_offset_layers` config (default: layers 3 and 8).

## Single-axis experiment (the 2 runs)

| Experiment | `key_offset_layers` | Control |
|---|---|---|
| `partial-key-offset-fixed_time_10min` | 3,8 | fixed_compute 600 s |
| `partial-key-offset-fixed_tokens_10b` | 3,8 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | partial-key-offset |
|---|---|---|
| RoPE | Full rotation (head_dim/2 freqs) | Half-truncated (head_dim/4 freqs, rest zero) |
| Key shift | None | Shift stationary dims forward on layers 3,8 |
| Parameters | 17.04M | **+0** (unchanged) |

## Origin

- Source record: modded-nanogpt `2025-12-14_PartialKeyOffset`
- README: `records/track_1_short/2025-12-14_PartialKeyOffset/README.md`

## Impact on training

- **Memory.** No new parameters; cos/sin cache shrinks slightly.
- **Compute.** Key shift is a trivial in-place tensor assignment.
- **Convergence.** Partial key offset enables 1-layer induction heads by giving
  attention a direct "previous position" signal via stationary key dims.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | partial-key-offset | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | partial-key-offset | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with partial key offset (look for `# trick: partial-key-offset`)
- `partial-key-offset.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
# Dry-run the manifest
python exp/run_experiments.py exp/partial-key-offset/partial-key-offset.json --dry-run

# Launch both experiments
python exp/run_experiments.py exp/partial-key-offset/partial-key-offset.json
```
