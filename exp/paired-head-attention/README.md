# Paired Head Attention — interleave adjacent head pairs in sequence dimension

## Method

trick: paired-head-attention — Adjacent heads are paired by interleaving their sequence
dimensions, creating 2×T sequences for each head pair. Each token in the paired sequence
can attend to tokens from both heads in the pair, enriching the attention pattern:

```
# Interleave: (B, H, T, D) → (B, H//2, 2*T, D)
q = q.reshape(B, H//2, 2, T, D).transpose(2, 3).reshape(B, H//2, 2*T, D)
# Run standard sdpa with is_causal=True
y = sdpa(q, k, v, is_causal=True)
# De-interleave: (B, H//2, 2*T, D) → (B, H, T, D)
y = y.reshape(B, H//2, T, 2, D).transpose(2, 3).reshape(B, H, T, D)
```

Applied only to user-specified layers (default: 0, 2, 5, 9). For a 9-layer model,
layers not listed use standard attention.

## Single-axis experiment (the 2 runs)

| Experiment | `paired_head_layers` | Control |
|---|---|---|
| `paired-head-attention-fixed_time_10min` | 0,2,5,9 | fixed_compute 600 s |
| `paired-head-attention-fixed_tokens_10b` | 0,2,5,9 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | paired-head-attention |
|---|---|---|
| Attention mechanism | Standard causal (per-head) | Paired-head interleave on selected layers |
| Sequence length (paired layers) | T | 2T (head dim compressed H→H//2) |
| Parameters | 17.04M | **Identical** (zero parameter increase) |

## Origin

- Source record: modded-nanogpt `2026-01-07_PairedHeadAttention`
- README: `records/track_1_short/2026-01-07_PairedHeadAttention/README.md`

## Impact on training

- **Memory.** Identical — reshapes are view operations with no copy.
- **Compute.** sdpa operates on (H//2, 2T) instead of (H, T). Same total elements, but
  attention FLOPs scale as O(T²) per head, so paired heads pay 2× attention cost on
  paired layers.
- **Convergence.** Interleaved attention allows each query to attend to keys/values from
  both heads in the pair, potentially reducing head redundancy and improving signal
  quality.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | paired-head-attention | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | paired-head-attention | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with paired head attention (look for `# trick: paired-head-attention`)
- `paired-head-attention.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
python exp/run_experiments.py exp/paired-head-attention/paired-head-attention.json --dry-run
python exp/run_experiments.py exp/paired-head-attention/paired-head-attention.json
```
