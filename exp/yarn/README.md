# YaRN — Yet another RoPE extensioN

## Method

trick: yarn — Extends RoPE's effective context window beyond the training sequence length
by applying a ramp-based frequency interpolation:

```python
scale = train_seq_len / yarn_max_len       # e.g. 1024/4096 = 0.25
freq_idx = torch.arange(0, dim, 2)
ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
```

- **High frequencies** (first 25% of dims, ramp=0): linear scaling by `scale`
- **Mid frequencies** (25%-100% of dims): smooth interpolation via ramp
- **Low frequencies** (>100% of dims): no scaling (ramp=1)

This preserves high-frequency short-range positional information while stretching
low-frequency dimensions for longer-range position discrimination.

## Single-axis experiment (the 2 runs)

| Experiment | `use_yarn` | `yarn_max_len` | Control |
|---|---|---|---|
| `yarn-fixed_time_10min` | 1 | 4096 | fixed_compute 600 s |
| `yarn-fixed_tokens_10b` | 1 | 4096 | fixed_tokens 10 B |

## Key differences from baseline

| Parameter | Baseline | YaRN |
|---|---|---|
| RoPE frequencies | `1/base^(freq_idx / dim)` | YaRN-interpolated with scale=1024/4096 |
| Effective context window | ~1024 tokens | ~4096 tokens (theoretical) |
| Parameters | 17.04M | **Identical** (frequency scaling, no new params) |

## Origin

- Source record: parameter-golf `2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon`
  - A14, static YaRN variant
- Paper: Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models"

## Impact on training

- **Memory.** Zero additional memory.
- **Compute.** Zero additional compute at runtime (frequencies precomputed in __init__).
- **Convergence.** YaRN enables smooth extrapolation: the 1024-length training positions
  map to longer-range patterns through frequency interpolation, potentially improving
  generalization on long-range dependencies without training on longer sequences.

## Results

*To be filled after experiment completion.*

### Fixed Compute (600 s wall-clock)

| Metric | Baseline | YaRN | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Train Tokens | — | — | — |
| Peak Memory | — | — | — |

### Fixed Tokens (10 B tokens)

| Metric | Baseline | YaRN | Delta |
|---|---|---|---|
| **Val BPB** | — | — | — |
| Val Loss | — | — | — |
| Wall-clock | — | — | — |
| Peak Memory | — | — | — |

## Analysis

*To be filled after experiment completion.*

## Files

- `train_gpt.py` — trainer with YaRN (look for `# trick: yarn`)
- `yarn.json` — 2-experiment manifest
- `logs/` — experiment outputs (automatically generated)

## How to run

```bash
python exp/run_experiments.py exp/yarn/yarn.json --dry-run
python exp/run_experiments.py exp/yarn/yarn.json
```
