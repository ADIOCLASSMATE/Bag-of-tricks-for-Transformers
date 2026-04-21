# Lower-LR — Halved Learning Rates

## Method Overview

This experiment lowers the main training learning rates from the baseline defaults to:

- `matrix_lr`: 0.04 → **0.02**
- `scalar_lr`: 0.04 → **0.02**
- `tied_embed_lr`: 0.05 → **0.03**

This isolates whether the baseline recipe is simply stepping too aggressively.

## Impact on Training

- **Parameters**: No change
- **Throughput**: Essentially unchanged
- **Peak memory**: unchanged at 10,246 MiB
- **Optimization**: Smaller update size across Muon and Adam groups

## Key Differences from Baseline

| Parameter | baseline-sp1024 | lower-lr |
|---|---|---|
| `matrix_lr` | 0.04 | **0.02** |
| `scalar_lr` | 0.04 | **0.02** |
| `tied_embed_lr` | 0.05 | **0.03** |
| Everything else | identical | identical |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | lower-lr | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | **1.2182** | **−0.0012** |
| Val Loss | 2.0589 | **2.0569** | **−0.0020** |
| Steps completed | 13,263 | 13,327 | +64 |
| Tokens processed | 6.95B | 6.99B | +0.03B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | lower-lr | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | **1.2100** | **−0.0018** |
| Val Loss | 2.0460 | **2.0430** | **−0.0030** |
| Steps | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 838.6s | +5.8s |

## BPB Analysis

The lower learning rate helps, but only moderately.

- **Fixed compute**: The gain is 0.0012 BPB with effectively identical throughput, so the smaller step size is slightly cleaner but not transformative.
- **Fixed tokens**: The gain increases to 0.0018 BPB, which suggests the lower LR is more helpful over a long horizon than in a short fixed-compute run.
- **Related sweep point**: `lower-lr-003` on the same codebase reached 1.2175 / 1.2097 BPB, slightly better than `lower-lr` in fixed compute and slightly better again in fixed tokens, so the best LR in this environment may sit between the baseline and the halved setting.

## Files

- `train_gpt.py`: Training script (identical to baseline; LR overrides come from the manifest)
- `lower-lr.json`: Experiment manifest
- `logs/`: Training output
