# Grad-Clip-03 — Gradient Clipping at 0.3

## Method Overview

This experiment enables **gradient norm clipping** by setting `grad_clip_norm` from the baseline's `0.0` to **`0.3`**.

The clipping path already exists in the trainer; this ablation simply activates it to test whether rare large-gradient updates are hurting convergence.

## Impact on Training

- **Parameters**: No change
- **Throughput**: Essentially unchanged
- **Peak memory**: unchanged at 10,246 MiB
- **Optimization**: Caps full-model gradient norm before optimizer steps

## Key Differences from Baseline

| Parameter | baseline-sp1024 | grad-clip-03 |
|---|---|---|
| `grad_clip_norm` | 0.0 | **0.3** |
| Everything else | identical | identical |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | grad-clip-03 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | **1.2193** | **−0.0001** |
| Val Loss | 2.0589 | **2.0587** | **−0.0002** |
| Steps completed | 13,263 | 13,170 | −93 |
| Tokens processed | 6.95B | 6.90B | −0.05B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | grad-clip-03 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | **1.2106** | **−0.0012** |
| Val Loss | 2.0460 | **2.0441** | **−0.0019** |
| Steps | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 846.9s | +14.1s |

## BPB Analysis

The effect is positive but small.

- **Fixed compute**: The improvement is effectively neutral at this budget. Clipping does not hurt, but it also does not move the needle much.
- **Fixed tokens**: There is a modest gain of 0.0012 BPB, which suggests clipping helps smooth some updates over long training, but the effect size is limited.
- **Interpretation**: `grad_clip_norm=0.3` is safe and mildly beneficial, but not a high-leverage trick in this setup.

## Files

- `train_gpt.py`: Training script (identical to baseline; `grad_clip_norm` override comes from the manifest)
- `grad-clip-03.json`: Experiment manifest
- `logs/`: Training output
