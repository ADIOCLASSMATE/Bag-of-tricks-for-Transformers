# Lower-LR-003 — Moderately Reduced Learning Rates

## Method Overview

This experiment is a **补充消融**，用于在 baseline 与 `lower-lr` 之间补一个中间 LR 点。它把主要学习率降低到大约 baseline 的 75%：

- `matrix_lr`: 0.04 → **0.03**
- `scalar_lr`: 0.04 → **0.03**
- `tied_embed_lr`: 0.05 → **0.04**

这样可以更细地观察 LR 下降曲线，而不是只比较 baseline 和“直接减半”的 `lower-lr`。

## LR Sweep 对照

| 实验 | matrix_lr | scalar_lr | tied_embed_lr | 相对 baseline |
|---|---|---|---|---|
| baseline-sp1024 | 0.04 | 0.04 | 0.05 | 1× |
| **lower-lr-003** | **0.03** | **0.03** | **0.04** | **~0.75×** |
| lower-lr | 0.02 | 0.02 | 0.03 | ~0.5× |

## Impact on Training

- **Parameters**: No change
- **Throughput**: Essentially unchanged
- **Peak memory**: unchanged at 10,246 MiB
- **Optimization**: 比 baseline 更保守，但又没有像 `lower-lr` 那样把步长压得太低

## Key Differences from Baseline

| Parameter | baseline-sp1024 | lower-lr-003 |
|---|---|---|
| `matrix_lr` | 0.04 | **0.03** |
| `scalar_lr` | 0.04 | **0.03** |
| `tied_embed_lr` | 0.05 | **0.04** |
| Everything else | identical | identical |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | lower-lr-003 | lower-lr | Δ (003 vs baseline) |
|---|---|---|---|---|
| **Val BPB** | 1.2194 | **1.2175** | 1.2182 | **−0.0018** |
| Val Loss | 2.0589 | **2.0558** | 2.0569 | **−0.0031** |
| Steps completed | 13,263 | 13,313 | 13,327 | +50 |
| Tokens processed | 6.95B | 6.98B | 6.99B | +0.03B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | lower-lr-003 | lower-lr | Δ (003 vs baseline) |
|---|---|---|---|---|
| **Val BPB** | 1.2118 | **1.2097** | 1.2100 | **−0.0021** |
| Val Loss | 2.0460 | **2.0425** | 2.0430 | **−0.0035** |
| Steps | 19,074 | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 840.3s | 838.6s | +7.5s |

## BPB Analysis

这个中间 LR 点的结果是有价值的，而且比 `lower-lr` 更好。

- **Fixed compute**: `lower-lr-003` 相比 baseline 提升 0.0018 BPB，也略好于 `lower-lr` 的 0.0012。说明在固定 10 分钟预算下，baseline 的默认 LR 确实略高，但“直接减半”不是最优点。
- **Fixed tokens**: 在相同 10B token 下，`lower-lr-003` 达到 1.2097 BPB，也小幅优于 `lower-lr` 的 1.2100。差距不大，但方向一致。
- **Interpretation**: 当前结果更支持“适度降 LR”而不是“激进降 LR”。如果后续只保留一个 LR trick，`lower-lr-003` 比 `lower-lr` 更值得优先保留。

## Files

- `train_gpt.py`: Training script (identical to baseline; LR overrides come from the manifest)
- `lower-lr-003.json`: Experiment manifest
- `logs/`: Training output
