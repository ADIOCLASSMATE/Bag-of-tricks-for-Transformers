# Engram-HC — Engram Memory Branch With Tokenizer Compression and Hyper-Connection

## Method Overview

This experiment extends `engram-compressed` by enabling **hyper-connection coupling** inside the Engram path.

In the current 9-layer baseline, Engram branches are inserted at layers **1, 4, and 8**. Each branch still builds **2-gram and 3-gram** hashed memory features from compressed canonical token IDs, but hidden states are expanded into a multi-branch representation with **`hc_mult=4`**. The Engram branch then applies branch-wise gating, value mixing, and short convolution over the full hyper-connection tensor.

Relative to `engram-compressed`, the incremental change is the hyper-connection structure itself.

### Origin

This variant is the closest Bag-of-tricks rewrite of the official TinyEngram hyper-connection path in `/inspire/hdd/global_user/tangkezheng-253108540239/TinyEngram/engram.py`.

### Motivation

- Test whether Engram benefits from multi-branch hidden-state coupling rather than a single residual branch
- Measure whether hyper-connection provides additional gains beyond memory lookup plus tokenizer compression
- Evaluate the full Engram-style coupling mechanism under the same baseline trainer and BPB protocol

## Impact on Training

- **Parameters**: 17.06M -> 25.45M (+49.2%)
- **Peak memory**: 10,246 MiB -> 48,416 MiB
- **Fixed-compute throughput**: 13,263 -> 2,900 steps in 10 minutes
- **Fixed-tokens wall-clock**: 832.8s -> 3921.7s

## Key Differences from Baseline

| Component | baseline-sp1024 | engram-hc |
|---|---|---|
| Engram branch | none | **enabled at layers 1, 4, 8** |
| N-gram hashing | none | **2-gram and 3-gram** |
| Hash input IDs | — | **compressed canonical token IDs** |
| Tokenizer compression | none | **enabled** |
| Hyper-connection | none | **enabled with `hc_mult=4`** |
| Engram branch shape | — | **multi-branch hidden states with branch-wise gating and short conv** |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | engram-hc | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | **1.3183** | **+0.0989** |
| Val Loss | 2.0589 | **2.2258** | **+0.1670** |
| Steps completed | 13,263 | 2,900 | −10,363 |
| Tokens processed | 6.95B | 1.52B | −5.43B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | engram-hc | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | **1.2513** | **+0.0395** |
| Val Loss | 2.0460 | **2.1127** | **+0.0667** |
| Steps | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 3921.7s | +3088.9s |

## BPB Analysis

This variant is clearly unsuccessful in its current form.

- **Fixed compute**: The quality collapse is severe. BPB worsens by 0.0989 and throughput drops to only 2,900 steps in 10 minutes, so the model loses both efficiency and final quality.
- **Fixed tokens**: Even after seeing the same 10B tokens, BPB is still worse by 0.0395. This rules out a pure fixed-time artifact and indicates that the hyper-connection configuration itself is harming training.
- **Interpretation**: Relative to `engram-compressed`, the added hyper-connection coupling is not a marginal miss but a large negative change under the current `hc_mult=4` setup.

## Files

- `train_gpt.py`: Baseline trainer with native Engram branches, tokenizer compression, and hyper-connection-aware gating
- `engram-hc.json`: Experiment manifest
- `logs/`: Training output
