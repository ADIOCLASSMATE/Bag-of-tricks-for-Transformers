# Engram-Compressed — Engram Memory Branch With Tokenizer Compression

## Method Overview

This experiment keeps the same native Engram branch as `engram-core`, but adds **tokenizer compression** before n-gram hashing.

In the current 9-layer baseline, Engram branches are inserted at layers **1, 4, and 8**. Each branch builds **2-gram and 3-gram** hashed memory features, but instead of hashing raw token IDs directly, it first maps text-equivalent tokens to a shared compressed ID space. In this setup, the compressed Engram vocabulary shrinks from the original 1024 token IDs to **766** canonical IDs.

Relative to `engram-core`, this experiment changes only one variable:

- tokenizer compression is enabled before Engram hashing
- hyper-connection remains disabled with `hc_mult=1`
- the training loop and evaluation protocol remain identical to baseline

### Origin

This variant is derived from the official TinyEngram design in `/inspire/hdd/global_user/tangkezheng-253108540239/TinyEngram/engram.py`, but implemented directly inside the Bag-of-tricks baseline trainer for controlled ablation.

### Motivation

- Test whether canonical token remapping improves Engram memory sharing
- Reduce duplication across text-equivalent token forms before 2-gram and 3-gram hashing
- Measure whether compression gives extra BPB gains beyond `engram-core`

## Impact on Training

- **Parameters**: 17.06M -> 24.23M (+42.1%)
- **Peak memory**: 10,246 MiB -> 12,582 MiB
- **Fixed-compute throughput**: 13,263 -> 10,660 steps in 10 minutes
- **Fixed-tokens wall-clock**: 832.8s -> 1051.8s

## Key Differences from Baseline

| Component | baseline-sp1024 | engram-compressed |
|---|---|---|
| Engram branch | none | **enabled at layers 1, 4, 8** |
| N-gram hashing | none | **2-gram and 3-gram** |
| Hash input IDs | — | **compressed canonical token IDs** |
| Tokenizer compression | none | **enabled** |
| Hyper-connection | none | none (`hc_mult=1`) |
| Extra optimizer group | none | **Engram embedding tables use separate Adam LR** |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | engram-compressed | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | **1.2053** | **−0.0141** |
| Val Loss | 2.0589 | **2.0351** | **−0.0238** |
| Steps completed | 13,263 | 10,660 | −2,603 |
| Tokens processed | 6.95B | 5.59B | −1.36B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | engram-compressed | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | **1.1904** | **−0.0214** |
| Val Loss | 2.0460 | **2.0099** | **−0.0361** |
| Steps | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 1051.8s | +219.0s |

## BPB Analysis

This is the best completed Engram variant.

- **Fixed compute**: BPB improves by 0.0141 despite the throughput penalty, slightly better than `engram-core`.
- **Fixed tokens**: The gain reaches 0.0214 BPB, again slightly ahead of `engram-core` at 1.1910 BPB.
- **Interpretation**: Tokenizer compression is a real but small positive term. The main gain still comes from the Engram branch itself, while compression provides a modest extra improvement at nearly the same cost.

## Files

- `train_gpt.py`: Baseline trainer with native Engram branches and tokenizer compression before hashing
- `engram-compressed.json`: Experiment manifest
- `logs/`: Training output
