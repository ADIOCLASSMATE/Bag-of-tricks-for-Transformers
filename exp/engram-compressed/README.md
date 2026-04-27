# Engram-Compressed -- Engram Memory Branch With Tokenizer Compression

## Method Overview

This experiment keeps the same native Engram branch as `engram-core`, but adds **tokenizer compression** before n-gram hashing.

In the current 9-layer baseline, Engram branches are inserted at layers **1, 4, and 8**. Each branch builds **2-gram and 3-gram** hashed memory features, but instead of hashing raw token IDs directly, it first maps text-equivalent tokens to a shared compressed ID space. The compressed Engram vocabulary shrinks from the original 1024 token IDs to **766** canonical IDs.

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

## What This Ablation Tests

Whether hash-based vocabulary reduction -- mapping the full token vocabulary into a smaller hash space for the n-gram lookup -- can match or exceed the quality of the uncompressed engram-core variant while using fewer parameters.

## Impact on Training

- **Parameters**: 17.04M -> 22.62M (+5.58M, +32.7% vs baseline)
- **Peak memory**: 8,389 MiB -> 10,655 MiB (+2,266 MiB)
- **Fixed-tokens wall-clock**: 771s -> 973s (+26.2%)

## Key Differences from Baseline

| Component | Baseline | engram-compressed |
|---|---|---|
| Engram branch | none | **enabled at layers 1, 4, 8** |
| N-gram hashing | none | **2-gram and 3-gram** |
| Hash input IDs | -- | **compressed canonical token IDs** |
| Tokenizer compression | none | **enabled** |
| Hyper-connection | none | none (`hc_mult=1`) |
| Extra optimizer group | none | **Engram embedding tables use separate Adam LR** |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | Baseline | engram-compressed | Delta |
|---|---|---|---|
| **Val BPB** | 1.2938 | **1.2589** | **-0.0349** |
| Val Loss | 2.1845 | **2.1256** | -0.0589 |
| Train Tokens | 7.63B | 6.03B | -1.60B (-21.0%) |
| Peak Memory | 8,389 MiB | 10,655 MiB | +2,266 MiB |

### Fixed Tokens (10B tokens)

| Metric | Baseline | engram-compressed | Delta |
|---|---|---|---|
| **Val BPB** | 1.2847 | **1.2425** | **-0.0422** |
| Val Loss | 2.1692 | **2.0979** | -0.0713 |
| Wall-clock time | 771s | 973s | +202s (+26.2%) |
| Peak Memory | 8,389 MiB | 10,655 MiB | +2,266 MiB |

| | Total Params |
|---|---|
| Baseline | 17.04M |
| engram-compressed | 22.62M (+5.58M) |

## Analysis

**Best FT BPB in the entire ablation suite.** The fixed-tokens BPB of 1.2425 (delta -0.0422 vs baseline) is the single largest FT improvement across all experiments, making engram-compressed the most impactful trick when compute is not the bottleneck. It edges out engram-core's FT delta of -0.0403, confirming that tokenizer compression provides a tangible benefit under fixed-token conditions.

**Compression outperforms the uncompressed variant under fixed tokens.** The compressed version achieves better FT quality than engram-core (1.2425 vs 1.2444) with 1.59M fewer parameters. The hash-based vocabulary reduction maps multiple token types into shared hash buckets, which acts as a useful regularizer -- the model learns more general n-gram representations rather than overfitting to specific token sequences.

**Compute-limited regime favors the uncompressed variant.** Under fixed compute, engram-core achieves a larger BPB improvement (-0.0429 vs -0.0349). The larger embedding table in engram-core provides more granular retrieval that helps when the token budget is limited, while the compressed variant processes 21.0% fewer tokens due to higher per-step cost.

**Trade-off summary**: Engram-compressed trades a 21.0% throughput reduction and 2,266 MiB additional memory for the strongest per-token quality improvement in the suite. The compression regularizer is a net positive under fixed-token conditions, but the throughput overhead means the uncompressed variant is preferable when wall-clock time is the binding constraint.

## Files

- `train_gpt.py`: Baseline trainer with native Engram branches and tokenizer compression before hashing
- `engram-compressed.json`: Experiment manifest
- `logs/`: Training output
