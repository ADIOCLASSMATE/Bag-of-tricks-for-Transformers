# Engram-Core — Native Engram Memory Branch Without Compression or Hyper-Connection

## Method Overview

This experiment inserts a native **Engram memory branch** into the baseline `GPT` blocks instead of wrapping the model with `engram-peft`.

In the current 9-layer baseline, Engram branches are inserted at layers **1, 4, and 8**. Each branch builds **2-gram and 3-gram** hashed memory features directly from the raw token IDs, then mixes the retrieved values back through a residual path.

This `core` variant is the cleanest Engram ablation:

- It keeps the baseline training loop, data pipeline, validation BPB metric, and result reporting unchanged
- It uses raw token IDs for n-gram hashing
- It disables tokenizer compression
- It does not include hyper-connection (not implemented in this codebase)

### Origin

This is an in-framework adaptation of the official TinyEngram idea in `/inspire/hdd/global_user/tangkezheng-253108540239/TinyEngram/engram.py`, rewritten to run inside the Bag-of-tricks baseline trainer.

### Motivation

- Test whether the Engram memory branch itself is useful without conflating it with tokenizer compression
- Measure Engram on the same trainer and evaluation harness as the baseline
- Isolate the architectural value of hashed n-gram memory before adding more aggressive coupling mechanisms

## Impact on Training

- **Parameters**: 17.04M -> 24.21M (+42.1%, +7.17M)
- **Peak memory**: 8,389 MiB -> 10,666 MiB (+2,277 MiB)
- **Fixed-compute throughput**: ~6.01B tokens in 10 minutes (vs 7.67B baseline, -21.6%)
- **Fixed-tokens wall-clock**: 772s -> 982s (+27.2%)

## Key Differences from Baseline

| Component | Baseline | engram-core |
|---|---|---|
| Engram branch | none | **enabled at layers 1, 4, 8** |
| N-gram hashing | none | **2-gram and 3-gram** |
| Hash input IDs | — | **raw token IDs** |
| Tokenizer compression | none | none |
| Hyper-connection | none | not implemented |
| Extra optimizer group | none | **Engram embedding tables use separate Adam LR** |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | Baseline | engram-core | Delta |
|---|---|---|---|
| **Val BPB** | 1.2979 | **1.2534** | **-0.0445** |
| Val Loss | 2.1914 | **2.1163** | -0.0751 |
| Tokens processed | 7.67B | 6.01B | -1.66B (-21.6%) |
| Peak Memory | 8,389 MiB | 10,666 MiB | +2,277 MiB |

### Fixed Tokens (10B tokens)

| Metric | Baseline | engram-core | Delta |
|---|---|---|---|
| **Val BPB** | 1.2857 | **1.2466** | **-0.0391** |
| Val Loss | 2.1709 | **2.1049** | -0.0660 |
| Wall-clock time | 772s | 982s | +210s (+27.2%) |
| Peak Memory | 8,389 MiB | 10,666 MiB | +2,277 MiB |

| Total Params | Baseline | engram-core | Delta |
|---|---|---|---|
| — | 17.04M | 24.21M | +7.17M (+42.1%) |

## Analysis

Engram is one of the strongest tricks in the benchmark: -0.0445 FC BPB, -0.039 FT BPB.

**Why it works.** The n-gram hash-based retrieval mechanism provides a direct lookup channel for recurring token patterns, complementing the attention mechanism's content-based retrieval. This is particularly powerful for language modeling where n-gram statistics are highly predictive. The short convolution in the engram branch provides temporal smoothing of the retrieved embeddings, allowing the model to integrate retrieved information over a local context window rather than applying it pointwise. The engram mechanism's value lies in its orthogonality to attention: attention captures long-range content dependencies while engram captures local statistical patterns. Their combination is more powerful than either alone.

**Cost.** The improvement is dramatic but comes at significant cost: +7.17M params (+42%), +2.3GB memory, +27% wall-clock. Under fixed compute, the model processes 21.6% fewer tokens due to the overhead of the engram branch — yet still improves by -0.0445 BPB, indicating the n-gram retrieval signal is extremely valuable per token.

## Files

- `train_gpt.py`: Baseline trainer with native Engram branches added to selected Transformer blocks
- `engram-core.json`: Experiment manifest
- `logs/`: Training output
