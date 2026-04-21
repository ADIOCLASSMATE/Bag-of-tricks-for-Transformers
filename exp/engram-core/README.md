# Engram-Core — Native Engram Memory Branch Without Compression or Hyper-Connection

## Method Overview

This experiment inserts a native **Engram memory branch** into the baseline `GPT` blocks instead of wrapping the model with `engram-peft`.

In the current 9-layer baseline, Engram branches are inserted at layers **1, 4, and 8**. Each branch builds **2-gram and 3-gram** hashed memory features directly from the raw token IDs, then mixes the retrieved values back through a residual path.

This `core` variant is the cleanest Engram ablation:

- It keeps the baseline training loop, data pipeline, validation BPB metric, and result reporting unchanged
- It uses raw token IDs for n-gram hashing
- It disables tokenizer compression
- It disables hyper-connection by keeping `hc_mult=1`

### Origin

This is an in-framework adaptation of the official TinyEngram idea in `/inspire/hdd/global_user/tangkezheng-253108540239/TinyEngram/engram.py`, rewritten to run inside the Bag-of-tricks baseline trainer.

### Motivation

- Test whether the Engram memory branch itself is useful without conflating it with tokenizer compression
- Measure Engram on the same trainer and evaluation harness as `baseline-sp1024`
- Isolate the architectural value of hashed n-gram memory before adding more aggressive coupling mechanisms

## Impact on Training

- **Parameters**: 17.06M -> 24.23M (+42.1%)
- **Peak memory**: 10,246 MiB -> 12,581 MiB
- **Fixed-compute throughput**: 13,263 -> 10,596 steps in 10 minutes
- **Fixed-tokens wall-clock**: 832.8s -> 1053.3s

## Key Differences from Baseline

| Component | baseline-sp1024 | engram-core |
|---|---|---|
| Engram branch | none | **enabled at layers 1, 4, 8** |
| N-gram hashing | none | **2-gram and 3-gram** |
| Hash input IDs | — | **raw token IDs** |
| Tokenizer compression | none | none |
| Hyper-connection | none | none (`hc_mult=1`) |
| Extra optimizer group | none | **Engram embedding tables use separate Adam LR** |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | engram-core | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | **1.2057** | **−0.0137** |
| Val Loss | 2.0589 | **2.0358** | **−0.0231** |
| Steps completed | 13,263 | 10,596 | −2,667 |
| Tokens processed | 6.95B | 5.56B | −1.40B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | engram-core | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | **1.1910** | **−0.0208** |
| Val Loss | 2.0460 | **2.0109** | **−0.0351** |
| Steps | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 1053.3s | +220.5s |

## BPB Analysis

This is a clear positive result.

- **Fixed compute**: Even after processing 1.40B fewer tokens in the same 10-minute budget, the model still improves BPB by 0.0137. The Engram branch is paying for its own throughput cost.
- **Fixed tokens**: The gain grows to 0.0208 BPB under equal 10B-token budget, which is the cleaner signal that the native Engram branch improves sample efficiency.
- **Interpretation**: The core Engram memory mechanism is effective on this benchmark even before adding tokenizer compression or hyper-connection.

## Files

- `train_gpt.py`: Baseline trainer with native Engram branches added to selected Transformer blocks
- `engram-core.json`: Experiment manifest
- `logs/`: Training output
