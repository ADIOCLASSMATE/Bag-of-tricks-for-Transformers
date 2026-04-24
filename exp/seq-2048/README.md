# SEQ2048 — Sequence Length Ablation (2048)

## Method

This experiment is a **sequence length ablation** that increases `train_seq_len` from the baseline's 1024 to **2048**, while keeping all other hyperparameters (model architecture, optimizer, learning rates, `train_batch_tokens`, tokenizer, data) identical to `baseline`.

Since `train_batch_tokens` is held constant at 524,288, doubling the sequence length halves the number of sequences per step (from 64 to 32 per GPU), but each step still processes the same total number of tokens.

## Key Differences from Baseline

| Parameter | baseline | seq-2048 |
|---|---|---|
| `train_seq_len` | 1024 | **2048** |
| Sequences per GPU per step | 64 | **32** |
| `train_batch_tokens` | 524,288 | 524,288 (unchanged) |
| Tokens per step (total) | 524,288 | 524,288 (unchanged) |

## Results

Model size: 17.04M parameters (unchanged from baseline).

### Fixed Compute (600s wall-clock)

| Metric | Baseline | seq-2048 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2979 | 1.2900 | **-0.0079** |
| Val Loss | 2.1914 | 2.1781 | -0.0133 |
| Train Tokens | 7.67B | 6.40B | -16.6% |
| Peak Memory | 8,389 MiB | 8,390 MiB | +1 MiB |

### Fixed Tokens (10B tokens)

| Metric | Baseline | seq-2048 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2857 | 1.2711 | **-0.0146** |
| Val Loss | 2.1709 | 2.1461 | -0.0248 |
| Wall-clock | 772s | 924s | +19.7% |
| Peak Memory | 8,389 MiB | 8,390 MiB | +1 MiB |

## Analysis

Doubling the sequence length from 1024 to 2048 improves validation BPB under both evaluation regimes: -0.008 at fixed compute and -0.015 at fixed tokens. The model architecture and parameter count are identical; only the context window changes.

**Why longer context helps.** Each token can attend to 2x more preceding positions, capturing dependencies that are truncated at seq_len=1024. Doubling the window also halves the number of sequence boundaries where the model starts with no prior context, reducing the fraction of tokens trained without full receptive-field access. The resulting gradients are richer because each sample contains more contiguous context.

**Throughput cost.** Attention is O(n^2) in sequence length, so seq_len=2048 processes ~20% fewer tokens per unit wall-clock time (6.40B vs 7.67B in 600s). Despite this, per-token quality improves enough that the fixed-compute comparison still favors the longer window. At fixed tokens, the 20% wall-clock overhead buys a -0.015 BPB improvement -- a favorable trade-off when compute budget is flexible.

**Memory impact.** Peak GPU memory is essentially unchanged (+1 MiB), because `train_batch_tokens` is held constant: the longer sequences are offset by fewer sequences per micro-batch (32 vs 64).

## Files

- `train_gpt.py`: Training script (identical to baseline except `train_seq_len` override)
- `seq-2048.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
