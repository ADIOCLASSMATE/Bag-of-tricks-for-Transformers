# Baseline SEQ2048 — Sequence Length Ablation (2048)

## Method

This experiment is a **sequence length ablation** that increases `train_seq_len` from the baseline's 1024 to **2048**, while keeping all other hyperparameters (model architecture, optimizer, learning rates, `train_batch_tokens`, tokenizer, data) identical to `baseline-sp1024`.

Since `train_batch_tokens` is held constant at 524,288, doubling the sequence length halves the number of sequences per step (from 64 to 32 per GPU), but each step still processes the same total number of tokens.

## Key Differences from Baseline

| Parameter | baseline-sp1024 | baseline-seq2048 |
|---|---|---|
| `train_seq_len` | 1024 | **2048** |
| Sequences per GPU per step | 64 | **32** |
| `train_batch_tokens` | 524,288 | 524,288 (unchanged) |
| Tokens per step (total) | 524,288 | 524,288 (unchanged) |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | baseline-seq2048 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | **1.2030** | **−0.0164** |
| Val Loss | 2.0589 | 2.0313 | −0.0276 |
| Steps completed | 13,263 | 11,556 | −1,707 |
| Tokens processed | 6.95B | 6.06B | −0.90B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | baseline-seq2048 | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | **1.1917** | **−0.0201** |
| Val Loss | 2.0460 | 2.0121 | −0.0339 |
| Steps | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 974.4s | +141.6s |

## Analysis

### Why does BPB improve with longer sequences?

1. **Longer context window**: With `seq_len=2048`, the model can attend to 2× more preceding tokens, capturing longer-range dependencies. This directly benefits next-token prediction for tokens that depend on distant context, lowering the per-byte prediction error.

2. **Better gradient signal**: Each training sequence contains more contiguous context, providing richer and more coherent gradient signals per sample. This can lead to more effective parameter updates compared to shorter, more fragmented sequences.

3. **Reduced boundary artifacts**: Shorter sequences introduce more artificial truncation boundaries where the model has no context at the start of each sequence. Doubling the sequence length halves the number of such "cold start" positions per batch.

### Why are steps identical under fixed-tokens?

Steps are determined by `total_tokens / train_batch_tokens`. Since both experiments use the same `train_batch_tokens` (524,288) and target the same 10B tokens, both require exactly ⌈10B / 524,288⌉ = 19,074 steps.

### Compute trade-off

Longer sequences increase per-step computation (attention is O(n²) in sequence length), resulting in ~17% more wall-clock time (974s vs 833s) for the same number of tokens. Despite processing **fewer tokens** in the fixed-compute regime (6.06B vs 6.95B), the seq2048 model still achieves a **lower BPB**, indicating that the quality gain from longer context outweighs the throughput loss.

## Files

- `train_gpt.py`: Training script (identical to baseline except `train_seq_len` override)
- `baseline-seq2048.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
