# SEQ4096 — Sequence Length Ablation (4096)

## Method

This experiment is a **sequence length ablation** that increases `train_seq_len` from the baseline's 1024 to **4096**, while keeping all other hyperparameters (model architecture, optimizer, learning rates, `train_batch_tokens`, tokenizer, data) identical to `baseline`.

Since `train_batch_tokens` is held constant at 524,288, quadrupling the sequence length reduces the number of sequences per step to 16 per GPU (from 64), but each step still processes the same total number of tokens.

## Key Differences from Baseline

| Parameter | baseline | seq-4096 |
|---|---|---|
| `train_seq_len` | 1024 | **4096** |
| Sequences per GPU per step | 64 | **16** |
| `train_batch_tokens` | 524,288 | 524,288 (unchanged) |
| Tokens per step (total) | 524,288 | 524,288 (unchanged) |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline | seq-2048 | seq-4096 | Δ (4096 vs 1024) |
|---|---|---|---|---|
| **Val BPB** | 1.2194 | 1.2030 | **1.1973** | **−0.0221** |
| Val Loss | 2.0589 | 2.0313 | 2.0215 | −0.0374 |
| Steps completed | 13,263 | 11,556 | 8,928 | −4,335 |
| Tokens processed | 6.95B | 6.06B | 4.68B | −2.27B |

### Fixed Tokens (10B tokens)

> **Note**: The `fixed_tokens_10b` run for seq4096 has not yet completed. Results will be updated when available.

## Analysis

### Why does BPB improve with longer sequences?

1. **Extended context window**: With `seq_len=4096`, the model can attend to 4× more preceding tokens than the baseline. This allows the model to capture much longer-range dependencies (e.g., cross-paragraph coherence, multi-sentence reasoning), directly lowering prediction error for context-dependent tokens.

2. **Better gradient signal**: Each training sequence covers more contiguous text, reducing the ratio of "cold start" positions (where the model lacks context) and providing richer, more coherent gradient signals per update.

3. **Reduced boundary artifacts**: Quadrupling the sequence length reduces the number of artificial truncation boundaries to ¼ of the baseline, minimizing the fraction of tokens where the model is forced to predict without adequate context.

### Diminishing returns at longer sequences

Comparing the three sequence lengths under fixed compute:

| Seq Length | BPB | Tokens Processed | BPB Improvement vs 1024 |
|---|---|---|---|
| 1024 | 1.2194 | 6.95B | — |
| 2048 | 1.2030 | 6.06B | −0.0164 |
| 4096 | 1.1973 | 4.68B | −0.0221 |

Going from 1024→2048 yields a BPB gain of 0.0164, while 2048→4096 yields only an additional 0.0057. This suggests **diminishing returns**: the marginal benefit of longer context shrinks as the window grows, while the compute cost (O(n²) attention) increases quadratically.

### Compute trade-off

Longer sequences significantly reduce throughput. In the fixed 10-minute budget:
- **seq1024** processed 6.95B tokens (13,263 steps)
- **seq2048** processed 6.06B tokens (11,556 steps, −13% throughput)
- **seq4096** processed 4.68B tokens (8,928 steps, −33% throughput)

Despite seeing **33% fewer tokens**, seq4096 still achieves the lowest BPB under fixed compute, indicating that longer context provides a net benefit even when trading off raw throughput. However, the gap is narrowing — the fixed-tokens experiment (when completed) will clarify whether the sample-efficiency gain justifies the compute overhead at this scale.

## Files

- `train_gpt.py`: Training script (identical to baseline except `train_seq_len` override)
- `seq-4096.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
