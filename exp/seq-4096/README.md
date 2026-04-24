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

Model size: 17.04M parameters (identical to baseline).

### Fixed Compute (10 min wall-clock)

| Metric | Baseline | seq-4096 | Delta |
|---|---|---|---|
| **Val BPB** | 1.2979 | 1.2883 | **-0.0096** |
| Val Loss | 2.1914 | 2.1752 | -0.0162 |
| Train Tokens | 7.67B | 4.91B | -36.0% |
| Peak Memory | 8,389 MiB | 8,392 MiB | +3 MiB |

### Fixed Tokens (10B tokens)

| Metric | Baseline | seq-4096 | Delta |
|---|---|---|---|
| **Val BPB** | 1.2857 | 1.2618 | **-0.0239** |
| Val Loss | 2.1709 | 2.1306 | -0.0403 |
| Wall-clock | 772s | 1,209s | +56.6% |
| Peak Memory | 8,389 MiB | 8,392 MiB | +3 MiB |

## Analysis

Quadrupling the sequence length from 1024 to 4096 improves validation quality in both evaluation regimes. The mechanism is straightforward: each training example now spans 4x more contiguous tokens, giving the model longer-range dependencies to learn from. The trade-off is quadratic attention cost, which reduces throughput per step.

**Fixed-compute.** Despite processing 36% fewer tokens (4.91B vs 7.67B) due to the O(n^2) attention overhead, seq-4096 still achieves -0.010 BPB improvement. Longer context more than compensates for fewer gradient updates -- each step produces higher-quality gradients because the loss is computed over more coherent, longer spans.

**Fixed-tokens.** With equal token budgets the improvement is substantial: -0.024 BPB, the largest fixed-tokens gain among all ablation experiments. The cost is 57% more wall-clock time. This confirms that when compute is not the binding constraint, longer context is one of the most effective ways to improve language modeling quality.

**Diminishing returns across sequence lengths.** The seq-2048 ablation yields -0.015 FT BPB at +19.7% wall-clock, while the further doubling to 4096 adds -0.009 BPB at an additional +36.9% wall-clock. The marginal return per compute unit halves, but the total -0.024 BPB gain from 1024 to 4096 remains the strongest single ablation result.

**Memory overhead is negligible.** Peak memory increases by only 3 MiB because `train_batch_tokens` is held constant; the smaller per-GPU batch size (16 vs 64 sequences) offsets the longer per-sequence activation storage.

## Files

- `train_gpt.py`: Training script (identical to baseline except `train_seq_len` override)
- `seq-4096.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
