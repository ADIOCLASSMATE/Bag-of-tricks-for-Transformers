# SEQ2048 -- Sequence Length Ablation (2048)

## Motivation

Standard causal transformers attend only within a fixed context window: each token can condition on at most `train_seq_len` preceding positions. At the baseline's `train_seq_len=1024`, many semantic dependencies that span longer distances are inevitably truncated. Document-level coherence patterns, long-range coreference resolution (e.g., pronouns referring to entities introduced many sentences earlier), and multi-clause logical dependencies can all exceed 1024 tokens, forcing the model to learn from fragmented examples that omit critical antecedent context. This truncation represents a real signal bottleneck: the model never sees the full dependency structure, so it cannot learn to model it.

Doubling the sequence length to 2048 means each token can attend to 2x more preceding context, substantially reducing how often these long-range dependencies cross a sequence boundary. Fewer sequence boundaries also mean a larger fraction of tokens are trained with a full receptive field rather than with a warm-start window that starts empty at position zero. Intuitively, richer per-sample context should produce gradients that better capture the true data distribution.

The trade-off is computational. Self-attention scales as O(n^2) in the sequence length, so doubling the window increases the cost of each forward/backward pass. Even though `train_batch_tokens` is held constant (524,288), meaning the same total number of tokens is processed per step, the longer sequences halve the number of independent sequences per micro-batch (from 64 to 32 per GPU), and the quadratic attention cost reduces tokens-per-second throughput. This design choice -- holding the token budget constant while varying only the sequence length -- isolates the pure effect of context length on learning, without confounding it with changes in total tokens seen per step.

This experiment is an ablation that answers a quantitative question: at what context length does the signal benefit of a longer window stop paying for its throughput cost? When compute is held fixed (e.g., a 600-second budget), does seq_len=2048 outperform the baseline despite processing fewer tokens? And when tokens are held fixed (e.g., a 10B token budget), how much does the longer window improve quality, and at what wall-clock premium? The answers are compared against the baseline (`exp/baseline/`) and the further doubling to `seq_len=4096` (`exp/seq-4096/`).

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

| Metric | Baseline | seq-2048 | Delta |
|---|---|---|---|
| **Val BPB** | 1.2938 | 1.2864 | **-0.0074** |
| Val Loss | 2.1845 | 2.1720 | -0.0125 |
| Train Tokens | 7.63B | 6.42B | -15.9% |
| Peak Memory | 8,389 MiB | 8,390 MiB | +1 MiB |

### Fixed Tokens (10B tokens)

| Metric | Baseline | seq-2048 | Delta |
|---|---|---|---|
| **Val BPB** | 1.2847 | 1.2663 | **-0.0184** |
| Val Loss | 2.1692 | 2.1382 | -0.0310 |
| Wall-clock | 771s | 916s | +18.8% |
| Peak Memory | 8,389 MiB | 8,390 MiB | +1 MiB |

## Analysis

Doubling the sequence length from 1024 to 2048 improves validation BPB under both evaluation regimes: -0.007 at fixed compute and -0.018 at fixed tokens. The model architecture and parameter count are identical; only the context window changes.

**Why longer context helps.** Each token can attend to 2x more preceding positions, capturing dependencies that are truncated at seq_len=1024. Doubling the window also halves the number of sequence boundaries where the model starts with no prior context, reducing the fraction of tokens trained without full receptive-field access. The resulting gradients are richer because each sample contains more contiguous context.

**Throughput cost.** Attention is O(n^2) in sequence length, so seq_len=2048 processes ~16% fewer tokens per unit wall-clock time (6.42B vs 7.63B in 600s). Despite this, per-token quality improves enough that the fixed-compute comparison still favors the longer window. At fixed tokens, the ~19% wall-clock overhead buys a -0.018 BPB improvement -- a favorable trade-off when compute budget is flexible.

**Memory impact.** Peak GPU memory is essentially unchanged (+1 MiB), because `train_batch_tokens` is held constant: the longer sequences are offset by fewer sequences per micro-batch (32 vs 64).

## Files

- `train_gpt.py`: Training script (identical to baseline except `train_seq_len` override)
- `seq-2048.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
