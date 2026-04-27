# SEQ4096 — Sequence Length Ablation (4096)

## Motivation

The seq-2048 ablation demonstrated that doubling the context window from 1024 to 2048 tokens yields a meaningful validation BPB improvement (-0.018 FT) with a modest 18.9% wall-clock cost. This raises a natural follow-up question: **does the benefit scale further with a 4x context window, or does it saturate?**

Many real-world language dependencies span well beyond 2048 tokens. Document-level structure, narrative arcs spanning multiple paragraphs, code files with cross-file references, and long-form reasoning chains can all require attention across thousands of tokens. At `seq_len=1024`, the model sees at most a few paragraphs of context; at 2048, it captures roughly a page of prose; but at 4096, the receptive field begins to encompass meaningful document-scale structure. Each token can now attend to **4x more preceding positions** than the baseline, potentially surface richer long-range signals during training.

However, this comes at a steep computational price. Self-attention is O(n^2) in sequence length, so quadrupling the context window increases the per-sequence attention cost by **16x** relative to the baseline. This drastically reduces throughput: the same wall-clock budget processes far fewer tokens. The experiment therefore tests the extreme end of the sequence-length spectrum to determine whether the benefit of longer context eventually plateaus, or whether the trade-off remains favorable even at 4096.

As with seq-2048, `train_batch_tokens` is held constant at 524,288 across all experiments. This isolates the effect of sequence length: any difference in validation quality comes purely from how tokens are arranged into sequences, not from the total data volume processed per step.

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
| **Val BPB** | 1.2938 | 1.2975 | **+0.0037** |
| Val Loss | 2.1845 | 2.1907 | +0.0062 |
| Train Tokens | 7.63B | 4.91B | -35.6% |
| Peak Memory | 8,389 MiB | 8,392 MiB | +3 MiB |

### Fixed Tokens (10B tokens)

| Metric | Baseline | seq-4096 | Delta |
|---|---|---|---|
| **Val BPB** | 1.2847 | 1.2644 | **-0.0203** |
| Val Loss | 2.1692 | 2.1348 | -0.0344 |
| Wall-clock | 771s | 1,208s | +56.7% |
| Peak Memory | 8,389 MiB | 8,392 MiB | +3 MiB |

## Analysis

Quadrupling the sequence length from 1024 to 4096 improves validation quality under fixed-tokens but hurts under fixed-compute. The mechanism is straightforward: each training example now spans 4x more contiguous tokens, giving the model longer-range dependencies to learn from. The trade-off is quadratic O(n^2) attention cost, which significantly reduces throughput per step.

**Fixed-compute.** Processing 35.6% fewer tokens (4.91B vs 7.63B) due to the O(n^2) attention overhead causes a small regression (+0.0037 BPB). Under a tight time budget, the longer context does not fully compensate for the substantial reduction in gradient updates -- the model simply sees too few tokens for the longer-range benefits to outweigh the throughput loss.

**Fixed-tokens.** With equal token budgets, seq-4096 delivers a large improvement: -0.0203 BPB, one of the strongest fixed-tokens gains among all ablation experiments. The cost is 57% more wall-clock time. This confirms that when compute throughput is not the binding constraint, longer context is very effective for improving language modeling quality.

**Diminishing returns across sequence lengths.** The seq-2048 ablation yields -0.018 FT BPB at +18.9% wall-clock, while the further doubling to 4096 adds an additional -0.002 BPB at +31.9% more wall-clock. The marginal return per compute unit shrinks, but the cumulative -0.0203 BPB gain from 1024 to 4096 remains among the strongest ablation results.

**Memory overhead is negligible.** Peak memory increases by only 3 MiB because `train_batch_tokens` is held constant; the smaller per-GPU batch size (16 vs 64 sequences) offsets the longer per-sequence activation storage.

## Files

- `train_gpt.py`: Training script (identical to baseline except `train_seq_len` override)
- `seq-4096.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
