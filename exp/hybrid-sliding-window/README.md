# Hybrid Sliding Window — Alternating Local/Global Attention Ablation

## Motivation

Standard causal self-attention computes pairwise interactions across the entire sequence, incurring O(n²) memory and computation cost. For long sequences, this becomes the dominant bottleneck. However, not all tokens require global context — many syntactic and semantic patterns are local in nature.

Gemma 4 introduces a **hybrid sliding window** pattern: most layers attend only within a local window (e.g., 512 tokens), while a subset of layers (every N-th layer) retain full global attention. This design exploits the observation that global information only needs to propagate through select layers, while local pattern matching suffices for the majority of layers.

## What This Ablation Tests

This experiment replaces the uniform global attention in all layers with a hybrid pattern where every 3rd layer uses global attention and the remaining 2/3 use sliding window attention (window size 512). All other architectural choices remain identical to the baseline.

| Component | Baseline | Hybrid Sliding Window |
|---|---|---|
| All layers | Full causal attention (O(n²)) | Alternating sliding/global |
| Sliding layers (0,1,3,4,6) | — | Windowed attention (w=512) |
| Global layers (2,5,8) | — | Full causal attention |
| Last layer | — | Forced global |

Layer type assignment for a 9-layer model with `sliding_window_pattern=3`:
- Layers 0, 1: **sliding** (window=512)
- Layer 2: **global**
- Layers 3, 4: **sliding** (window=512)
- Layer 5: **global**
- Layers 6, 7: **sliding** (window=512)
- Layer 8: **global** (forced)

## Expected Impact

1. **Compute efficiency**: Sliding window attention reduces per-layer computation from O(n²) to O(n·w), where w is the window size. With 6/9 layers using sliding windows, this yields a significant wall-clock speedup per step. Under fixed-compute, the model can complete more training steps, potentially seeing more data.

2. **Long-range dependency modeling**: Global layers act as information highways, allowing tokens to propagate context across the full sequence every N layers. The ablation tests whether this sparse global connectivity is sufficient for maintaining long-range dependency quality, compared to full global attention at every layer.

3. **Local pattern specialization**: Sliding layers may develop sharper local representations since their receptive field is constrained, while global layers focus on integrating information across distant positions. This specialization could improve sample efficiency for local phenomena (syntax, morphology) while preserving global coherence.

4. **Potential degradation of long-context tasks**: If the downstream evaluation requires precise attention to very distant tokens in every layer, the hybrid pattern may underperform. The 2:1 ratio of sliding-to-global layers means that global information has fewer opportunities to directly influence each token.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `sliding_window_pattern` | — | 3 |
| `sliding_window_size` | — | 512 |
| Attention computation | O(n²) × 9 layers | O(n²) × 3 + O(n·512) × 6 layers |

## Results

| Regime | Metric | Baseline | Hybrid Sliding Window | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2194 | 1.2307 | +0.0113 |
| Fixed Compute (10 min) | Train Tokens | 6.954B | 4.466B | -35.8% |
| Fixed Compute (10 min) | Peak Memory | 10,246 MiB | 22,545 MiB | +12,299 MiB (+120%) |
| Fixed Tokens (10B) | Val BPB | 1.2118 | 1.2124 | +0.0006 |
| Fixed Tokens (10B) | Wall-clock | 832.8s | 1,234.0s | +48.2% |
| — | Total Params | 17.06M | 17.06M | 0 |

## Analysis

Hybrid sliding window is the **worst-performing trick** under fixed-compute and essentially neutral under fixed-tokens, making it the least favorable ablation result in this suite.

**The compute cost dominates**: The fixed-compute result is striking — the model processes only 4.47B tokens vs. the baseline's 6.95B, a 35.8% reduction. This massive throughput loss is due to the sliding window attention implementation: the manual matmul + softmax + mask approach (necessitated by `scaled_dot_product_attention` not supporting `attn_mask` with flash kernels) falls back to an eager implementation that cannot leverage FlashAttention. The 6 sliding-window layers each incur O(n²) computation without the memory-efficient tiling of flash attention, resulting in both slower per-step time and dramatically higher memory usage.

**Memory explosion**: Peak memory doubles from 10.2 GiB to 22.5 GiB (+120%), because the sliding window attention materializes full attention score matrices in GPU memory. This is the direct consequence of not being able to use flash attention with custom masks — a practical constraint that significantly undermines the theoretical O(n·w) efficiency of sliding window attention.

**Fixed-tokens regime tells a different story**: When controlling for token count rather than compute, hybrid sliding window nearly matches baseline BPB (+0.0006), suggesting that the architectural change itself (restricting receptive field in 2/3 of layers) does not inherently degrade model quality. The slight regression may be because seq_len=1024 is shorter than the window size (512), meaning the sliding window layers are only marginally restricted — most tokens can still attend to nearly the full sequence within a 512-token window.

**Implementation caveat**: A properly optimized sliding window attention kernel (e.g., using `torch.nn.functional.scaled_dot_product_attention` with a block-sparse mask, or a custom Triton kernel) would dramatically improve both throughput and memory, potentially changing the fixed-compute result. The current result reflects the *implementation cost* of the trick, not just its *architectural merit*.

**Conclusion**: The hybrid sliding window pattern shows no architectural benefit at seq_len=1024 with w=512, and its naive implementation imposes prohibitive compute and memory costs. The trick is designed for much longer sequences (e.g., 8K–32K tokens) where the gap between full O(n²) and windowed O(n·w) attention is substantial. At this scale, the window barely constrains the receptive field, making the architectural change nearly vacuous while the implementation cost is severe. Future work should test this at longer sequence lengths with an optimized kernel.

## Files

- `train_gpt.py`: Training script (baseline + hybrid sliding window modification)
- `hybrid-sliding-window.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
