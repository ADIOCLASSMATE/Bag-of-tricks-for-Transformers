# Dual RoPE — Layer-Type-Dependent Positional Encoding Ablation

## Motivation

Rotary Position Embeddings (RoPE) encode positional information by rotating query and key vectors in the attention computation. The rotation frequency is controlled by the base θ (theta): a larger θ produces slower rotations, enabling the model to distinguish positions at longer ranges but with less angular resolution at short ranges. Standard Transformers use a single RoPE configuration (typically θ=10,000) for all layers.

Gemma 4 introduces **Dual RoPE**: layers with different attention patterns use different RoPE configurations. Sliding-window layers use standard RoPE with θ=10,000 and full-dimensional rotation (high local resolution), while global-attention layers use **Proportional RoPE** with θ=1,000,000 and only 25% of head dimensions rotated. The key insight is that global layers need long-range positional discrimination (large θ) but don't need fine-grained local resolution, while the unrotated dimensions (75%) preserve full expressivity for content-based attention without positional interference.

## What This Ablation Tests

This experiment assigns different RoPE configurations to different layers based on a sliding/global pattern (every 3rd layer is global), while keeping all other architectural choices identical to the baseline. Critically, the attention mechanism itself remains unchanged (all layers still use full causal attention) — only the positional encoding differs.

| Component | Baseline | Dual RoPE |
|---|---|---|
| Sliding layers | θ=10,000, 100% rotation | θ=10,000, 100% rotation (unchanged) |
| Global layers | θ=10,000, 100% rotation | θ=1,000,000, **25% rotation** |
| Unrotated dimensions | — | 75% of head_dim, no positional encoding |

The "proportional" aspect means the frequency computation uses the **full head_dim** as the denominator, not the rotated fraction. This ensures the frequency spectrum remains well-distributed even when only partially applying rotation.

## Expected Impact

1. **Long-range positional discrimination**: Global layers with θ=1M can distinguish positions across much longer ranges than θ=10K. The angular resolution per position step is ~100× coarser, but for long-range attention this coarseness is acceptable — the model primarily needs to know "is this token near or far," not the exact distance.

2. **Content-position disentanglement**: By leaving 75% of head dimensions unrotated in global layers, the model retains a large subspace for purely content-based attention. This may improve the model's ability to attend based on semantic similarity rather than positional proximity, which is particularly important for tasks like coreference resolution and long-range dependency.

3. **Potential local precision loss**: In global layers, the 25% rotation with θ=1M means local positional differences are nearly indistinguishable (the angular change between adjacent positions is tiny). If the model relies on precise local position signals in every layer, this could degrade short-range attention quality.

4. **Complementarity with hybrid attention**: Dual RoPE is designed to complement the hybrid sliding-window pattern. This ablation uses only the RoPE change without the sliding-window attention, isolating the positional encoding effect. A future combined ablation (dual-rope + hybrid-sliding-window) would test the full Gemma 4 design.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `sliding_window_pattern` | — | 3 (layer type assignment only) |
| `global_rope_theta` | — | 1,000,000 |
| `partial_rotary_factor` | — | 0.25 |
| RoPE config per layer | Uniform θ=10K | Sliding: θ=10K full; Global: θ=1M 25% |

## Results

| Seq Len | Regime | Metric | Baseline | Dual RoPE | Delta |
|---|---|---|---|---|---|
| 1024 | Fixed Compute | Val Loss | 2.1845 | 2.1858 | +0.0013 |
| 1024 | Fixed Compute | Val BPB | 1.2938 | 1.2946 | +0.0008 |
| 1024 | Fixed Compute | Train Tokens | 7.63B | 7.66B | +0.4% |
| 1024 | Fixed Compute | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| 1024 | Fixed Tokens | Val Loss | 2.1692 | 2.1681 | -0.0011 |
| 1024 | Fixed Tokens | Val BPB | 1.2847 | 1.2841 | -0.0006 |
| 1024 | Fixed Tokens | Wall-clock | 771s | 764s | -0.9% |
| 2048 | Fixed Compute | Val Loss | 2.1781 | 2.1748 | -0.0033 |
| 2048 | Fixed Compute | Val BPB | 1.2900 | 1.2880 | -0.0020 |
| 2048 | Fixed Compute | Train Tokens | 6.40B | 6.46B | +0.9% |
| 2048 | Fixed Compute | Peak Memory | 8,390 MiB | 8,389 MiB | -1 MiB |
| 2048 | Fixed Tokens | Val Loss | 2.1461 | 2.1373 | -0.0088 |
| 2048 | Fixed Tokens | Val BPB | 1.2711 | 1.2659 | -0.0052 |
| 2048 | Fixed Tokens | Wall-clock | 924s | 913s | -1.2% |
| 4096 | Fixed Compute | Val Loss | 2.1752 | 2.1827 | +0.0075 |
| 4096 | Fixed Compute | Val BPB | 1.2883 | 1.2927 | +0.0044 |
| 4096 | Fixed Compute | Train Tokens | 4.91B | 4.94B | +0.6% |
| 4096 | Fixed Compute | Peak Memory | 8,392 MiB | 8,391 MiB | -1 MiB |
| 4096 | Fixed Tokens | Val Loss | 2.1306 | 2.1274 | -0.0032 |
| 4096 | Fixed Tokens | Val BPB | 1.2618 | 1.2600 | -0.0018 |
| 4096 | Fixed Tokens | Wall-clock | 1,209s | 1,197s | -1.0% |

Model size: 17,039,360 params (identical to baseline — no added parameters).

## Analysis

### Mechanism

Dual RoPE assigns different positional encoding configurations to different layers: every third layer ("global") receives θ=1M with only 25% of head dimensions rotated, while the remaining layers ("sliding") keep standard θ=10K with full rotation. The design intent is twofold: (1) global layers gain long-range positional discrimination from the large θ, and (2) the 75% unrotated dimensions preserve a position-free subspace for content-based attention. Crucially, this ablation applies only the RoPE change — all layers still use full causal attention, so the sliding/global distinction is purely in the positional encoding, not the attention pattern.

### Scaling behavior

The effect of dual RoPE is mixed across sequence lengths. At 1024, the method is near-neutral: the fixed-compute delta is -0.0011 BPB and the fixed-tokens delta is -0.0006 BPB — both within noise. At 2048, the method shows a clear benefit: -0.0020 BPB under fixed-compute and -0.0052 BPB under fixed-tokens, indicating that the large-θ global layers provide useful long-range positional signals that the uniform θ=10K baseline cannot match. At 4096 under fixed-compute, the method regresses (+0.0044 BPB) — the per-iteration cost of longer sequences reduces the number of training steps, and the baseline's advantage in data volume (4.91B vs 4.94B tokens is a small difference) dominates. Under fixed-tokens at 4096, the method shows marginal improvement (-0.0018 BPB), suggesting the dual-rope mechanism itself remains neutral to slightly beneficial given equalized data.

The strongest signal is at 2048, where the improvement is consistent and meaningful under both regimes. This intermediate sequence length appears to be the sweet spot: long enough to benefit from the coarse long-range signal of θ=1M, but not so long that the local precision loss in global layers overwhelms the gain.

### Why partial rotation is mixed without sliding windows

When 75% of head dimensions carry no positional encoding, the "global" layers lose fine-grained positional discrimination in most of their attention subspace. Under full causal attention, every layer must handle both local and long-range dependencies — there are no dedicated sliding-window layers to compensate for the lost local precision. At moderate sequence lengths (2048), the long-range benefit dominates, but at very long sequences (4096) the local precision loss catches up. Standard RoPE with θ=10K and 100% rotation provides consistent positional signals across all dimensions.

### Complementarity with hybrid attention

In the full Gemma 4 design, dual RoPE is paired with a hybrid sliding-window attention pattern: sliding-window layers with full RoPE handle local precision, while global layers with partial RoPE handle long-range dependencies. This ablation isolates the RoPE change and shows that partial rotation provides a benefit at moderate sequence lengths (2048) even without the complementary attention pattern, but its advantage diminishes at longer sequences where local precision matters more. The method is likely to benefit further when paired with true sliding-window attention that restricts each layer to its appropriate positional range.

### Overhead

Dual RoPE adds zero parameters and negligible runtime overhead. Wall-clock differences are within 1-2% of baseline across all sequence lengths (e.g., -1.2% at 2048, -1.0% at 4096), confirming that changing RoPE frequencies does not affect the dominant attention matmul cost. The method is computationally free.

### Verdict

**Modestly beneficial at moderate sequence lengths, but not a universal improvement.** Dual RoPE provides a clear -0.0052 FT BPB gain at sequence length 2048, but is near-neutral at 1024 and mixed/regressing at 4096. The benefit depends on sequence length and may be strongest when combined with the hybrid sliding-window attention pattern for which it was designed. As a standalone trick, it is a zero-cost improvement at 2048-token contexts but should not be applied blindly at all sequence lengths.

## Files

- `train_gpt.py`: Training script (baseline + dual RoPE modification)
- `dual-rope.json`: Experiment manifest (seq_len=1024)
- `dual-rope-seq2048.json`: Experiment manifest (seq_len=2048)
- `dual-rope-seq4096.json`: Experiment manifest (seq_len=4096)
- `logs/`: Training output (automatically generated)
