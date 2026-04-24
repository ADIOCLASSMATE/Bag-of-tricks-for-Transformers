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
| 1024 | Fixed Compute | Val Loss | 2.1914 | 2.1887 | -0.0027 |
| 1024 | Fixed Compute | Val BPB | 1.2979 | 1.2962 | -0.0017 |
| 1024 | Fixed Compute | Train Tokens | 7.67B | 7.63B | -0.5% |
| 1024 | Fixed Compute | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| 1024 | Fixed Tokens | Val Loss | 2.1709 | 2.1688 | -0.0021 |
| 1024 | Fixed Tokens | Val BPB | 1.2857 | 1.2845 | -0.0012 |
| 1024 | Fixed Tokens | Wall-clock | 772s | 770s | -0.3% |
| 2048 | Fixed Compute | Val Loss | 2.1781 | 2.1728 | -0.0053 |
| 2048 | Fixed Compute | Val BPB | 1.2900 | 1.2868 | -0.0032 |
| 2048 | Fixed Compute | Train Tokens | 6.40B | 6.43B | +0.5% |
| 2048 | Fixed Compute | Peak Memory | 8,390 MiB | 8,389 MiB | -1 MiB |
| 2048 | Fixed Tokens | Val Loss | 2.1461 | 2.1466 | +0.0005 |
| 2048 | Fixed Tokens | Val BPB | 1.2711 | 1.2714 | +0.0003 |
| 2048 | Fixed Tokens | Wall-clock | 924s | 920s | -0.4% |
| 4096 | Fixed Compute | Val Loss | 2.1752 | 2.1899 | +0.0147 |
| 4096 | Fixed Compute | Val BPB | 1.2883 | 1.2970 | +0.0087 |
| 4096 | Fixed Compute | Train Tokens | 4.91B | 4.93B | +0.4% |
| 4096 | Fixed Compute | Peak Memory | 8,392 MiB | 8,391 MiB | -1 MiB |
| 4096 | Fixed Tokens | Val Loss | 2.1306 | 2.1322 | +0.0016 |
| 4096 | Fixed Tokens | Val BPB | 1.2618 | 1.2628 | +0.0010 |
| 4096 | Fixed Tokens | Wall-clock | 1,209s | 1,202s | -0.6% |

Model size: 17,039,360 params (identical to baseline — no added parameters).

## Analysis

### Mechanism

Dual RoPE assigns different positional encoding configurations to different layers: every third layer ("global") receives θ=1M with only 25% of head dimensions rotated, while the remaining layers ("sliding") keep standard θ=10K with full rotation. The design intent is twofold: (1) global layers gain long-range positional discrimination from the large θ, and (2) the 75% unrotated dimensions preserve a position-free subspace for content-based attention. Crucially, this ablation applies only the RoPE change — all layers still use full causal attention, so the sliding/global distinction is purely in the positional encoding, not the attention pattern.

### Scaling behavior

The effect of dual RoPE is mixed and ultimately negative at longer sequences. Under fixed-compute, the improvement grows from 0.0017 BPB at 1024 tokens to 0.0032 at 2048 tokens, but under fixed-tokens the 2048 improvement is near-zero (+0.0003), and at 4096 the method regresses under both regimes (+0.0087 fixed-compute, +0.0010 fixed-tokens). The growing fixed-compute benefit at 2048 is misleading — it reflects fewer training steps due to longer sequences rather than better per-token learning. The fixed-tokens regime, which equalizes data seen, tells the clearer story: the benefit vanishes by 2048 and turns negative by 4096. This is the opposite of the intended scaling: if the large-θ global layers provided useful long-range positional signals, the benefit should grow with sequence length.

### Why partial rotation hurts under full causal attention

The regression at longer sequences reveals a fundamental mismatch between partial RoPE and full causal attention. When 75% of head dimensions carry no positional encoding, the "global" layers lose fine-grained positional discrimination in most of their attention subspace. Under full causal attention, every layer must handle both local and long-range dependencies — there are no dedicated sliding-window layers to compensate for the lost local precision. Standard RoPE with θ=10K and 100% rotation provides consistent positional signals across all dimensions, which proves more valuable than the coarse long-range signal from θ=1M in a quarter of dimensions.

### Complementarity with hybrid attention

In the full Gemma 4 design, dual RoPE is paired with a hybrid sliding-window attention pattern: sliding-window layers with full RoPE handle local precision, while global layers with partial RoPE handle long-range dependencies. This ablation isolates the RoPE change and shows that partial rotation is actively harmful when global layers must also perform precise local attention. The method likely requires the complementary attention pattern to be beneficial — the partial rotation is a feature only when the layer's attention is restricted to long-range interactions.

### Overhead

Dual RoPE adds zero parameters and negligible runtime overhead across all sequence lengths. Wall-clock differences are within 1% of the baseline, confirming that changing RoPE frequencies does not affect the dominant attention matmul cost. The method is computationally free but provides no standalone benefit.

### Verdict

**Not beneficial in isolation.** Dual RoPE slightly helps at short sequences but regresses at longer ones, contradicting its design intent. The partial rotation in global layers removes positional information that full causal attention needs. The method's value depends on pairing with hybrid sliding-window attention — without that, it is at best neutral and at worst slightly harmful.

## Files

- `train_gpt.py`: Training script (baseline + dual RoPE modification)
- `dual-rope.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
