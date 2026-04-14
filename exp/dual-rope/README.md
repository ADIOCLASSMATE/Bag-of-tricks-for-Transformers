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

| Regime | Metric | Baseline | Dual RoPE | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2194 | 1.2203 | +0.0009 |
| Fixed Compute (10 min) | Train Tokens | 6.954B | 6.953B | ~0 |
| Fixed Compute (10 min) | Peak Memory | 10,246 MiB | 10,246 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2118 | 1.2121 | +0.0003 |
| Fixed Tokens (10B) | Wall-clock | 832.8s | 848.4s | +1.9% |
| — | Total Params | 17.06M | 17.06M | 0 |

## Analysis

Dual RoPE is **essentially neutral** — the BPB deltas are +0.0009 (fixed-compute) and +0.0003 (fixed-tokens), both within the noise margin of a single training run. The trick produces no measurable benefit or harm.

**Why dual RoPE has no effect here**: The dual RoPE design assigns θ=1M with 25% partial rotation to "global" layers and θ=10K with 100% rotation to "sliding" layers. However, in this ablation, all layers still use **full causal attention** — the sliding/global distinction only affects the RoPE configuration, not the attention pattern. At seq_len=1024, the position range is short enough that even θ=10K provides adequate positional discrimination across the full sequence. The θ=1M configuration with partial rotation was designed for sequences of 8K–32K+ tokens, where the standard θ=10K RoPE suffers from excessive angular overlap at distant positions. At 1024 tokens, both configurations effectively provide the same positional information quality.

**The 75% unrotated dimensions are underutilized**: Leaving 75% of head dimensions unrotated in global layers creates a large content-attention subspace, but at seq_len=1024 where positional resolution is already high, there is little to gain from reducing positional interference. The unrotated dimensions simply carry the same information they would with standard RoPE, since the positional signal in the rotated 25% is already sufficient for the short sequence.

**No throughput cost**: The trick adds zero parameters and negligible wall-clock overhead (1.9% in fixed-tokens), confirming that the RoPE configuration change is computationally free. This makes it a "free lunch" trick if it provides any benefit at longer sequence lengths.

**Complementarity note**: Dual RoPE is designed to work in conjunction with hybrid sliding-window attention — global layers with partial RoPE handle long-range dependencies while sliding layers with full RoPE focus on local precision. Testing dual RoPE in isolation, without the corresponding attention pattern change, may underestimate its potential. The partial rotation in global layers makes the most sense when those layers are specifically responsible for integrating information across the full sequence, which is the intended design in Gemma 4.

**Conclusion**: Dual RoPE produces no measurable effect at seq_len=1024, which is expected given the design targets much longer sequences. The trick is likely sequence-length-dependent and should be re-evaluated at seq_len ≥ 4096 where the positional encoding capacity of θ=10K becomes a limiting factor.

## Files

- `train_gpt.py`: Training script (baseline + dual RoPE modification)
- `dual-rope.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
