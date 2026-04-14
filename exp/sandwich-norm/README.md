# Sandwich Norm — Post-Normalization Ablation

## Motivation

In standard pre-norm Transformers, each sub-layer (attention, MLP) is preceded by a normalization layer, but its output is added to the residual stream without any subsequent normalization. This leaves the magnitude of each sub-layer's contribution uncontrolled, which can lead to unstable residual stream growth in deep networks. The baseline already addresses this partially through learnable `attn_scale`/`mlp_scale` with zero-initialized output projections, but these are scalar vectors that cannot fully normalize the output distribution.

**Sandwich Norm** (pre-norm + post-norm) was introduced in Gemma 2 and retained in Gemma 4 as a core architectural choice. By applying RMSNorm after each sub-layer output — before it enters the residual stream — the model explicitly constrains the output distribution to unit variance at every layer, regardless of the sub-layer's internal dynamics.

## What This Ablation Tests

This experiment isolates the effect of **post-normalization** by adding an RMSNorm after the attention output and after the MLP output in each block, while keeping all other architectural choices identical to the baseline.

| Component | Baseline | Sandwich Norm |
|---|---|---|
| Attention path | `x + scale(attn(pre_norm(x)))` | `x + scale(post_norm(attn(pre_norm(x))))` |
| MLP path | `x + scale(mlp(pre_norm(x)))` | `x + scale(post_norm(mlp(pre_norm(x))))` |
| Norms per block | 2 (pre-attn, pre-MLP) | **4** (pre+post attn, pre+post MLP) |

## Expected Impact

1. **Training stability**: Post-norm bounds the output magnitude of each sub-layer, preventing exponential growth or collapse of the residual stream in deeper networks. This should be particularly visible in the fixed-compute regime where training dynamics matter.

2. **Gradient flow**: By normalizing sub-layer outputs, the effective learning rate seen by downstream layers becomes more uniform, which may improve optimization landscape smoothness and reduce the need for careful scale initialization.

3. **Interaction with existing mechanisms**: The baseline already uses `attn_scale`/`mlp_scale` (learnable per-dim scalars) and zero-initialized output projections for implicit magnitude control. Sandwich norm provides an alternative, potentially more robust mechanism. The ablation reveals whether explicit normalization is more effective than learned scaling.

4. **Expressivity trade-off**: Post-norm constrains the output distribution, which could limit the model's ability to propagate sharp, high-magnitude signals through the residual stream. In early training, this might slightly slow convergence if the model needs to learn to bypass the normalization effect.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `post_attn_norm` | — | RMSNorm (no learnable weight) |
| `post_mlp_norm` | — | RMSNorm (no learnable weight) |
| Additional parameters | — | 0 (RMSNorm with no affine weight) |

Note: The baseline's `RMSNorm` implementation uses `F.rms_norm` without a learnable scale, so post-norm adds zero additional parameters.

## Results

| Regime | Metric | Baseline | Sandwich Norm | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2194 | 1.2210 | +0.0016 |
| Fixed Compute (10 min) | Train Tokens | 6.954B | 6.827B | -1.8% |
| Fixed Compute (10 min) | Peak Memory | 10,246 MiB | 10,250 MiB | +4 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2118 | 1.2127 | +0.0010 |
| Fixed Tokens (10B) | Wall-clock | 832.8s | 861.8s | +3.5% |
| — | Total Params | 17.06M | 17.06M | 0 |

## Analysis

Sandwich norm produces a **slight regression** in both regimes, with BPB increasing by 0.001–0.002 over baseline. The result is consistent across fixed-compute and fixed-tokens, indicating the degradation is not merely a throughput artifact.

The most likely explanation is an **interaction with the baseline's existing magnitude control mechanisms**. The baseline already employs `attn_scale`/`mlp_scale` (learnable per-dimension scalars) and zero-initialized output projections, which together provide an adaptive, learned mechanism for controlling sub-layer output magnitudes. Sandwich norm's rigid, non-parametric RMSNorm constrains outputs to unit variance regardless of what the learned scales would prefer, effectively overriding a more flexible mechanism with a fixed one. The two mechanisms are in tension: the learned scales try to adjust magnitudes dynamically while post-norm forces them back to unit variance, creating a conflict that slightly harms optimization.

The fixed-compute regime shows marginally fewer tokens processed (6.83B vs 6.95B), reflecting the negligible overhead of the two additional RMSNorm operations per block. The fixed-tokens regime shows 3.5% longer wall-clock time, consistent with this overhead.

**Conclusion**: At this scale (9 layers, 512-dim), sandwich norm provides no benefit over the baseline's existing learned scaling mechanism. The result suggests that post-norm may only become beneficial in deeper networks where the residual stream magnitude grows more explosively, or when the baseline lacks adaptive scaling mechanisms. The Gemma 4 design context (much deeper networks, no `attn_scale`/`mlp_scale`) likely makes sandwich norm more valuable there than in this setting.

## Files

- `train_gpt.py`: Training script (baseline + sandwich-norm modification)
- `sandwich-norm.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
