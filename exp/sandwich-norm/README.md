# Sandwich Norm — Post-Normalization Ablation

## Motivation

In standard pre-norm Transformers, each sub-layer (attention, MLP) is preceded by a normalization layer, but its output is added to the residual stream without any subsequent normalization. This leaves the magnitude of each sub-layer's contribution uncontrolled, which can lead to unstable residual stream growth in deep networks. The baseline uses plain residual connections with no magnitude control on sub-layer outputs.

**Sandwich Norm** (pre-norm + post-norm) was introduced in Gemma 2 and retained in Gemma 4 as a core architectural choice. By applying RMSNorm after each sub-layer output — before it enters the residual stream — the model explicitly constrains the output distribution to unit variance at every layer, regardless of the sub-layer's internal dynamics.

## What This Ablation Tests

This experiment isolates the effect of **post-normalization** by adding an RMSNorm after the attention output and after the MLP output in each block, while keeping all other architectural choices identical to the baseline.

| Component | Baseline | Sandwich Norm |
|---|---|---|
| Attention path | `x + attn(pre_norm(x))` | `x + post_norm(attn(pre_norm(x)))` |
| MLP path | `x + mlp(pre_norm(x))` | `x + post_norm(mlp(pre_norm(x)))` |
| Norms per block | 2 (pre-attn, pre-MLP) | **4** (pre+post attn, pre+post MLP) |

## Expected Impact

1. **Training stability**: Post-norm bounds the output magnitude of each sub-layer, preventing exponential growth or collapse of the residual stream in deeper networks. This should be particularly visible in the fixed-compute regime where training dynamics matter.

2. **Gradient flow**: By normalizing sub-layer outputs, the effective learning rate seen by downstream layers becomes more uniform, which may improve optimization landscape smoothness and reduce the need for careful scale initialization.

3. **Comparison with alternative magnitude control**: Other experiments in this suite (e.g., `scale-residual`, `zero-init`) explore different approaches to controlling sub-layer output magnitudes. Sandwich norm provides an explicit, non-parametric mechanism via RMSNorm. The ablation reveals whether post-normalization is more effective than no magnitude control at all.

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
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2757 | **-0.0181** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1540 | -0.0305 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.45B | -2.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 9,289 MiB | +10.7% |
| Fixed Compute (10 min) | Wall-clock | 600s | 600s | — |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2656 | **-0.0191** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1370 | -0.0322 |
| Fixed Tokens (10B) | Wall-clock | 771s | 790s | +2.5% |
| — | Total Params | 17.04M | 17.04M | 0 |

## Analysis

Sandwich norm improves validation BPB by 0.018 (fixed-compute) and 0.019 (fixed-tokens), with consistent loss reductions across both regimes. The gains come from constraining each sub-layer output to unit variance via post-normalization before it enters the residual stream, which prevents unbounded magnitude growth and homogenizes the effective learning rate across layers.

The improvement indicates that even at 9 layers, the residual stream can accumulate enough magnitude variance to harm optimization. The baseline has no mechanism to control sub-layer output magnitudes, so explicit RMSNorm after each sub-layer provides a significant benefit.

The overhead is modest: +900 MiB peak memory (from the extra RMSNorm buffers) and +2.5% wall-clock time in the fixed-tokens regime. Under fixed-compute, throughput drops only 2.4% (7.45B vs 7.63B tokens), and the BPB improvement more than compensates. The post-norm layers add zero parameters since the baseline's RMSNorm uses no learnable affine weight.

**Verdict**: Sandwich norm is a clear net positive at this scale -- meaningful accuracy gain, zero parameter overhead, and negligible compute cost. The result supports the Gemma design rationale that post-normalization improves residual stream dynamics regardless of depth.

## Files

- `train_gpt.py`: Training script (baseline + sandwich-norm modification)
- `sandwich-norm.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
