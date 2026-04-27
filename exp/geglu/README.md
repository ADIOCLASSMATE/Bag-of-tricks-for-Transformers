# GeGLU — Gated GeLU Activation Ablation

## Motivation

The choice of activation function in the feed-forward network (FFN) is a fundamental architectural decision that affects the model's ability to approximate nonlinear transformations. The baseline uses **GELU** (Gaussian Error Linear Unit with tanh approximation), a smooth, differentiable activation that is standard in modern transformer architectures. The baseline MLP applies GELU as a simple nonlinearity between two linear projections: `proj(gelu(fc(x)))`.

Gemma 4 uses **GeGLU** (Gated GeLU), which extends standard GELU with a multiplicative gating mechanism. The gating allows the model to learn input-dependent feature selection: one projection decides "which features to activate" (gate), while another provides "what values to pass" (up), and their element-wise product forms the gated output. This transforms GELU from a simple pointwise nonlinearity into a learned, input-dependent gating operation.

## What This Ablation Tests

This experiment replaces the standard GELU MLP with a GeGLU MLP, keeping all other architectural choices identical to the baseline. The hidden dimension remains 2x model_dim for both variants.

| Component | Baseline (GELU) | GeGLU |
|---|---|---|
| Forward | `proj(gelu(fc(x)))` | `down_proj(gelu(gate_proj(x)) * up_proj(x))` |
| Projection count | 2 (fc, proj) | 3 (gate_proj, up_proj, down_proj) |
| Activation | GELU (smooth, differentiable) | GELU (smooth, differentiable) |
| Gating | None (simple pointwise activation) | Explicit (gate x value) |

Note: GeGLU uses 3 projection matrices (gate, up, down) vs. the baseline's 2 (fc, proj), so the MLP parameter count increases. With `mlp_mult=2`, the baseline has `2 x dim^2` MLP parameters, while GeGLU has `3 x dim^2` -- a 50% increase in MLP parameters.

## Expected Impact

1. **Multiplicative gating**: The key architectural difference is not the activation function (both use GELU), but the multiplicative gating mechanism. In the baseline, GELU is applied as a simple pointwise nonlinearity to a single projection. In GeGLU, the GELU-activated gate projection is multiplied element-wise with a separate up-projection, enabling the model to learn input-dependent feature suppression and amplification. This provides a richer function approximation space where the model can selectively route information through the FFN.

2. **Increased capacity from extra projection**: The third projection (up_proj) gives the model an additional learnable linear transformation. Combined with the gating, this allows the network to represent a wider class of functions than a simple GELU MLP with the same hidden width. The gate and up projections can learn complementary feature representations whose element-wise product captures interactions that a single projection cannot.

3. **Sparsity vs. density**: Standard GELU produces near-zero but not exactly-zero outputs for strongly negative inputs. GeGLU's gating amplifies this effect: when the gate output is near zero, the entire corresponding dimension is suppressed regardless of the up-projection value. This creates a form of learned, input-dependent sparsity that is more flexible than the fixed soft-sparsity of standard GELU.

4. **Parameter efficiency trade-off**: The 50% MLP parameter increase from GeGLU must be weighed against the potential performance gain. Under fixed-compute budgets, the additional parameters slightly increase per-step computation. Under fixed-token budgets, the extra capacity may improve sample efficiency. A parameter-matched variant (GeGLU with `mlp_mult=1`) would provide a fairer comparison but is not tested here.

5. **Comparison with SwiGLU**: The ablation suite also tests SwiGLU as a known trick. GeGLU differs by using GELU instead of Swish/SiLU as the gate activation. The results inform whether the specific choice of gate nonlinearity (GELU vs. Swish) matters in this training regime. At this scale, SwiGLU's SiLU gate proves more effective than GeGLU's GeLU gate, particularly under fixed compute.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| MLP activation | GELU (pointwise) | GeGLU (gated GELU) |
| MLP projections | fc (dim->2xdim), proj | gate_proj, up_proj (dim->2xdim each), down_proj |
| MLP parameter count | 2 x dim^2 | 3 x dim^2 (+50%) |

## Results

| Regime | Metric | Baseline | GeGLU | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2755 | **-0.0183** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1536 | -0.0309 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 6.80B | -10.9% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 9,908 MiB | +1,519 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2524 | **-0.0323** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1146 | -0.0546 |
| Fixed Tokens (10B) | Wall-clock | 771s | 870s | +12.8% |
| -- | Total Params | 17.04M | 21.76M | +4.72M (+27.7%) |

## Analysis

GeGLU delivers solid improvement over the baseline: -0.018 BPB under fixed-compute and -0.032 BPB under fixed-tokens. The gain is consistent across both regimes, though the fixed-token improvement is notably stronger, reflecting the architecture's sample efficiency.

**Mechanism.** The improvement comes from the gated architecture (`gelu(gate_proj(x)) * up_proj(x))`), which provides two advantages over the baseline's standard GELU MLP (`proj(gelu(fc(x)))`). First, multiplicative gating enables input-dependent feature selection -- the model learns per-sample suppression or amplification of features, rather than applying a fixed pointwise nonlinearity uniformly. The gate and up projections can learn complementary representations whose element-wise product captures feature interactions that a single projection cannot. Second, the additional up_proj gives the model an extra learnable transformation, expanding its representational capacity within the FFN.

**Parameter overhead is not the sole cause.** GeGLU adds 4.72M parameters (+27.7%), but two observations separate the architectural effect from mere parameter scaling. Under fixed-compute, GeGLU processes 10.9% fewer tokens (6.80B vs 7.63B) due to the heavier per-step cost, yet still achieves lower BPB -- per-token learning efficiency is markedly better. Under fixed-tokens, wall-clock increases by only 12.8% versus the 27.7% parameter increase, because the extra projection parallelizes well as a matmul. A parameter-matched control would fully disentangle the effects, but the compute-constrained result already indicates the gating mechanism contributes beyond raw capacity.

**Comparison with SwiGLU.** SwiGLU outperforms GeGLU in both regimes: -0.0316 vs -0.0183 FC, -0.0341 vs -0.0323 FT. While GeGLU's FT performance is competitive, its FC gap is substantial. Both architectures have identical parameter counts (21.76M), so the difference is purely architectural. SiLU as a gate activation appears to provide better learned feature selection than GeLU at this scale, particularly when compute time is the binding constraint.

**Resource cost.** Peak memory rises ~1.5 GiB (+18.1%), from three projection matrices and their activation storage for backprop.

**Summary.** GeGLU is an architectural win over standard GELU, delivering meaningful BPB improvements in both regimes. However, among gated activations in this ablation suite, SwiGLU is the stronger choice, with a clear FC advantage and a slight FT edge.

## Files

- `train_gpt.py`: Training script (baseline + GeGLU modification)
- `geglu.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
