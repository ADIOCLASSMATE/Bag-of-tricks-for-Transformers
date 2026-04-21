# GeGLU — Gated GeLU Activation Ablation

## Motivation

The choice of activation function in the feed-forward network (FFN) is a fundamental architectural decision that affects the model's ability to approximate nonlinear transformations. The baseline uses **ReLU²** (squared ReLU), a simple nonlinearity from the modded-nanogpt speedrun community that provides sparse, non-negative gating with quadratic growth. While effective in practice, ReLU² has a sharp non-differentiable point at zero and its quadratic growth can lead to large output magnitudes that require careful scaling.

Gemma 4 uses **GeGLU** (Gated GeLU), which combines a smooth, differentiable activation (GeLU with tanh approximation) with a multiplicative gating mechanism. The gating allows the model to learn input-dependent feature selection: one projection decides "which features to activate" (gate), while another provides "what values to pass" (up), and their element-wise product forms the gated output.

## What This Ablation Tests

This experiment replaces the ReLU² MLP with a GeGLU MLP, keeping all other architectural choices identical to the baseline. The hidden dimension remains 2× model_dim for both variants.

| Component | Baseline (ReLU²) | GeGLU |
|---|---|---|
| Forward | `down_proj(relu(fc(x)).square())` | `down_proj(gelu(gate_proj(x)) * up_proj(x))` |
| Projection count | 2 (fc, down_proj) | 3 (gate_proj, up_proj, down_proj) |
| Activation | ReLU (hard threshold at 0) | GeLU (smooth, differentiable) |
| Gating | Implicit (relu × relu) | Explicit (gate × value) |

Note: GeGLU uses 3 projection matrices (gate, up, down) vs. the baseline's 2 (fc, down), so the MLP parameter count increases. With `mlp_mult=2`, the baseline has `2 × dim²` MLP parameters, while GeGLU has `3 × dim²` — a 50% increase in MLP parameters.

## Expected Impact

1. **Nonlinear approximation**: GeGLU's smooth, gated activation may provide a richer function approximation space than ReLU². The multiplicative gating allows the model to learn input-dependent feature suppression and amplification, rather than the fixed quadratic scaling of ReLU². This could improve the model's ability to represent complex token interactions.

2. **Gradient smoothness**: GeLU is differentiable everywhere (unlike ReLU's discontinuous derivative at 0), which may lead to smoother gradient landscapes and more stable optimization, particularly with the Muon optimizer that relies on gradient spectral properties.

3. **Sparsity vs. density**: ReLU² naturally produces sparse activations (exact zeros for negative inputs), which can act as an implicit regularizer. GeGLU with its smooth activation produces dense outputs, potentially increasing the model's effective capacity but possibly requiring more data to avoid overfitting.

4. **Parameter efficiency trade-off**: The 50% MLP parameter increase from GeGLU must be weighed against the potential performance gain. Under fixed-compute budgets, the additional parameters slightly increase per-step computation. Under fixed-token budgets, the extra capacity may improve sample efficiency. A parameter-matched variant (GeGLU with `mlp_mult=1`) would provide a fairer comparison but is not tested here.

5. **Comparison with SwiGLU**: The TRICK_SUMMARY already records SwiGLU as a known trick. GeGLU differs by using GeLU instead of Swish/SiLU as the gate activation. The ablation informs whether the specific choice of gate nonlinearity (GeLU vs. Swish vs. ReLU²) matters in this training regime.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| MLP activation | ReLU² | GeGLU (GeLU-tanh gating) |
| MLP projections | fc (dim→2×dim), down_proj | gate_proj, up_proj (dim→2×dim each), down_proj |
| MLP parameter count | 2 × dim² | 3 × dim² (+50%) |

## Results

| Regime | Metric | Baseline | GeGLU | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2194 | 1.2022 | **-0.0172** |
| Fixed Compute (10 min) | Train Tokens | 6.954B | 6.218B | -10.6% |
| Fixed Compute (10 min) | Peak Memory | 10,246 MiB | 11,637 MiB | +1,391 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2118 | 1.1910 | **-0.0208** |
| Fixed Tokens (10B) | Wall-clock | 832.8s | 952.1s | +14.3% |
| — | Total Params | 17.06M | 21.78M | +27.6% |

## Analysis

GeGLU is the **strongest single-trick improvement** in this ablation suite, achieving -0.017 BPB under fixed-compute and -0.021 BPB under fixed-tokens. The improvement is substantial and consistent across both regimes.

**Why GeGLU works so well**: The gated architecture provides two key advantages over ReLU². First, the multiplicative gate (`gelu(gate_proj(x)) * up_proj(x)`) enables input-dependent feature selection — the model can learn to suppress or amplify specific features on a per-sample basis, rather than applying the fixed quadratic nonlinearity of ReLU² to all inputs uniformly. Second, GeLU's smooth differentiability eliminates the gradient discontinuity at zero that ReLU introduces, which likely produces smoother gradient landscapes that are better exploited by the Muon optimizer's spectral normalization.

**The parameter increase is not the full story**: GeGLU adds 27.6% more parameters (4.7M), which could be mistaken as the sole reason for improvement. However, two observations argue against this. Under fixed-compute, GeGLU processes 10.6% fewer tokens (6.22B vs 6.95B) due to the heavier forward/backward pass, yet still achieves significantly lower BPB — meaning the per-token learning efficiency is dramatically better. Under fixed-tokens, the 14.3% wall-clock increase is proportionally much smaller than the 27.6% parameter increase, because the additional projection is a matmul that parallelizes well on H100s. A parameter-matched control (e.g., baseline with wider MLP) would help disentangle the effects, but the strong result even under compute-constrained conditions suggests the architectural change itself contributes beyond mere parameter scaling.

**Memory**: Peak memory increases by ~1.4 GiB (13.6%), attributable to the three projection matrices vs. two and the corresponding activation storage for backprop.

**Conclusion**: GeGLU is a clear win for this training regime. The smooth, gated activation provides meaningful improvements in both compute-efficiency and sample-efficiency, despite the parameter overhead. This aligns with the broader industry trend of adopting gated activations (SwiGLU in LLaMA, GeGLU in Gemma) as a standard architectural choice.

## Files

- `train_gpt.py`: Training script (baseline + GeGLU modification)
- `geglu.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
