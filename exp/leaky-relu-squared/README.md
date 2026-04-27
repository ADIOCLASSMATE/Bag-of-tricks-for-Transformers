# Leaky-ReLU-Squared -- leaky_relu(x, 0.5).square() Activation

## Method Overview

This experiment replaces the baseline MLP activation `F.gelu(x)` with **`leaky_relu(x, 0.5).square()`**.

The goal is to test whether a quadratic activation that grows on both sides of zero provides stronger gradient signal for large activations compared to GELU's approximately linear regime.

## Motivation

### GELU's Limitation

GELU approximates `x * Phi(x)` (where `Phi` is the Gaussian CDF), producing a smooth, non-monotonic activation. For large |x|, GELU approaches the identity function -- it is approximately linear for large positive x and approximately zero for large negative x. In the linear regime, the gradient `dGELU/dx` approaches 1 (positive side) or 0 (negative side), meaning the gradient magnitude does not scale with activation magnitude. Large pre-activations and small pre-activations contribute equally to the upstream gradient, which can dilute the training signal from important features.

### Leaky ReLU with Negative Slope 0.5

The standard ReLU activation zeros out all negative inputs, which can create "dead" units -- neurons whose pre-activation is consistently negative, producing zero output and zero gradient, permanently disabling the unit. Leaky ReLU addresses this with a non-zero slope `alpha` for negative inputs: `leaky_relu(x) = x` for `x > 0`, `alpha * x` for `x <= 0`.

We choose `alpha = 0.5`, which is intentionally larger than the commonly used 0.01. This aggressive slope means that half of the negative signal passes through, strongly preventing dead units rather than merely tolerating them. The value 0.5 balances dead-unit prevention with a degree of sparsity: negative activations are attenuated (not zeroed), so the representation still carries information about which inputs are negative, but positive inputs receive twice the weight.

### Why Squaring?

Applying `.square()` after `leaky_relu(x, 0.5)` produces `x^2` for positive x and `0.25 * x^2` for negative x. The quadratic growth on both sides means:

1. **Stronger gradient signal for large activations**: The gradient is `2x` (positive side) or `0.5x` (negative side), so large pre-activations produce proportionally larger gradients. This amplifies the learning signal for features that strongly activate a neuron, causing the optimizer to prioritize learning from the most informative patterns.

2. **Symmetric quadratic regime**: Both positive and negative pre-activations contribute proportionally to their magnitude squared. Unlike ReLU^2 which discards all negative information as zeros, Leaky-ReLU^2 preserves negative activations at 1/4 strength (since `0.5^2 = 0.25`), giving the network access to the full pre-activation distribution.

3. **Soft sparsity**: Near-zero pre-activations produce near-zero outputs (quadratic squashing around zero), naturally inducing a form of soft sparsity without hard zeroing. This may act as an implicit regularizer similar to ReLU^2's exact sparsity, but with smoother gradients.

### Comparison to ReLU^2

ReLU^2 (`relu(x).square()`) has proven effective in the modded-nanogpt speedrun community, achieving -0.0144 BPB over GELU under fixed compute in this benchmark (see `exp/relu-sq/README.md`). However, ReLU^2 has a hard zero threshold: all negative pre-activations produce exactly zero output and zero gradient, permanently losing that information. Leaky-ReLU^2 preserves the quadratic amplification benefit while retaining negative-side information at reduced magnitude, trading ReLU^2's exact sparsity for more complete representational capacity.

## Impact on Training

- **Parameters**: No change
- **Throughput**: Essentially unchanged
- **Peak memory**: unchanged at 8,389 MiB
- **Optimization**: Quadratic growth on both sides provides stronger gradient signal for large activations vs. GELU's near-linear regime

## Key Differences from Baseline

| Component | Baseline | leaky-relu-squared |
|---|---|---|
| MLP activation | `F.gelu(x)` | **`leaky_relu(x, 0.5).square()`** |
| Everything else | identical | identical |

## Results

**Parameters**: 17.04M (identical to baseline)

### Fixed Compute (10 min wall-clock)

| Metric | Baseline | leaky-relu-squared | Delta |
|---|---|---|---|
| **Val BPB** | 1.2938 | **1.2779** | **-0.0159** |
| Val Loss | 2.1845 | **2.1576** | **-0.0269** |
| Tokens processed | 7.63B | 7.60B | -0.4% |
| Peak Memory | 8,389 MiB | 8,389 MiB | 0 |

### Fixed Tokens (10B tokens)

| Metric | Baseline | leaky-relu-squared | Delta |
|---|---|---|---|
| **Val BPB** | 1.2847 | **1.2708** | **-0.0139** |
| Val Loss | 2.1692 | **2.1457** | **-0.0235** |
| Wall-clock time | 771s | 774s | +0.4% |
| Peak Memory | 8,389 MiB | 8,389 MiB | 0 |

## Analysis

Leaky-ReLU-squared improves over the GELU baseline by **0.014-0.016 BPB** with zero parameter or memory cost and negligible throughput change. This places it as one of the strongest single-trick improvements in the benchmark.

The mechanism operates at two levels. First, the quadratic scaling provides **gradient amplification** -- large pre-activations produce proportionally larger gradients, causing the optimizer to focus on neurons that are already strongly activated by the data. This creates a positive feedback loop: features that match the input well get stronger updates, while weak or ambiguous features receive gentler updates. In contrast, GELU's approximately linear gradient for large |x| treats all activations equally, diluting the training signal.

Second, the leaky negative path preserves **bidirectional information flow**. Unlike ReLU^2 which zeros negative activations entirely, Leaky-ReLU^2 retains negative pre-activations at 25% magnitude (since `0.5^2 = 0.25`). The network can thus represent features at any point on the real line rather than only in the non-negative half-space. For a transformer MLP processing post-attention representations, this may be particularly valuable: the attention mechanism can produce both positive and negative correlations between token positions, and Leaky-ReLU^2 lets the MLP process both symmetries.

The FC improvement (-0.0159) is slightly larger than the FT improvement (-0.0139), suggesting the quadratic activation's benefit is most pronounced when compute time is the binding constraint -- the stronger per-token gradient signal extracts more value from each training step within a fixed time budget. In the FT regime, both models process the same total tokens, and the quadratic activation's representational advantage still yields a meaningful but slightly smaller gain.

Computationally, `leaky_relu(x, 0.5).square()` involves one comparison, one multiply-branch, and one element-wise square -- cheaper than GELU's erf-based approximation. However, as with ReLU^2, this per-step cost advantage is negligible relative to the full training step and does not translate into a wall-clock speed difference in practice.

In summary, Leaky-ReLU^2 combines the quadratic gradient amplification of ReLU^2 with Leaky ReLU's preservation of negative-side information. The result is a stronger activation function that outperforms both plain GELU and ReLU^2 while adding no parameters, no memory, and no throughput overhead.

## Files

- `train_gpt.py`: Modified training script with `leaky_relu(x, 0.5).square()` in `MLP.forward`
- `leaky-relu-squared.json`: Experiment manifest
- `logs/`: Training output
