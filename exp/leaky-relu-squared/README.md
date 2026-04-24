# Leaky-ReLU-Squared — leaky_relu(x, 0.5).square() Activation

## Method Overview

This experiment replaces the baseline MLP activation `F.gelu(x)` with **`leaky_relu(x, 0.5).square()`**.

The goal is to test whether a quadratic activation that grows on both sides of zero provides stronger gradient signal for large activations compared to GELU's approximately linear regime.

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

| Metric | Baseline | leaky-relu-squared | Δ |
|---|---|---|---|
| **Val BPB** | 1.2979 | **1.2755** | **−0.0224** |
| Val Loss | 2.1914 | **2.1536** | **−0.0378** |
| Tokens processed | 7.67B | 7.61B | −0.8% |
| Peak Memory | 8,389 MiB | 8,389 MiB | 0 |

### Fixed Tokens (10B tokens)

| Metric | Baseline | leaky-relu-squared | Δ |
|---|---|---|---|
| **Val BPB** | 1.2857 | **1.2654** | **−0.0203** |
| Val Loss | 2.1709 | **2.1366** | **−0.0343** |
| Wall-clock time | 772s | 780s | +1.0% |
| Peak Memory | 8,389 MiB | 8,389 MiB | 0 |

## Analysis

Leaky-ReLU-squared is a consistent, zero-cost improvement over the GELU baseline. By applying `leaky_relu(x, 0.5).square()`, the activation grows quadratically on both sides of zero, providing stronger gradient signal for large activations compared to GELU's approximately linear regime near and beyond zero. GELU already provides non-zero gradient for negative inputs, so the advantage here is not about eliminating dead zones but about the quadratic scaling that amplifies gradients for larger pre-activations, potentially leading to more expressive feature transformations.

Both evaluation regimes tell the same story: a ~0.02 BPB gain that is robust to the compute-vs-tokens tradeoff. Throughput and memory are unaffected.

## Files

- `train_gpt.py`: Modified training script with `leaky_relu(x, 0.5).square()` in `MLP.forward`
- `leaky-relu-squared.json`: Experiment manifest
- `logs/`: Training output
