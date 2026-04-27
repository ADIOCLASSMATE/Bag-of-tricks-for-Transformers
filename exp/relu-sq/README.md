# ReLU^2 -- Squared ReLU Activation Ablation

## Method Overview

This experiment replaces the baseline's GELU activation with **ReLU^2** (squared ReLU) in the MLP. The forward pass becomes `proj(relu(fc(x)).square())` instead of `proj(gelu(fc(x)))`.

### Motivation

ReLU^2 is a cheaper alternative to GELU with competitive performance, popular in the modded-nanogpt speedrun community. Beyond the superficial appeal of saving a few FLOPs, the activation has deeper structural properties that make it an interesting design choice.

GELU applies a smooth, non-negative gating function g(x) = x * Phi(x) that tapers to near-zero for negative inputs rather than hitting zero. In theory, this preserves small negative contributions, giving the network richer gradient signal and avoiding the "dead neuron" problem. In practice, however, those small activations may act as noise: the MLP's output is added directly into the residual stream, and every active feature dimension -- regardless of how weak -- contributes to the representation that downstream layers must disentangle. Preserving every whisper of signal can increase interference between features.

ReLU^2 takes the opposite approach. Its hard threshold at zero creates exact sparsity: roughly half of pre-activations become zero (assuming a zero-mean, roughly symmetric pre-activation distribution), eliminating cross-feature interference from weak or negative signals entirely. This sparsity acts as an implicit regularizer -- the network is forced to represent information using a subset of available feature dimensions rather than spreading signal thinly across all of them. Fewer active units per token means less interference and a cleaner residual stream for the next layer.

The squaring operation adds a second effect: it amplifies large positive activations quadratically while further suppressing values near zero. A pre-activation of 0.1 becomes 0.01 after squaring (negligible), while a pre-activation of 2.0 becomes 4.0 (dominant). This creates a sharper "signal threshold" where only the most confident features propagate with strength, and the rest are either zeroed or squashed. Important features become proportionally more dominant.

Together, sparsity and quadratic gain produce a **high-contrast representation**. Below zero, there is nothing. Near zero, there is almost nothing. Well above zero, there is amplified signal. The MLP output becomes sparse and concentrated: only the features that "fire strongly" contribute meaningfully to the residual stream, while noise and marginal signals are suppressed. This winner-take-most dynamic differs fundamentally from GELU's approach of letting everything through at varying strength.

This is a zero-cost change: same parameter count (17.04M), same memory footprint (8,389 MiB), and ReLU^2 is strictly cheaper per FLOP than GELU's erf-based approximation (hard threshold + element-wise square vs. the Gaussian CDF approximation). There are no additional parameters, no architectural modifications, and no memory overhead. It is a pure activation swap in one line of code.

Where ReLU^2 sits in the activation landscape is instructive. GELU represents the dense, smooth extreme: every pre-activation contributes, and the transition from negative to positive is gradual. Leaky-ReLU^2 (another ablation in this project) preserves negative signal with a scaled negative slope (0.5x) before squaring, so both positive and negative activations affect the output quadratically -- it keeps the quadratic gain but rejects sparsity. ReLU^2 chooses the opposite tradeoff: it accepts the information loss from zeroing negatives in exchange for exact sparsity, betting that the regularization benefit of silencing marginal features outweighs the lost signal. The empirical results suggest this bet pays off: ReLU^2 achieves better BPB than GELU within the same compute budget, and is competitive with leaky-ReLU^2 while being conceptually simpler (no negative slope hyperparameter to tune).

## What This Ablation Tests

| Component | Baseline | ReLU^2 |
|---|---|---|
| MLP activation | GELU | `relu(x).square()` |
| Sparsity | Dense (smooth activation) | Exact zeros for negative inputs |
| Gradient | Smooth everywhere | Discontinuous at zero |

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| MLP activation | GELU | ReLU^2 |
| Code change | -- | MLP.forward: `relu(fc(x)).square()` instead of `gelu(fc(x))` |

## Results

| Regime | Metric | Baseline | ReLU^2 | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2794 | **-0.0144** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1603 | -0.0242 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.63B | 0% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Compute (10 min) | Wall-clock | 600s | 600s | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2753 | **-0.0094** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1533 | -0.0159 |
| Fixed Tokens (10B) | Wall-clock | 771s | 768s | -0.4% |
| -- | Total Params | 17.04M | 17.04M | 0 |

## Analysis

ReLU^2 improves over GELU by **0.009-0.014 BPB** in both evaluation regimes with no increase in parameters or memory.

The mechanism is twofold. First, ReLU introduces exact sparsity by zeroing negative pre-activations, which acts as an implicit regularizer -- GELU's smooth taper preserves small negative contributions that add noise. Second, the squaring operation amplifies large positive activations quadratically while suppressing values near zero, creating a high-contrast representation that concentrates signal into fewer active units. The combination yields sparser, more informative MLP outputs for the downstream residual path.

Computationally, ReLU^2 is strictly cheaper than GELU per step (hard threshold + element-wise square vs. erf-based approximation), but this does not translate into a wall-clock advantage in the fixed-compute regime where both models train for the same 600s time budget. Instead, ReLU^2 achieves lower validation BPB (1.2794 vs 1.2938) within that identical time budget -- a quality improvement, not a speed improvement. The two models process the same volume of tokens (7.63B) because both hit the 600s limit. The per-step speedup from ReLU^2's cheaper activation is negligible relative to the full training step.

ReLU^2 is a drop-in replacement that trades GELU's smooth gradients for exact sparsity and quadratic gain. It achieves better BPB than GELU within the same compute budget at no extra parameter or memory cost. The FC improvement (-0.0144) is substantially larger than previously reported, making ReLU^2 a much stronger trick than originally assessed.

## Files

- `train_gpt.py`: Training script (baseline + ReLU^2 modification)
- `relu-sq.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
