# muP-Scaled-Output-Projections — Depth-Scaled Projection Initialization

## Method Overview

This experiment applies a muP-style output-projection scaling after initialization: all 18 output
projection layers (9 `attn.proj` + 9 `mlp.proj`) are multiplied by `1 / sqrt(2 * L)`, where `L`
is the number of transformer blocks. With `L = 9`, the scaling factor is `1 / sqrt(18) ≈ 0.236`.

We scale all output projection weights by `1/sqrt(N)` where `N` is the total number of residual
branches, following muP (maximal update parameterization).

### Origin

From later **parameter-golf** combo records that use **OrthoInit + muP-scaled output projections**.
This experiment isolates the projection-scaling part from the orthogonal-init part.

### Motivation

- Smaller output-projection scale can reduce early residual-branch amplitude
- The trick is initialization-only, so it does not change the runtime graph
- Isolating it from OrthoInit makes the interaction testable

## Impact on Training

- **Parameters**: No change
- **Throughput**: No measurable runtime impact after initialization
- **Memory**: No change
- **Optimization**: All 18 `.proj` weights are rescaled by `1/sqrt(2L)` at initialization

## Important Note

ALL 18 output projection layers (9 attn.proj + 9 mlp.proj) are scaled by 1/sqrt(18) ≈ 0.236 at
initialization. The baseline uses default PyTorch initialization for these projections (no special
scaling), so this trick affects every residual branch in the model.

## Key Differences from Baseline

| Component | Baseline | mup-scaled-output-projections |
|---|---|---|
| Output projection (`.proj`) weights | default init scale | **multiplied by `1 / sqrt(2L)` ≈ 0.236** |
| Runtime graph | unchanged | unchanged |
| Everything else | identical | identical |

## Results

| Regime | Metric | Baseline | muP-Scaled | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2977 | **+0.0039** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1911 | +0.0066 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.62B | -0.1% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2829 | **-0.0018** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1661 | -0.0031 |
| Fixed Tokens (10B) | Wall-clock | 771s | 772s | +0.1% |
| — | Total Params | 17.04M | 17.04M | 0 |

## Analysis

The muP output-projection scaling multiplies all 18 output projection weights by `1/sqrt(2L) ≈ 0.236` at initialization. The intent is to dampen residual-branch amplitude early in training so that deeper stacks do not explode, inspired by muP's width- and depth-aware scaling rules.

The trick is NOT inert -- it affects every output projection in the model. Results are regime-dependent: a moderate regression under fixed-compute (+0.0039 BPB) but a mild improvement under fixed-tokens (-0.0018 BPB). The FC regression suggests the `1/sqrt(2L)` scaling (shrinking initial residual contributions by roughly 4.2x) slows early convergence compared to default initialization, hurting quality in the time-limited regime. However, under FT, the slightly dampened residual branches may allow for marginally better long-run convergence when given enough training tokens.

The scaling factor `1/sqrt(18) ≈ 0.236` may be too aggressive for a shallow 9-layer model, where the default Kaiming/Xavier initialization already produces reasonable residual magnitudes. The mild FT improvement hints that some degree of projection downscaling could be beneficial with more data, but the current factor is suboptimal for early training throughput. The trick could still prove beneficial in deeper models where uncontrolled residual growth is more of a concern.

## Files

- `train_gpt.py`: Baseline trainer with isolated muP-style output projection scaling
- `mup-scaled-output-projections.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
