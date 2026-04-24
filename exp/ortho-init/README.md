# Ortho-Init — Orthogonal Initialization for All Linear Layers

## Method Overview

This experiment applies `nn.init.orthogonal_` to **every linear weight matrix** in the model, replacing PyTorch's default Kaiming uniform initialization with orthogonal initialization.

All linear layers — including attention and MLP projections — switch from default PyTorch init to orthogonal init. This matches the earlier standalone parameter-golf OrthoInit formulation more closely than later stacked variants that also rescaled projection weights.

### Origin

From **parameter-golf T07 OrthoInit**. The source-faithful standalone formulation initializes all linear layers with `nn.init.orthogonal_()`.

### Motivation

- Orthogonal weights preserve directional structure better than generic random initializations
- The change targets only linear layer initialization, not runtime computation
- Replacing default init with orthogonal init isolates the effect of initialization geometry alone

## Impact on Training

- **Parameters**: No change
- **Throughput**: No measurable runtime impact after initialization
- **Memory**: No change
- **Optimization**: All linear maps start orthogonal instead of using PyTorch's default Kaiming uniform init

## Key Differences from Baseline

| Component | Baseline | ortho-init |
|---|---|---|
| Linear weight init | default PyTorch init (Kaiming uniform) | **orthogonal** |
| Runtime graph | unchanged | unchanged |
| Everything else | identical | identical |

## Results

| Regime | Metric | Baseline | Ortho-Init | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2995 | +0.0016 |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1942 | +0.0028 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.64B | -0.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2886 | +0.0029 |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1757 | +0.0048 |
| Fixed Tokens (10B) | Wall-clock | 772s | 775s | +0.4% |
| — | Total Params | 17.04M | 17.04M | 0 |

## Analysis

Orthogonal initialization degrades validation BPB by +0.0016 (fixed compute) and +0.0029 (fixed tokens), a consistent though small regression across both evaluation regimes.

The baseline already incorporates QK-Norm, RMSNorm, and the Muon optimizer, each of which normalizes gradient flow during training. Orthogonal initialization aims to improve conditioning at the start of training, but these runtime normalization mechanisms largely subsume that benefit. The default Kaiming uniform initialization already provides adequate initial conditioning for this architecture, and orthogonal init offers no additional advantage.

The method is parameter-free and adds no runtime cost, but offers no advantage at this scale. Orthogonal initialization is **near-neutral to slightly harmful** when strong normalization is already present.

## Files

- `train_gpt.py`: Baseline trainer with orthogonal init for all linear weight matrices
- `ortho-init.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
