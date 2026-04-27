# Ortho-Init -- Orthogonal Initialization for All Linear Layers

## Method Overview

This experiment applies `nn.init.orthogonal_` to **every linear weight matrix** in the model, replacing PyTorch's default Kaiming uniform initialization with orthogonal initialization.

All linear layers -- including attention and MLP projections -- switch from default PyTorch init to orthogonal init. This matches the earlier standalone parameter-golf OrthoInit formulation more closely than later stacked variants that also rescaled projection weights.

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
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2994 | +0.0056 |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1940 | +0.0095 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.64B | +0.1% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,389 MiB | 0 |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2916 | +0.0069 |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1808 | +0.0116 |
| Fixed Tokens (10B) | Wall-clock | 771s | 773s | +0.3% |
| -- | Total Params | 17.04M | 17.04M | 0 |

## Analysis

Orthogonal initialization degrades validation BPB by +0.0056 (fixed compute) and +0.0069 (fixed tokens), a consistent regression across both evaluation regimes.

**What orthogonal init targets and why it is not redundant with normalization layers.**

Orthogonal initialization operates on weight matrices (parameter space), setting all 2D linear weights to have orthogonal rows/columns at the start of training. This is fundamentally different from the normalization mechanisms shared by both baseline and ortho-init:

- **RMSNorm and QK-Norm** normalize activations during the forward pass (activation space). They constrain per-sample statistics but do not control the conditioning of the weight matrices themselves.
- **Muon** normalizes gradient updates via Newton-Schulz orthogonalization and spectral scaling (gradient space / backward pass). It influences how weights evolve during training, not their initial values.

None of these mechanisms is a substitute for weight matrix initialization. Each operates in a different space (activation / gradient / parameter) and at a different stage (forward pass / backward pass / initialization). The regression does not indicate that normalization "subsumes" orthogonal init -- it indicates that at this depth and width, Kaiming uniform already provides adequate initial conditioning.

**Why Kaiming uniform works well at this scale.**

At 9 layers with model dimension 512, the network is relatively shallow. Kaiming uniform initialization is designed to preserve activation variance across layers, which is sufficient for stable training at this depth. Orthogonal initialization adds the stronger constraint that weight matrices are exactly orthogonal, but this additional constraint does not translate into a material training advantage here. At deeper scales (e.g., 50+ layers) where signal propagation through many linear transformations becomes more sensitive to conditioning, orthogonal init might provide more benefit.

**Interaction with CastedLinear.**

All linear layers in this codebase use CastedLinear, which stores master weights in fp32 for optimizer state quality but casts them to bf16 at matmul time. The `nn.init.orthogonal_()` call operates on the fp32 master weights, so the initialization itself is applied in fp32. However, during the forward pass, these fp32 weights are cast to bf16 (7-bit mantissa), which introduces small floating-point rounding errors that break exact orthogonality of the compute-time weight matrix. This means the effective conditioning benefit of orthogonal init is partially diluted by bf16 quantization at every forward pass. Kaiming uniform weights do not rely on exact orthogonal structure, so this quantization does not degrade their intended initialization properties.

**Conclusion.**

The regression is a genuine finding: orthogonal init does not improve over default Kaiming uniform at this scale (9L, dim=512). However, the cause is not "redundancy with normalization layers" -- which serve completely different roles in the training stack -- but rather that default init is already sufficient for stable training at shallow depth, and the CastedLinear bf16 cast partially erodes the exact orthogonality that orthogonal init provides.

## Files

- `train_gpt.py`: Baseline trainer with orthogonal init for all linear weight matrices
- `ortho-init.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
