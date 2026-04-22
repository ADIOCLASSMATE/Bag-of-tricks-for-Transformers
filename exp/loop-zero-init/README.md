# Loop-Zero-Init -- Zero-Initialization of Output Projections in Looped Transformers

## Motivation

In a looped (weight-shared) transformer, the same block is applied repeatedly across the depth of the network. This creates a fundamental initialization challenge: if each application of the block significantly alters the hidden state, the repeated composition across many loop iterations can cause activations to explode or vanish at initialization. This is distinct from a standard transformer, where each layer has independent weights that can be initialized independently.

**Loop, Think, & Generalize** (arXiv 2604.07822) addresses this by zero-initializing the output projection (`c_proj`) of attention and FFN blocks in the looped portion of the model. With zero-init, each block becomes an identity mapping at initialization (output = zero, added via residual connection to input), so the entire loop is an identity regardless of unrolling depth. This ensures a stable Jacobian across arbitrary unrolling depth, preventing the gradient explosion or vanishing that can occur when random initial projections are composed repeatedly.

## What This Ablation Tests

This experiment is identical to **loop-pure** (3 shared blocks x 3 repeats, no U-Net, no skip connections) but adds an explicit `loop_zero_init` flag that controls whether the output projections (`proj`) in the looped blocks are zero-initialized.

The baseline codebase already sets `self.proj._zero_init = True` on `CausalSelfAttention.proj` and `MLP.proj`, and `_init_weights()` applies `nn.init.zeros_()` to any `nn.Linear` with `_zero_init = True`. So **loop-pure already has zero-init enabled by default**. This experiment makes that behavior explicit and controllable, allowing us to test the effect of removing it.

### Single change from loop-pure

Add a `zero_init` parameter to `CausalSelfAttention`, `MLP`, and `Block`, threaded through to the `_zero_init` attribute on the output projection. When `loop_zero_init=1` (default), the behavior matches loop-pure exactly. When `loop_zero_init=0`, the output projections use standard PyTorch initialization instead of zero-init.

| Component | loop-pure (implicit) | loop-zero-init ON | loop-zero-init OFF |
|---|---|---|---|
| `CausalSelfAttention.proj` | `_zero_init = True` | `_zero_init = True` | `_zero_init = False` |
| `MLP.proj` | `_zero_init = True` | `_zero_init = True` | `_zero_init = False` |
| Block at init | Identity (zero output) | Identity (zero output) | Non-trivial random output |
| Full loop at init | Identity | Identity | Random composition |

## Experiments

Four configurations test the effect under both training regimes:

| Name | `LOOP_ZERO_INIT` | Regime |
|---|---|---|
| `loop-zero-init-on-fixed_time_10min` | 1 (default) | Fixed compute (10 min) |
| `loop-zero-init-on-fixed_tokens_10b` | 1 (default) | Fixed tokens (10B) |
| `loop-zero-init-off-fixed_time_10min` | 0 | Fixed compute (10 min) |
| `loop-zero-init-off-fixed_tokens_10b` | 0 | Fixed tokens (10B) |

The "ON" configurations serve as the control (should match loop-pure results). The "OFF" configurations test the ablation.

## Hypothesis

Zero-init should improve training stability in looped models. Without it:

1. **Gradient instability**: Each loop iteration applies a random projection at init, so the loop's Jacobian is a product of random matrices. With 3 blocks x 3 repeats = 9 compositions, the gradient can explode or vanish depending on the spectral radius of the composed transformations.

2. **Loss spikes**: The initial forward pass produces arbitrarily large activations from the composed random projections, leading to high initial loss and potential training instability in early steps.

3. **Slower convergence**: Even if training does not diverge, the model must first "undo" the random initialization before it can learn useful representations, wasting optimization steps.

With zero-init, the loop starts as an identity and gradually learns to transform the hidden state. This is analogous to how residual networks benefit from identity initialization of the residual branch.

## Key Differences from loop-pure

| Parameter | loop-pure | loop-zero-init |
|---|---|---|
| `CausalSelfAttention.__init__` | `zero_init` hardcoded `True` | `zero_init` parameter (default `True`) |
| `MLP.__init__` | `zero_init` hardcoded `True` | `zero_init` parameter (default `True`) |
| `Block.__init__` | No `zero_init` param | `zero_init` parameter (default `True`) |
| `GPT.__init__` | No `loop_zero_init` param | `loop_zero_init` parameter |
| `Hyperparameters` | No `loop_zero_init` | `loop_zero_init = bool(int(os.environ.get("LOOP_ZERO_INIT", "1")))` |

When `LOOP_ZERO_INIT=1` (default), behavior is identical to loop-pure.

## Reference

- Loop, Think, & Generalize (arXiv 2604.07822): Zero-init c_proj for stable Jacobian across arbitrary unrolling depth.

## Files

- `train_gpt.py`: Training script (loop-pure + controllable zero-init)
- `loop-zero-init.json`: Experiment manifest (4 experiments)
- `logs/`: Training output (automatically generated)
