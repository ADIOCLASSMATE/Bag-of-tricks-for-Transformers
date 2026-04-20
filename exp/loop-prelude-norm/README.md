# loop-prelude-norm

Prelude normalization for looped transformers, inspired by Parcae (arXiv 2604.12946).

## Single Change from loop-pure

This experiment adds **input re-injection with normalization** after each loop iteration. Specifically, after each full pass through the shared loop blocks, the normalized original embedding is added back with a learnable scale:

```
h = h + loop_inject_scale * RMSNorm(x0)
```

This is the only change from the `loop-pure` baseline. All other architecture details (3 shared blocks, 3 repeats, no U-Net, resid_mix within blocks) are identical.

## Mechanism

Two new parameters are added to the GPT model:

- **`prelude_norm`** (`RMSNorm`): Normalizes the original input embedding `x0` before re-injection. This ensures the re-injected signal has consistent magnitude across loop iterations, regardless of how the embedding space evolves during training.

- **`loop_inject_scale`** (1D parameter, shape `[model_dim]`, initialized to **zeros**): A per-dimension learnable scale that controls how much of the normalized input to re-inject. Zero initialization means the model starts with no injection (identical to loop-pure at init), and must learn the appropriate injection strength.

During forward, after each full loop iteration (i.e., after all `num_loop_layers` blocks have processed the state):

```python
for rep in range(num_loop_repeats):
    for block in self.loop_blocks:
        h = block(h, x0)
    # Re-inject normalized input
    if self.prelude_norm is not None:
        h = h + self.loop_inject_scale * self.prelude_norm(x0)
```

## Why This Matters

In standard looped transformers, the original input signal can get attenuated across many iterations. Each block's `resid_mix` blends the current state with `x0`, but the blending ratio is learned and may shift during training. Over 9+ effective layers of processing (3 blocks x 3 repeats), the model can progressively "forget" the original input, leading to:

1. **Signal decay**: The contribution of the original embedding diminishes with depth.
2. **Late-stage loss spikes**: Without a stable reference signal, later iterations can produce unstable gradients (observed in Parcae).
3. **Representational drift**: Each iteration may subtly shift the representation space, compounding across repeats.

The prelude norm injection addresses these by providing a **fresh, normalized reference signal** at each iteration boundary. The model always has access to a consistently-scaled version of the input, regardless of how many iterations have elapsed.

## Context from Parcae

Parcae (arXiv 2604.12946) applies LayerNorm to the prelude output before injection:

```
e = LN(P(s))
```

where `P(s)` is the prelude (initial embedding processing block), and `e` is the normalized signal injected via `B_bar * e` at each recurrence step. Parcae reports that this "empirically prevents late-stage loss spikes" during training.

Our implementation adapts this idea to the simpler looped architecture:
- We use RMSNorm instead of LayerNorm (consistent with the rest of the architecture).
- The injection uses a per-dimension learnable scale (initialized to zero) rather than a fixed linear projection `B_bar`.
- Injection occurs at the iteration boundary, not within each block.

## Difference from resid_mix

The existing `resid_mix` mechanism operates **within** each block:

```python
x = mix[0] * x + mix[1] * x0
```

This blends the current residual state with the raw `x0` before attention and MLP. It is a fine-grained, per-block blend.

The prelude norm injection operates at the **iteration boundary**, providing a coarse-grained signal boost after each full pass through all looped blocks. These two mechanisms are complementary:

| Mechanism | Location | Granularity | Signal |
|-----------|----------|-------------|--------|
| `resid_mix` | Within each block | Per-block | Raw `x0` |
| `prelude_norm` injection | After each iteration | Per-iteration | Normalized `x0` (scaled) |

## Hypothesis

Input re-injection with normalization should stabilize deep looping by:

1. Preventing the model from "forgetting" the original input across many iterations.
2. Reducing loss spikes during late-stage training (as observed in Parcae).
3. Improving final BPB by maintaining a stable gradient pathway back to the embedding.

The zero-initialized `loop_inject_scale` ensures the model starts from the same behavior as loop-pure and must actively learn to use the injection, making this a clean ablation.

## Experiments

Four experiment configurations are defined:

| Experiment | Prelude Norm | Control Mode | Target |
|------------|-------------|--------------|--------|
| `loop-prelude-norm-on-fixed_time_10min` | ON | Fixed compute | 600s wallclock |
| `loop-prelude-norm-on-fixed_tokens_10b` | ON | Fixed tokens | 10B tokens |
| `loop-prelude-norm-off-fixed_time_10min` | OFF | Fixed compute | 600s wallclock |
| `loop-prelude-norm-off-fixed_tokens_10b` | OFF | Fixed tokens | 10B tokens |

The OFF variants (with `LOOP_PRELUDE_NORM=0`) serve as an internal control, equivalent to loop-pure but run from this trainer script. Comparing ON vs OFF isolates the effect of the prelude norm injection.

## Hyperparameters

All default hyperparameters match the baseline and loop-pure experiments:

- `num_loop_layers`: 3
- `num_loop_repeats`: 3
- `loop_prelude_norm`: 1 (enabled by default)
- Effective depth: 3 x 3 = 9 (same as baseline's 9 layers)
