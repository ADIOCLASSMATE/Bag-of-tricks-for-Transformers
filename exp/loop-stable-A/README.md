# Loop-Stable-A -- Parcae Stable State Transition Ablation

## Motivation

Looped transformers reuse the same set of blocks across multiple iterations, trading depth for parameter efficiency. However, in the pure looped setting (no U-Net skip connections), the residual stream can grow unbounded across iterations. If each iteration amplifies the hidden state, the model enters a positive feedback loop: larger states produce larger outputs, which feed back as even larger inputs to the next iteration. This manifests as training loss spikes, gradient explosion, or outright divergence.

Parcae (arXiv 2604.12946) provides a theoretical framework for this instability. By recasting the looped forward pass as a discrete dynamical system:

```
h_{t+1} = A_bar * h_t + B_bar * e + R_bar(h_t, e)
```

they show that instability arises when the spectral radius rho(A_bar) >= 1. When rho(A_bar) < 1, the linear part of the recurrence is contractive, preventing unbounded state growth even if the nonlinear residual R_bar temporarily amplifies the state.

Parcae enforces rho(A_bar) < 1 via a structured parameterization: A_bar is the zero-order hold (ZOH) discretization of a negative diagonal continuous-time matrix A = Diag(-exp(log_A)). For diagonal A with dt=1, this simplifies to A_bar = Diag(exp(-exp(log_A))), where each diagonal entry is guaranteed to lie in (0, 1) regardless of the learned log_A values.

## What This Ablation Tests

**Single change from loop-pure**: Add a learnable diagonal state transition matrix A_bar with guaranteed spectral norm rho(A_bar) < 1, applied between loop iterations. An input injection matrix B_bar = I - A_bar re-injects the original embedding to prevent information loss from the contractive decay.

| Component | loop-pure | loop-stable-A |
|---|---|---|
| Forward (per iteration) | `h = block(h, x0)` for each block | Same block pass, then `h = A_bar * h + B_bar * x0` |
| Inter-iteration transition | None (output feeds directly to next iteration) | `A_bar * h + B_bar * x0` applied after all blocks |
| Additional parameters | 0 | +512 (model_dim scalar log_A values) |
| Spectral radius guarantee | None | rho(A_bar) < 1 by construction |

### Parameterization Details

- `log_A`: learnable parameter of shape (model_dim,), initialized to `loop_log_a_init` (default 0.0)
- `A_bar = Diag(exp(-exp(log_A)))`: each diagonal entry in (0, 1), guaranteed stable
- `B_bar = I - A_bar = Diag(1 - exp(-exp(log_A)))`: complementary input injection

With the default `log_a_init=0.0`:
- A_bar ~ Diag(exp(-1)) ~ Diag(0.368): moderate state retention between iterations
- B_bar ~ Diag(1 - 0.368) ~ Diag(0.632): strong re-injection of the original embedding

This means at initialization, each loop iteration retains about 37% of the previous state and re-injects 63% of the original embedding. The model can learn to adjust these ratios: as log_A grows large, A_bar approaches 0 (forget everything, rely on fresh embedding); as log_A grows negative, A_bar approaches 1 (retain nearly all of previous state).

### Relationship to Parcae

This is a simplified adaptation of Parcae's core stability mechanism. In the full Parcae model, the recurrence applies at a finer granularity (potentially per-block), and the model includes additional components like learnable input/output projections. Here we apply the stable transition only between full loop iterations (after all blocks in one iteration have processed the state), which is the coarsest possible application point and the cleanest ablation.

## Expected Impact

1. **Training stability**: The contractive A_bar should prevent residual state explosion, eliminating the loss spikes that can occur in loop-pure training. Parcae demonstrated this at 100M--1.3B parameter scales; this ablation tests whether the same benefit appears at the smaller 5M scale used in this suite.

2. **Gradient flow**: By bounding the state magnitude between iterations, gradients should propagate more stably through the looped computation graph. This may be particularly beneficial with the Muon optimizer, which relies on the spectral properties of gradient matrices.

3. **Information retention trade-off**: The A_bar decay factor means each iteration "forgets" a portion of its state. While this prevents explosion, it also limits how much information from early iterations can influence later ones. The model must learn to encode important persistent information in a way that survives the contractive decay, or rely on B_bar to re-inject it from the original embedding.

4. **Comparison with loop-pure**: If loop-pure trains without divergence at this scale, the stable-A mechanism may provide no benefit (or slight harm from the reduced expressivity). The ablation is most informative when loop-pure shows instability; in that case, loop-stable-A should train more smoothly while achieving comparable or better final BPB.

## Key Differences from loop-pure

| Parameter | loop-pure | loop-stable-A |
|---|---|---|
| Inter-iteration transition | None | `A_bar * h + B_bar * x0` |
| `log_A` parameter | -- | nn.Parameter of shape (model_dim,) |
| `loop_stable_a` flag | -- | True (enable) / False (disable, reverts to loop-pure) |
| `loop_log_a_init` | -- | 0.0 (A_bar ~ 0.37, B_bar ~ 0.63) |
| Additional optimizer state | 0 | +512 scalars in Adam (scalar_params) |
| CONTROL_TENSOR_NAME_PATTERNS | (excludes log_A) | includes "log_A" (kept in fp32) |

Note: The `loop_stable_a` flag allows disabling the stable transition, reverting to loop-pure behavior. Setting `LOOP_STABLE_A=0` makes this experiment identical to loop-pure, which is useful for debugging and verification.

## Results

| Regime | Metric | loop-pure | loop-stable-A | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | -- | -- | -- |
| Fixed Compute (10 min) | Train Tokens | -- | -- | -- |
| Fixed Compute (10 min) | Peak Memory | -- | -- | -- |
| Fixed Tokens (10B) | Val BPB | -- | -- | -- |
| Fixed Tokens (10B) | Wall-clock | -- | -- | -- |
| -- | Total Params | -- | -- | -- |

*(Results to be filled after running experiments.)*

## Files

- `train_gpt.py`: Training script (loop-pure base + stable-A parameterization)
- `loop-stable-A.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
