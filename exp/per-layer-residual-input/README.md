# Per-Layer Residual Input — Gated Token Identity Re-injection Ablation

## Motivation

In a deep Transformer, the residual stream is a single vector that gets iteratively modified by attention and MLP sub-layers. As the network depth increases, the original token identity (embedding) can become diluted — the residual stream increasingly reflects the accumulated contextual transformations rather than the original token's identity. The baseline partially addresses this with `resid_mix`, which linearly mixes the current hidden state with the initial embedding x₀ at every layer. However, `resid_mix` uses the same static x₀ at every layer, limiting its capacity to provide layer-specific identity information.

Gemma 4 introduces a **per-layer residual input** mechanism: a separate low-dimensional embedding table (`embed_tokens_per_layer`) that injects a fresh, layer-conditional token identity signal at every block via a gated projection. This creates a "second residual stream" that carries token identity information independent of the main residual stream's contextual evolution.

## What This Ablation Tests

This experiment adds a per-layer embedding stream (dimension 128 vs. model_dim 512) that is combined with a projection of the main hidden state and injected at every layer through a ReLU-gated mechanism. All other architectural choices remain identical to the baseline.

| Component | Baseline | Per-Layer Residual Input |
|---|---|---|
| Identity re-injection | `resid_mix` (linear blend with x₀) | Gated per-layer embedding + `resid_mix` |
| Per-layer embedding dim | — | 128 (1/4 of model_dim) |
| Injection mechanism | — | `relu(gate(x)) * per_layer_embed + proj + norm` |

The injection occurs after the MLP block in each layer:
```
gate = relu(linear_down(x))              # project to per_layer_dim
gate = gate * per_layer_inputs            # element-wise multiply with per-layer embedding
output = norm(linear_up(gate))            # project back to model_dim + RMSNorm
x = x + output                            # residual connection
```

The `per_layer_inputs` signal is computed once from two sources:
1. `embed_tokens_per_layer(input_ids)` — token-identity embedding in low-dimensional space
2. `per_layer_model_projection(embeds)` — projection of the main embedding into low-dimensional space

These are summed and scaled, providing both a token-identity signal and a context-dependent modulation.

## Expected Impact

1. **Token identity preservation**: By re-injecting token identity at every layer, the model can maintain access to "what token am I" information even in deep layers where the main residual stream has been heavily contextualized. This should particularly benefit tasks requiring precise token discrimination (e.g., rare word prediction, copy mechanisms).

2. **Expressivity vs. information bottleneck**: The per-layer embedding is low-dimensional (128 vs. 512), forcing the model to compress the most relevant token-identity features into a narrow channel. This bottleneck may act as a regularizer, preventing the model from over-relying on identity information and encouraging contextual representations.

3. **Comparison with `resid_mix`**: The baseline's `resid_mix` provides a simple linear blend with x₀ at every layer. The per-layer mechanism is strictly more expressive: it has its own embedding table, a gated injection, and a learned projection. The ablation reveals whether this additional expressivity translates to better performance, or whether the simpler `resid_mix` already captures the essential benefit.

4. **Parameter overhead**: The per-layer mechanism adds `vocab_size × 128` (embedding) + `512 × 128` (model projection) + per-layer `2 × 512 × 128` (gate + projection) parameters. This is modest relative to the total model size but non-zero. Under fixed parameter budgets, this overhead must be justified by performance gains.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `per_layer_embed_dim` | — | 128 |
| `per_layer_embed_scale` | — | 2^(-0.5) ≈ 0.707 |
| `embed_tokens_per_layer` | — | nn.Embedding(1024, 128) |
| Per-block injection | — | gated ReLU + project + RMSNorm |

## Results

| Regime | Metric | Baseline | Per-Layer Residual Input | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2194 | 1.2231 | +0.0037 |
| Fixed Compute (10 min) | Train Tokens | 6.954B | 6.297B | -9.4% |
| Fixed Compute (10 min) | Peak Memory | 10,246 MiB | 11,775 MiB | +1,529 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2118 | 1.2132 | +0.0015 |
| Fixed Tokens (10B) | Wall-clock | 832.8s | 936.7s | +12.5% |
| — | Total Params | 17.06M | 18.44M | +8.1% |

## Analysis

Per-layer residual input produces a **small but consistent regression** in both regimes (+0.004 fixed-compute, +0.002 fixed-tokens), indicating the additional mechanism does not improve over the baseline.

**The identity re-injection hypothesis does not hold at this scale**: The core motivation — that token identity gets diluted in deep layers — appears to be insufficiently impactful for a 9-layer, 512-dim model. At this depth, the residual stream likely retains sufficient token identity information natively, and the existing `resid_mix` mechanism already provides adequate identity re-injection. The per-layer mechanism adds complexity without addressing a real bottleneck.

**Interaction with `resid_mix`**: The per-layer injection coexists with the baseline's `resid_mix`, which already blends x₀ into every layer. Adding a second, parallel identity stream may create redundancy or even interference — two mechanisms competing to inject similar information through different pathways could produce conflicting gradient signals that slow optimization.

**Parameter overhead not justified**: The 8.1% parameter increase (1.38M) does not translate to better performance. Under fixed-compute, the heavier forward pass reduces token throughput by 9.4% while also slightly degrading per-token quality. Under fixed-tokens, the 12.5% wall-clock overhead (from the additional per-layer projections and gated injection) is not compensated by improved sample efficiency.

**Comparison with Gemma 4 context**: Gemma 4 is a much deeper model (likely 40+ layers) where token identity dilution is a more pressing concern. The low-dimensional bottleneck (128 vs. 4096 model_dim) also has a different information-to-noise ratio at that scale. At 9 layers, the residual stream is short enough that identity information propagates naturally through residual connections without needing explicit re-injection.

**Conclusion**: Per-layer residual input is not beneficial at this model scale. The trick may be depth-dependent, providing value only when the residual stream is long enough for token identity to be substantially diluted. The baseline's simpler `resid_mix` appears sufficient for 9-layer models.

## Files

- `train_gpt.py`: Training script (baseline + per-layer residual input modification)
- `per-layer-residual-input.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
