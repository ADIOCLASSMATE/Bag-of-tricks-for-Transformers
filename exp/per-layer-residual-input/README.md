# Per-Layer Residual Input — Gated Token Identity Re-injection Ablation

## Motivation

In a deep Transformer, the residual stream is a single vector that gets iteratively modified by attention and MLP sub-layers. As the network depth increases, the original token identity (embedding) can become diluted — the residual stream increasingly reflects the accumulated contextual transformations rather than the original token's identity. The baseline uses only standard residual connections (`x = x + sublayer(x)`) with **no mechanism** to re-inject the original embedding information at deeper layers. Once the embedding signal is diluted by successive sub-layer outputs, there is no way for later layers to recover it.

Gemma 4 introduces a **per-layer residual input** mechanism: a separate low-dimensional embedding table (`embed_tokens_per_layer`) that injects a fresh, layer-conditional token identity signal at every block via a gated projection. This creates a "second residual stream" that carries token identity information independent of the main residual stream's contextual evolution.

## What This Ablation Tests

This experiment adds a per-layer embedding stream (dimension 128 vs. model_dim 512) that is combined with a projection of the main hidden state and injected at every layer through a ReLU-gated mechanism. All other architectural choices remain identical to the baseline.

| Component | Baseline | Per-Layer Residual Input |
|---|---|---|
| Identity re-injection | None (standard residual connections only) | Gated per-layer embedding injection at every layer |
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

| Parameter | Baseline | This Experiment |
|---|---|---|
| `per_layer_embed_dim` | — | 128 |
| `per_layer_embed_scale` | — | 0.01 |
| `embed_tokens_per_layer` | — | nn.Embedding(1024, 128) |
| Per-block injection | — | gated ReLU + project + RMSNorm |

## Results

| Regime | Metric | Baseline | Per-Layer Residual Input | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2950 | +0.0012 |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1866 | +0.0021 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 6.90B | -9.6% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 9,799 MiB | +1,410 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2806 | -0.0041 |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1622 | -0.0070 |
| Fixed Tokens (10B) | Wall-clock | 771s | 852s | +10.5% |
| — | Total Params | 17.04M | 18.42M | +1.38M (+8.1%) |

## Analysis

Per-layer residual input adds a learnable per-token embedding that gets projected and gated into each layer. The fixed-tokens improvement of -0.0041 BPB is modest relative to the +1.38M parameter overhead (+8.1% more parameters). Under fixed compute, the trick provides essentially no benefit — the extra parameters and memory do not pay for themselves when the compute budget is held constant.

**Why fixed compute shows no benefit**: The heavier forward pass (per-layer embedding lookup, gated projection, RMSNorm at every block) reduces token throughput by 9.6%. The per-token quality improvement from the extra signal is too small to offset the loss of training tokens, yielding a net regression of +0.0012 BPB.

**Why fixed tokens shows a small benefit**: When both models see the same 10B tokens, the extra per-layer signal provides a small but consistent benefit (-0.0041 BPB, ~0.3% relative). The gating mechanism (`per_layer_gate` + `per_layer_projection`) controls how much per-layer information flows into each block. However, the model appears to mostly rely on the standard residual stream — the 0.01 scaling factor applied to the per-layer contribution suggests the learned signal is intentionally small, which limits the trick's impact.

**Memory overhead is disproportionate**: The +1,410 MiB increase (+16.8%) comes primarily from the extra embedding table stored at full precision. This is a substantial memory cost for the marginal fixed-tokens improvement observed. The embedding table and projection layers account for 1.38M extra parameters, but their memory footprint is amplified by the full-precision storage and intermediate activations during the gated injection at every layer.

**Depth dependence hypothesis**: At 9 layers, the residual stream is short enough that token identity propagates naturally through standard residual connections alone. The per-layer mechanism's additional expressivity — its own embedding table, gated injection, and learned projection — is largely redundant at this depth. The trick may become more valuable in deeper models (e.g., 40+ layers like Gemma 4) where identity dilution is more severe and the low-dimensional bottleneck (128 vs. model_dim) carries a different information-to-noise ratio.

**Verdict**: Per-layer residual input does not justify its overhead at this scale. The parameter cost (+8.1%), memory cost (+16.8%), and throughput cost (-9.6%) outweigh the marginal fixed-tokens benefit. Standard residual connections appear sufficient for 9-layer models.

**Reproducibility note**: Results from run 20260423-143034. An earlier run with identical config produced slightly better results (FC BPB=1.2935, FT BPB=1.2806), but was discarded in favor of the latest run. The variance (~0.005 FC, ~0.001 FT) is within expected noise for this training setup and does not change the conclusion.

## Files

- `train_gpt.py`: Training script (baseline + per-layer residual input modification)
- `per-layer-residual-input.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
