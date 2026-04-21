# Attention-Head-Gating — Context-Dependent Per-Head Attention Gates

## Method Overview

This experiment adds a learned **per-head sigmoid gate** on the attention output before heads are merged back into the model dimension. Unlike a static per-head rescale, the gate is **computed from the current token activations**, so each token can suppress or amplify heads differently.

Concretely, each attention module uses a small linear map from the first 12 channels of the input activation to `num_heads` gate logits. The gate is applied after scaled dot-product attention and before the head outputs are reshaped and projected. This matches the slowrun implementation more closely than a static learned scalar per head.

### Origin

From **slowrun A04 AttentionHeadGating**. In slowrun, the gate is context-dependent: each token's activation produces its own head-wise gate values, enabling attention to become a learned no-op on selected heads when useful.

### Motivation

- Some heads may be consistently more useful than others during training
- A lightweight gate can suppress weak heads without changing the attention algorithm itself
- The trick is structurally simple and naturally compatible with GQA

## Impact on Training

- **Parameters**: Adds a tiny `12 x num_heads` gate matrix per block
- **Throughput**: Negligible overhead from one small linear layer and a broadcast multiply
- **Memory**: Negligible increase
- **Optimization**: The model can learn head-wise sparsity or reweighting patterns

## Key Differences from Baseline

| Component | baseline-sp1024 | attention-head-gating |
|---|---|---|
| Per-head gate | none | **context-dependent sigmoid gate per head** |
| Attention algorithm | standard | standard |
| GQA support | yes | yes |
| Everything else | identical | identical |

## Results

> Results will be filled in after running the experiment.

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | attention-head-gating | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | — | — |
| Val Loss | 2.0589 | — | — |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | attention-head-gating | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | — | — |
| Val Loss | 2.0460 | — | — |

## BPB Analysis

> To be completed after experiments.

- **If BPB improves**: Learned head reweighting helped the model emphasize consistently useful attention heads.
- **If BPB is unchanged**: The baseline already learns effective head usage through existing parameters, so explicit gates add little.
- **If BPB worsens**: The gates may over-constrain head contributions or introduce an unnecessary extra nonlinearity.

## Files

- `train_gpt.py`: Baseline trainer with slowrun-style context-dependent per-head gating
- `attention-head-gating.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
