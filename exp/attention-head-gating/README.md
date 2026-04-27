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

| Component | Baseline | attention-head-gating |
|---|---|---|
| Per-head gate | none | **context-dependent sigmoid gate per head** |
| Attention algorithm | standard | standard |
| GQA support | yes | yes |
| Everything else | identical | identical |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | Baseline | attention-head-gating | Delta |
|---|---|---|---|
| **Val BPB** | 1.2938 | 1.2891 | **-0.0047** |
| Val Loss | 2.1845 | 2.1766 | -0.0079 |
| Train Tokens | 7.63B | 7.39B | -3.1% |
| Peak Memory | 8,389 MiB | 8,400 MiB | +11 MiB |

### Fixed Tokens (10B tokens)

| Metric | Baseline | attention-head-gating | Delta |
|---|---|---|---|
| **Val BPB** | 1.2847 | 1.2770 | **-0.0077** |
| Val Loss | 2.1692 | 2.1562 | -0.0130 |
| Wall-clock | 771s | 796s | +3.2% |
| Peak Memory | 8,389 MiB | 8,400 MiB | +11 MiB |

**Total params**: 17,039,360 (17.04M), +864 vs baseline (12 x 8 x 9, negligible)

## Analysis

Attention-head gating provides a **consistent improvement** of -0.0047 BPB under fixed-compute and -0.0077 BPB under fixed-tokens. The per-head, per-position gate allows the model to learn which attention heads are useful at each position. This is a lightweight mechanism (only 864 extra parameters) that acts as a learned attention head router.

The gate operates before head merging: each head's output is scaled independently, allowing the model to suppress unhelpful heads at specific positions while amplifying useful ones. The fixed-compute gain is modest (-0.0047 BPB) because the throughput cost is slight (only 3.1% fewer tokens), while the per-token quality is meaningfully better. Under fixed-tokens, the improvement is larger at -0.0077 BPB, suggesting the gate provides genuine per-position head selection that the baseline cannot replicate even with more data.

The consistent improvement across both fixed-compute and fixed-tokens regimes suggests the gating mechanism captures genuine signal rather than overfitting. The improvement comes at near-zero cost: +11 MiB memory, +3.2% wall-clock, +864 parameters. This is a **high value-for-cost trick**: meaningful improvement with negligible overhead.

## Files

- `train_gpt.py`: Baseline trainer with slowrun-style context-dependent per-head gating
- `attention-head-gating.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
