# KV Sharing + Double-Wide MLP — Cross-Layer Parameter Efficiency Ablation

## Motivation

In standard Transformers, every layer has its own independent key (K) and value (V) projection matrices. However, the KV projections across layers may learn redundant representations, especially in adjacent layers that process similar features. If some layers can share KV projections without significant quality loss, the saved parameter budget can be reallocated to increase model capacity elsewhere.

Gemma 4 introduces **cross-layer KV sharing**: the last N layers reuse KV projections from earlier layers, eliminating their own `c_k`, `c_v`, `k_norm`, and `v_norm` parameters. To compensate for the reduced per-layer capacity, the saved parameter budget is supplemented with additional parameters to double the MLP width of the shared layers. Note that this is **not parameter-neutral** in our setting: the KV savings (2 layers × 2 projections × dim × kv_dim) are much smaller than the MLP width doubling (2 layers × 2 projections × dim × 2×dim), resulting in a net +9.2% parameter increase (17.04M → 18.63M). The rebalancing trades a small amount of attention capacity for substantially more feed-forward capacity.

## What This Ablation Tests

This experiment makes the last 2 layers (7, 8) of the 9-layer model share KV projections from the first 2 layers (0, 1), while doubling their MLP intermediate width from 2× to 4× model_dim. All other architectural choices remain identical to the baseline.

| Component | Baseline | KV Sharing + Double-Wide MLP |
|---|---|---|
| Layers 0–6 | Independent KV, MLP 2× | Independent KV, MLP 2× (unchanged) |
| Layers 7–8 | Independent KV, MLP 2× | **Shared KV** (from layers 0–1), **MLP 4×** |

KV source mapping: layer 7 → layer 0's KV, layer 8 → layer 1's KV.

## Expected Impact

1. **Parameter redistribution efficiency**: The ablation tests whether MLP capacity is more valuable than KV projection diversity in the later layers. Since later layers tend to process more abstract, high-level features, they may benefit more from wider feed-forward networks than from independent key/value representations.

2. **Representation sharing quality**: If layers 0 and 7 (or 1 and 8) produce similar key/value distributions, sharing should have minimal quality impact. However, if early and late layers learn fundamentally different attention patterns, forcing them to share KV projections could create a bottleneck that limits both layers' effectiveness.

3. **Training dynamics**: Shared KV means the shared projection weights receive gradient updates from both the source and consumer layers. This may act as a form of implicit regularization (similar to weight sharing in other contexts), potentially improving generalization but possibly slowing convergence.

4. **Cross-layer alignment**: The baseline uses standard sequential blocks with no skip connections between non-adjacent layers. With KV sharing mapping layer 7→0 and 8→1, the shared KV projections create an implicit cross-layer link between early and late layers. This may help or hinder depending on whether early and late layers produce compatible key/value distributions.

## Key Differences from Baseline

| Parameter | Baseline | This Experiment |
|---|---|---|
| `num_kv_shared_layers` | — | 2 |
| `use_double_wide_mlp` | — | True |
| Layers 7–8 MLP width | 1024 (2×512) | **2048 (4×512)** |
| Layers 7–8 c_k/c_v | Independent | **Shared from layers 0/1** |

## Results

| Regime | Metric | Baseline | KV Sharing + Double-Wide MLP | Delta |
|---|---|---|---|---|
| Fixed Compute (10 min) | Val BPB | 1.2979 | 1.2905 | **-0.0074** |
| Fixed Compute (10 min) | Val Loss | 2.1914 | 2.1789 | -0.0125 |
| Fixed Compute (10 min) | Train Tokens | 7.67B | 7.47B | -2.6% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,654 MiB | +265 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2857 | 1.2823 | **-0.0034** |
| Fixed Tokens (10B) | Val Loss | 2.1709 | 2.1651 | -0.0058 |
| Fixed Tokens (10B) | Wall-clock | 772s | 794s | +2.8% |
| Fixed Tokens (10B) | Peak Memory | 8,389 MiB | 8,654 MiB | +265 MiB |
| -- | Total Params | 17.04M | 18,612,224 (18.61M) | +1,572,864 (+9.2%) |

## Analysis

KV sharing + double-wide MLP provides a **consistent improvement** across both evaluation regimes: -0.0074 BPB under fixed-compute and -0.0034 BPB under fixed-tokens.

**Attention-to-MLP parameter reallocation works**: The shared K/V mechanism reduces attention parameter count for the shared layers (7 and 8), while the double-wide MLP compensates by adding more feed-forward capacity. This trades attention diversity for MLP expressivity. Under fixed-compute, the model processes 2.6% fewer tokens yet still achieves a -0.0074 BPB improvement, indicating that the wider MLP provides meaningful per-token quality gains that more than offset the throughput reduction. Under fixed-tokens, the -0.0034 BPB improvement confirms the sample efficiency gain is genuine.

**Cross-layer KV sharing is viable**: The shared K/V approach is inspired by GQA (Grouped Query Attention) taken further -- instead of just sharing K/V heads within a layer, K/V are shared across specific layers entirely, eliminating redundant key/value projections. Layers 7 and 8 reuse KV projections from layers 0 and 1 respectively, and the consistent improvement indicates the shared representations are not a bottleneck: early and late layers learn sufficiently compatible key/value distributions.

**Diminishing returns under fixed-tokens**: The improvement is meaningful under fixed-compute (-0.007 BPB) despite only a 2.6% token reduction, suggesting the parameter reallocation from attention to MLP is beneficial. Under fixed-tokens, the improvement is smaller (-0.003 BPB) -- the extra MLP capacity provides diminishing returns when the model has enough compute to train all parameters adequately.

**Improvement per added parameter is modest**: The +1.57M param overhead is significant (+9.2% more params). While the trick works, the improvement per added parameter is modest compared to simpler tricks like untie-embed, which achieves comparable or better BPB gains with far fewer additional parameters. The 2.8% wall-clock overhead in fixed-tokens is modest given the parameter increase, since the wider MLP in only 2 of 9 layers has limited impact on total FLOPs and KV sharing slightly reduces attention-path computation. Memory increases by 265 MiB.

**Conclusion**: KV sharing + double-wide MLP is viable when the parameter budget allows the extra MLP width. The trick provides consistent but modest improvements (-0.007 FC BPB, -0.003 FT BPB) at the cost of +1.57M params and +2.8% wall-clock. For parameter-constrained settings, simpler tricks likely offer better efficiency.

## Files

- `train_gpt.py`: Training script (baseline + KV sharing + double-wide MLP modification)
- `kv-sharing-dwide-mlp.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
