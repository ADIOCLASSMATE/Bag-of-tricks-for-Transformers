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
| Fixed Compute (10 min) | Val BPB | 1.2938 | 1.2954 | **+0.0016** |
| Fixed Compute (10 min) | Val Loss | 2.1845 | 2.1873 | +0.0028 |
| Fixed Compute (10 min) | Train Tokens | 7.63B | 7.45B | -2.4% |
| Fixed Compute (10 min) | Peak Memory | 8,389 MiB | 8,654 MiB | +265 MiB |
| Fixed Tokens (10B) | Val BPB | 1.2847 | 1.2763 | **-0.0084** |
| Fixed Tokens (10B) | Val Loss | 2.1692 | 2.1550 | -0.0142 |
| Fixed Tokens (10B) | Wall-clock | 771s | 789s | +2.3% |
| Fixed Tokens (10B) | Peak Memory | 8,389 MiB | 8,654 MiB | +265 MiB |
| -- | Total Params | 17.04M | 18,612,224 (18.61M) | +1,572,864 (+9.2%) |

## Analysis

KV sharing + double-wide MLP shows a **mixed result**: slightly worse under fixed-compute (+0.0016 BPB) but substantially better under fixed-tokens (-0.0084 BPB).

**FC regression is real but small**: Under fixed-compute, the model processes 2.4% fewer tokens due to throughput reduction from the wider MLP layers, yielding a small +0.0016 BPB regression. The wider MLP provides meaningful per-token quality gains, but these do not fully compensate for the throughput reduction in the fixed-time regime.

**Strong FT improvement**: Under fixed-tokens, the trick achieves a substantial -0.0084 BPB improvement. This demonstrates that the parameter reallocation from attention (shared KV) to MLP capacity is genuinely beneficial for sample efficiency -- the model simply needs enough data to capitalize on the extra MLP expressivity. The wider MLP in later layers provides meaningful representational benefits when data is abundant.

**Cross-layer KV sharing is viable**: The shared K/V mechanism maps layers 7->0 and 8->1, eliminating redundant key/value projections while the double-wide MLP compensates with additional feed-forward capacity. The strong FT result indicates that early and late layers develop sufficiently compatible key/value distributions for shared projections to work well. The shared KV projections receive gradient updates from both source and consumer layers, which may provide beneficial implicit regularization.

**Parameter efficiency context**: The +1.57M param overhead is significant (+9.2% more params). The improvement per added parameter is reasonable under FT (-0.0084 BPB for 9.2% more params) but the FC regression suggests the parameter budget is better spent when given sufficient training tokens. Memory increases by 265 MiB and wall-clock by 2.3% in FT.

**Conclusion**: KV sharing + double-wide MLP is a legitimate sample-efficiency trick -- it improves quality per training token (-0.0084 FT BPB) but at a slight cost in training throughput (+0.0016 FC BPB). The trick is appropriate when ample data and parameter budget are available. For compute-constrained settings, simpler tricks with less overhead may be preferred.

## Files

- `train_gpt.py`: Training script (baseline + KV sharing + double-wide MLP modification)
- `kv-sharing-dwide-mlp.json`: Experiment manifest
- `logs/`: Training output (automatically generated)
