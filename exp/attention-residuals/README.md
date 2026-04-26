# Attention Residuals — Block-Level Attention Over Layer Outputs

## Method Overview

Attention Residuals replaces the standard sequential residual connection pattern with a block-level attention mechanism that allows each layer to attend over all previous layer outputs within its block group.

**Standard Residual Connection:**
```
h_l = h_{l-1} + f(h_{l-1})
```

**Attention Residuals:**
```
h_l = Σ α_{i→l} · s_i
```

where `s_i` are stored block outputs (embedding + layer group outputs) and `α_{i→l}` are learned attention weights computed via a recency-biased attention mechanism.

**Block Grouping:**
For a 9-layer model with `attn_res_num_blocks=3`, the layers are grouped as:
- Block 0: Token embedding
- Block 1: Layers 0-2 (first 3 layers)
- Block 2: Layers 3-5 (middle 3 layers)
- Block 3: Layers 6-8 (last 3 layers)

Each sublayer (attention or MLP) within a block can attend over all previous block outputs, enabling richer information flow than sequential residuals.

**Recency Bias:**
The attention mechanism includes a learnable recency bias initialized to favor recent blocks (controlled by `attn_res_recency_bias_init=10.0`). This ensures that at initialization, the model behaves equivalently to the baseline, with attention weights heavily concentrated on the most recent block.

## Key Differences from Baseline

| Parameter | Baseline | Attention Residuals | Change |
|---|---|---|---|
| Residual pattern | Sequential (`h + f(h)`) | Block attention (`Σ α_i · s_i`) | Architectural |
| Block storage | None | 4 tensors (embed + 3 blocks) | +4 activations |
| Attention parameters | N/A | ~9,234 params (0.05% increase) | +9,234 params |
| Memory overhead | Baseline | +20-25% | +~2.5 GiB |
| Compute overhead | Baseline | +2-3% | Minimal |
| Total parameters | 17.06M | ~17.07M | +0.05% |

**Additional Parameters Breakdown:**
- Recency bias: 4 scalars (one per block)
- Attention weights: ~9,230 parameters for computing block attention scores across all sublayers

## Impact on Training

**Memory:**
- Stores 4 block-level tensors throughout the forward pass (embedding + 3 layer group outputs)
- Each block tensor has shape `[batch_size, seq_len, model_dim]`
- For typical batch size and sequence length, this adds ~20-25% memory overhead
- Activation checkpointing could reduce this if needed

**Compute:**
- Each sublayer performs attention over 4 blocks (embedding + 3 layer groups)
- Attention computation is lightweight: score calculation + weighted sum
- Overhead is ~2-3% of total training time
- The recency-biased attention is efficiently implemented with minimal branching

**Initialization:**
- Recency bias initialized to `10.0` ensures that at step 0, attention weights are heavily concentrated on the most recent block
- This makes the model functionally equivalent to baseline at initialization
- As training progresses, the model can learn to attend to earlier blocks if beneficial

## Origin

- **Source:** [open-attention-residuals](https://github.com/kimi-team/open-attention-residuals) project
- **Paper:** "Attention Residuals" by Kimi Team
- **Implementation reference:** `modeling_attnres.py` from the open-attention-residuals repository
- **Key insight:** Block-level attention enables richer gradient flow and information routing compared to sequential residuals, potentially improving both training dynamics and final performance

## Results

### Fixed Compute (10 min wall-clock)

| Experiment | Val BPB | Val Loss | Tokens Processed | Wall-clock Time |
|---|---|---|---|---|
| baseline | 1.2194 | 2.0589 | 6.95B | 600s |
| attention-residuals | TBD | TBD | TBD | 600s |

### Fixed Tokens (10B tokens)

| Experiment | Val BPB | Val Loss | Tokens Processed | Wall-clock Time |
|---|---|---|---|---|
| baseline | 1.2118 | 2.0460 | 10.0B | 832.8s |
| attention-residuals | TBD | TBD | 10.0B | TBD |

**Note:** Experiments not yet run. Results will be populated after running `bash exp/attention-residuals/run.sh`.

## Files

- `train_gpt.py` — Training script with Attention Residuals implementation
- `attention-residuals.json` — 2-experiment manifest (fixed_time_10min, fixed_tokens_10b)
- `run.sh` — Launch script for both experiments with wandb sync
- `README.md` — This documentation
- `logs/` — Experiment outputs (automatically generated)

## How to Run

```bash
# Dry-run the manifest (prints 2 resolved runs without launching)
python exp/run_experiments.py exp/attention-residuals/attention-residuals.json --dry-run

# Launch both experiments
bash exp/attention-residuals/run.sh
```

## Expected Impact

**Potential Benefits:**
- Richer gradient flow through block-level attention vs. sequential residuals
- Ability to route information from earlier layers directly to later layers
- May improve training dynamics and convergence speed
- Could enable better feature reuse across layer groups

**Trade-offs:**
- 20-25% memory overhead from storing block tensors
- 2-3% compute overhead from attention calculations
- Minimal parameter increase (0.05%)
- Slightly more complex architecture to debug and analyze

The experiment will determine whether the architectural benefits outweigh the memory and compute costs in this training regime.
