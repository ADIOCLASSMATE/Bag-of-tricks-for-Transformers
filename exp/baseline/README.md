# Baseline Experiment

Naive architecture GPT with strong training recipe. Architecture is standard (no tricks); training uses modded-nanogpt's Muon+Adam optimizer setup.

## Architecture (naive, no tricks)

- GQA (8 query heads, 4 KV heads)
- RoPE (base=10000)
- RMSNorm without learnable weight (pre-norm)
- QK-Norm (RMS normalize Q and K before RoPE)
- Tied Embeddings
- GELU MLP
- Standard residual `x + sublayer(x)`
- CastedLinear (fp32 master weights + bf16 matmul)
- restore_low_dim_params_to_fp32

## Architectural Tricks Removed (vs exp/baseline-sp1024)

| Removed Trick | Replaced With |
|---|---|
| U-Net skip connections | Standard sequential blocks |
| ReLU^2 MLP | GELU MLP |
| Q-Gain (learnable per-head scale) | Removed |
| Learnable resid_mix (x0 mixing) | Standard residual `x + sublayer(x)` |
| Learnable attn_scale / mlp_scale | Standard residual |
| Logit softcap | Raw logits |
| Input RMSNorm on embedding | Removed |
| Zero-init for output projections | Default PyTorch init |
| Small embedding init (std=0.005) | Default nn.Embedding init |

## Training Recipe (strong, from modded-nanogpt)

- **Muon** optimizer for 2D matrix params (lr=0.04)
- **Adam** for scalar/control params (lr=0.04)
- **Adam** for token embedding (lr=0.05 tied / 0.6 untied)
- **Adam** for lm_head when untied (lr=0.008)
- Muon momentum warmup (0.85 → 0.95 over 500 steps)
- No weight decay, no grad clip
- JIT warmup (20 steps) with state reset
- Linear warmdown in last 1200 steps

## Files

- `train_gpt.py`: Baseline training script
- `baseline.json`: Experiment manifest
