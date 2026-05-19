# Baseline Experiment

Naive architecture GPT with strong training recipe. Architecture is standard (no tricks); training uses modded-nanogpt's Muon + AdamW optimizer setup.

## Architecture (naive, no tricks)

- GQA (8 query heads, 4 KV heads on small; 16 / 8 on medium)
- RoPE (base=10000)
- RMSNorm without learnable weight (pre-norm)
- QK-Norm (RMS normalize Q and K before RoPE)
- **Untied embeddings** (separate `tok_emb` and `lm_head`)
- GELU MLP
- Standard residual `x + sublayer(x)`
- CastedLinear (fp32 master weights + bf16 matmul)
- restore_low_dim_params_to_fp32

## Configs

| Model | Layers | Dim | Heads | KV Heads | MLP Mult | Params |
|---|---|---|---|---|---|---|
| small | 9 | 512 | 8 | 4 | 2 | ~17.56M |
| medium | 18 | 1024 | 16 | 8 | 2 | (re-measure after re-run) |

## Architectural Tricks Removed (vs parameter-golf-baseline)

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
| Tied embeddings | Untied (separate tok_emb and lm_head) |

## Training Recipe (strong, from modded-nanogpt)

- **Muon** optimizer for 2D matrix params in transformer blocks (lr=0.04)
- **AdamW** for scalar/control params (lr=0.04)
- **AdamW** for token embedding (lr=0.6 untied)
- **AdamW** for `lm_head` (lr=0.008, weight_decay=0.1)
- Weight decay: 0 on token embedding, 0 on Muon, 0 on scalar/control params, 0.1 on `lm_head` only
- Muon momentum warmup (0.85 → 0.95 over 800 steps)
- JIT warmup (20 steps) with state reset
- Linear warmdown in last 1500 steps (small) / 800 steps (medium)

## Control modes (per experiment in this manifest)

| Name | Mode | Target |
|---|---|---|
| `*-fixed_time_20min` | `fixed_compute` | 1200 s wall-clock, iterations capped at 1M |
| `*-fixed_tokens_10b` | `fixed_tokens` | 10B training tokens, no wall-clock cap |

## Files

- `train_gpt.py` — baseline trainer
- `baseline.json` / `baseline-medium.json` — small / medium manifests
