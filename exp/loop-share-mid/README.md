# loop-share-mid — middle blocks weight-shared via looping

## Trick

Architecture splits the transformer into three parts:

```
input → encoder_blocks (unique) → [loop_blocks (shared)] × num_loop_repeats → decoder_blocks (unique) → output
```

With small defaults (`num_layers=9`, `num_unique_encoder=0.1111`, `num_loop_layers=0.3333`, `num_loop_repeats=3`):

- 1 unique encoder block
- 3 shared middle blocks, looped 3 times
- 5 unique decoder blocks
- **9 unique block definitions** (same parameter count as baseline)
- **15 effective forward passes** through the stack (~1.67× FLOPs vs baseline)

No skip connections; the shared blocks are re-applied with identical weights each loop pass.

## Motivation: anchor–refine–project

| Role | Layers | Why unique vs shared |
|---|---|---|
| **Anchor** | first 1 block | handles raw token embeddings → internal repr, IO-specialized |
| **Refine** | middle 3 blocks × 3 | mid-level abstractions can be iteratively composed by re-applying the same op |
| **Project** | last 5 blocks | maps refined repr → vocab logits, task-specialized |

The hypothesis is that the middle of a transformer operates entirely in the model's internal representation space and is more amenable to weight sharing than first/last layers.

## Ablation context

| Variant | Block defs | Sharing pattern | Effective depth | Params (small) |
|---|---|---|---|---|
| baseline | 9 unique | none | 9 | ~17.56M |
| loop-share-all | 3 unique | all 3 blocks × 3 | 9 | ~6.55M |
| loop-share-first | 9 defs (4 shared + 5 unique) | first 4 × 3 | 17 | ~17.56M |
| **loop-share-mid** | **9 defs (1 enc + 3 shared + 5 dec)** | **middle 3 × 3** | **15** | **~17.56M** |

**Fairness:** same param count as baseline, but ~1.67× FLOPs. In `fixed_compute` (20 min) the model processes fewer tokens per second; in `fixed_tokens` (10B) it gets the full data budget at the cost of longer wall-clock.

## Hyperparameters (set via env or `defaults` in JSON)

| Name | Default | Meaning |
|---|---|---|
| `NUM_UNIQUE_ENCODER` | 0.1111 (× num_layers) | unique blocks before the loop |
| `NUM_LOOP_LAYERS` | 0.3333 (× num_layers) | shared blocks inside the loop |
| `NUM_LOOP_REPEATS` | 3 | times the shared blocks are re-applied |

Decoder count is derived: `num_decoder = num_layers - num_unique_encoder - num_loop_layers`.

## Training recipe

Identical to baseline: Muon for 2D matrix params (encoder + loop + decoder all included), AdamW for `tok_emb` (wd=0) / `lm_head` (wd=0.1) / scalar params (wd=0).

## Files

- `train_gpt.py`
- `loop-share-mid.json` / `loop-share-mid-medium.json`
