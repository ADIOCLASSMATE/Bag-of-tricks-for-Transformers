# loop-share-first — first blocks weight-shared via looping

## Trick

```
input → [loop_blocks (shared)] × num_loop_repeats → decoder_blocks (unique) → output
```

With small defaults (`num_layers=9`, `num_loop_layers=0.4444`, `num_loop_repeats=3`):

- 4 shared blocks, looped 3 times
- 5 unique decoder blocks
- **9 unique block definitions** (same parameter count as baseline)
- **17 effective forward passes** through the stack (~1.89× FLOPs vs baseline)

No skip connections.

## Motivation

Companion to `loop-share-mid`: tests whether looping over the **first** blocks (which interact with raw embeddings) is as effective as looping over the middle. The "anchor–refine–project" hypothesis predicts this should be **worse** than `loop-share-mid`, because the first block has IO-specialized work that benefits from unique parameters.

## Ablation context

| Variant | Block defs | Sharing pattern | Effective depth | Params (small) |
|---|---|---|---|---|
| baseline | 9 unique | none | 9 | ~17.56M |
| loop-share-all | 3 unique | all 3 blocks × 3 | 9 | ~6.55M |
| **loop-share-first** | **9 defs (4 shared + 5 unique)** | **first 4 × 3** | **17** | **~17.56M** |
| loop-share-mid | 9 defs (1 enc + 3 shared + 5 dec) | middle 3 × 3 | 15 | ~17.56M |

**Fairness:** same param count as baseline, but ~1.89× FLOPs. Same caveats as `loop-share-mid`.

## Hyperparameters

| Name | Default | Meaning |
|---|---|---|
| `NUM_LOOP_LAYERS` | 0.4444 (× num_layers) | shared first blocks |
| `NUM_LOOP_REPEATS` | 3 | times the shared blocks are re-applied |

Decoder count is derived.

## Training recipe

Identical to baseline (see top-level baseline README).

## Files

- `train_gpt.py`
- `loop-share-first.json` / `loop-share-first-medium.json`
