# loop-share-all — all blocks weight-shared (small exploratory run)

## Trick

```
input → [loop_blocks (shared)] × num_loop_repeats → output
```

With small defaults (`num_layers=9`, `num_loop_repeats=3`):

- 3 unique block definitions (`num_loop_layers = num_layers / num_loop_repeats = 3`)
- Looped 3 times
- **9 effective forward passes** (same depth as baseline)
- **~1/3 the unique transformer parameters** of baseline (small: ~6.55M vs ~17.56M)

No skip connections.

## Scope

This is an **exploratory run, not a strict ablation** of weight-sharing vs baseline — it changes both the sharing pattern and the parameter budget at once. It is included to (a) sanity-check the loop implementation under maximal sharing, and (b) get a data point on how much capacity collapse the standard `mlp_mult=2`/`num_layers=9` configuration can absorb.

The clean ablations for weight sharing are `loop-share-mid` and `loop-share-first`, which both hold parameter count equal to baseline.

## Hyperparameters

| Name | Default | Meaning |
|---|---|---|
| `NUM_LOOP_REPEATS` | 3 | times the shared stack is re-applied |
| `NUM_LOOP_LAYERS` | derived | `num_layers // num_loop_repeats` (asserted exact) |

## Training recipe

Identical to baseline.

## Files

- `train_gpt.py`
- `loop-share-all.json` / `loop-share-all-medium.json`
