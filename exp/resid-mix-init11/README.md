# resid-mix-init11 — `resid-mix` with non-identity init `[1.1, 0.1]`

## Trick

Architecturally **identical** to `resid-mix`. The only difference is the initialization of the learnable per-channel `resid_mix` parameter:

| Experiment | `resid_mix` init | Behavior at step 0 |
|---|---|---|
| resid-mix | `[1.0, 0.0]` | identical to baseline (no x0 mixing) |
| **resid-mix-init11** | **`[1.1, 0.1]`** | non-identity: 10 % of the original embedding is mixed in at every layer |

## Intended comparison

This experiment is best read as an **ablation of `resid-mix`** (different init), not as a comparison vs baseline. Two changes from baseline are bundled here: (a) the `resid_mix` architecture itself, and (b) the non-identity init. Comparing only to `resid-mix` isolates (b).

For attribution table:

| Variant | resid_mix arch | init | Compare against |
|---|---|---|---|
| baseline | no | — | (reference) |
| resid-mix | yes | `[1.0, 0.0]` | baseline |
| **resid-mix-init11** | yes | `[1.1, 0.1]` | resid-mix |

## Optimizer routing & param count

Same as `resid-mix` (Muon for matrix params, AdamW for the `resid_mix` 2D parameter via `CONTROL_TENSOR_NAME_PATTERNS`).

## Training recipe

Identical to baseline.

## Files

- `train_gpt.py`
- `resid-mix-init11.json` / `resid-mix-init11-medium.json`
