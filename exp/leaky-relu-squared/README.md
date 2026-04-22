# Leaky-ReLU-Squared — leaky_relu(x, 0.5).square() Activation

## Method Overview

This experiment replaces the baseline MLP activation `relu(x).square()` with **`leaky_relu(x, 0.5).square()`**.

The goal is to preserve negative-side gradient flow through the MLP while keeping the same overall relu²-style nonlinearity pattern.

## Impact on Training

- **Parameters**: No change
- **Throughput**: Essentially unchanged
- **Peak memory**: unchanged at 10,246 MiB
- **Optimization**: Eliminates hard zero gradient on negative pre-activations

## Key Differences from Baseline

| Component | baseline-sp1024 | leaky-relu-squared |
|---|---|---|
| MLP activation | `relu(x).square()` | **`leaky_relu(x, 0.5).square()`** |
| Everything else | identical | identical |

## Results

### Fixed Compute (10 min wall-clock)

| Metric | baseline-sp1024 | leaky-relu-squared | Δ |
|---|---|---|---|
| **Val BPB** | 1.2194 | **1.2173** | **−0.0021** |
| Val Loss | 2.0589 | **2.0554** | **−0.0035** |
| Steps completed | 13,263 | 13,295 | +32 |
| Tokens processed | 6.95B | 6.97B | +0.02B |

### Fixed Tokens (10B tokens)

| Metric | baseline-sp1024 | leaky-relu-squared | Δ |
|---|---|---|---|
| **Val BPB** | 1.2118 | **1.2097** | **−0.0021** |
| Val Loss | 2.0460 | **2.0425** | **−0.0035** |
| Steps | 19,074 | 19,074 | 0 |
| Wall-clock time | 832.8s | 843.4s | +10.6s |

## BPB Analysis

This is a clean, low-cost win.

- **Fixed compute**: The model improves by 0.0021 BPB with essentially identical throughput.
- **Fixed tokens**: The same 0.0021 BPB gain appears again, which makes the result look robust rather than noise.
- **Interpretation**: The activation change is small, cheap, and consistently helpful. It is weaker than large capacity changes, but much cheaper to adopt.

## Files

- `train_gpt.py`: Modified training script with `leaky_relu(x, 0.5).square()` in `MLP.forward`
- `leaky-relu-squared.json`: Experiment manifest
- `logs/`: Training output
