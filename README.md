# Bag of Tricks for Transformers

A unified benchmarking framework for systematically evaluating training techniques, architectural modifications, and optimization strategies for large language model (LLM) pretraining.

## Motivation

The Transformer ecosystem has accumulated a large number of training "tricks" — spanning optimizers (e.g., Muon, SOAP), learning rate and batch schedules, architectural variants (e.g., SwiGLU, value embeddings, U-Net skips, XSA), initialization schemes, data ordering strategies, and more. These techniques originate from diverse sources such as the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), [Parameter Golf](https://github.com/KellerJordan/parameter-golf), and [Slowrun](https://github.com/KellerJordan/slowrun) competitions, yet are rarely compared *under identical, controlled conditions*.

This repository provides a **standardized experimental harness** that isolates the effect of each trick by holding everything else constant — same model skeleton, same data pipeline, same tokenizer, same hardware budget — so that observed gains can be attributed with confidence.

## Evaluation Protocol

Every trick is evaluated under two complementary ablation regimes:

| Regime | Control Variable | What It Measures |
|---|---|---|
| **Fixed Compute** | Wall-clock time (= approximate FLOPs) | *Efficiency*: how much validation loss improves within a fixed compute budget. |
| **Fixed Tokens** | Total training tokens | *Sample efficiency*: how much validation loss improves given the same amount of data. |

By reporting results on both axes, we disentangle tricks that merely redistribute compute (e.g., trading depth for width) from those that yield genuine algorithmic improvements.

## Repository Structure

```
├── exp/                        # Experiment directory (one sub-folder per trick)
│   ├── baseline-sp1024/        # Reference baseline
│   │   ├── train_gpt.py        # Trainer script
│   │   └── baseline-sp1024.json# Experiment manifest
│   ├── muon/                   # Example trick: Muon optimizer
│   └── run_experiments.py      # Unified experiment scheduler
├── data/                       # Data pipeline & tokenizer assets
├── docs/                       # Design documentation
├── TRICK_SUMMARY_TRAIN_ARCH.md # Curated catalog of known tricks
└── pyproject.toml
```

## Quick Start

### Adding a New Trick

Taking `my_trick` as an example:

**1. Create the method directory and prepare a trainer script**

```bash
mkdir -p exp/my_trick
cp exp/baseline-sp1024/train_gpt.py exp/my_trick/train_gpt.py
```

Edit `exp/my_trick/train_gpt.py` and annotate every modification with:
```python
# trick: <brief description of the change>
```

Keep everything else identical to the baseline to ensure a fair comparison.

**2. Create an experiment manifest**

```bash
cp exp/baseline-sp1024/baseline-sp1024.json exp/my_trick/my_trick.json
```

Update `trainer_path` and `name` in `exp/my_trick/my_trick.json`:

```json
{
  "version": 1,
  "trainer_path": "exp/my_trick/train_gpt.py",
  "experiments": [
    {
      "name": "my_trick-fixed_time_10min",
      "control": { "mode": "fixed_compute", "target_wallclock_seconds": 600 }
    },
    {
      "name": "my_trick-fixed_tokens_10b",
      "control": { "mode": "fixed_tokens", "target_train_tokens": 10485760000 }
    }
  ]
}
```

**3. Dry-run to verify the configuration**

```bash
uv run python exp/run_experiments.py exp/my_trick/my_trick.json --dry-run
```

**4. Launch**

```bash
uv run python exp/run_experiments.py exp/my_trick/my_trick.json
```

All experiments in the manifest are executed sequentially on 8×H100 GPUs.

### Output

Each experiment produces a `result.json` containing:

- **Control**: `mode`, `target / actual` tokens and wall-clock seconds.
- **Metrics**: `final_val_bpb` (bits-per-byte on held-out validation).
- **Model**: architecture config and total parameter count.

Compare against the baseline `result.json` to quantify the gain.

## Trick Catalog

See [TRICK_SUMMARY_TRAIN_ARCH.md](TRICK_SUMMARY_TRAIN_ARCH.md) for a curated inventory of 40+ training and architecture tricks drawn from nanogpt-speedrun, parameter-golf, and slowrun, organized by category (optimizer, schedule, architecture, data, initialization, etc.).

## Acknowledgments

This project builds upon the open-source training recipes and competitive results from the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), [Parameter Golf](https://github.com/KellerJordan/parameter-golf), and [Slowrun](https://github.com/KellerJordan/slowrun) communities.
