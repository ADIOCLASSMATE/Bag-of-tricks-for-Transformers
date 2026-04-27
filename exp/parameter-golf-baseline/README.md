# Baseline SP1024 Experiment

> **DEPRECATED**: This experiment has been superseded by `exp/baseline/`.
> This code retains qk_gain_init=1.5, logit_softcap=30.0, and tied_embed_init_std=0.005
> from the parameter-golf training script. These features are absent from the clean baseline.
> Use `exp/baseline/` as the reference for all ablation comparisons.

This directory contains the baseline SP1024 experiment, including the training script and manifest. The baseline model is inspired by parameters-golf.

## Files

- `train_gpt.py`: Baseline training script
- `baseline-sp1024.json`: Baseline manifest
- `logs/`: Training output (automatically generated)