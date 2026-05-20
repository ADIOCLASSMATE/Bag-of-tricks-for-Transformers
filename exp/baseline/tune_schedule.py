#!/usr/bin/env python3
"""
Schedule hyperparameter tuning for baseline models at fixed target_train_tokens.

Tunes warmdown_iters, muon_momentum_warmup_steps, and muon_momentum_warmup_start
against final validation loss (full training runs).

Usage:
  # Generate tuning manifests
  python exp/baseline/tune_schedule.py generate --model small
  python exp/baseline/tune_schedule.py generate --model medium
  python exp/baseline/tune_schedule.py generate --model both

  # Dry-run to inspect
  python exp/run_experiments.py exp/baseline/tune_small.json --dry-run

  # Run tuning
  python exp/run_experiments.py exp/baseline/tune_small.json
  python exp/run_experiments.py exp/baseline/tune_medium.json

  # Analyze results
  python exp/baseline/tune_schedule.py analyze --model small
  python exp/baseline/tune_schedule.py analyze --model medium
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path


# --- Model configs ---

SMALL_BASE = {
    "num_layers": 9,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "mlp_mult": 2,
    "vocab_size": 50257,
    "train_batch_tokens": 524288,
    "total_params": 67_978_240,
    "grad_accum_steps": 1,
}

MEDIUM_BASE = {
    "num_layers": 18,
    "model_dim": 1024,
    "num_heads": 16,
    "num_kv_heads": 8,
    "mlp_mult": 2,
    "vocab_size": 50257,
    "train_batch_tokens": 524288,
    "total_params": 235_046_912,
    "grad_accum_steps": 2,
}


def total_steps(model_cfg: dict, target_tokens: int) -> int:
    return math.ceil(target_tokens / model_cfg["train_batch_tokens"])


def build_tuning_grid(model_cfg: dict, target_tokens: int) -> list[dict]:
    steps = total_steps(model_cfg, target_tokens)

    warmdown_pcts = [5, 8, 10, 15]
    muon_warmup_pcts = [3, 5, 8]
    muon_warmup_starts = [0.80, 0.85]

    experiments = []
    for wd_pct in warmdown_pcts:
        for mw_pct in muon_warmup_pcts:
            for mw_start in muon_warmup_starts:
                warmdown_iters = max(1, int(steps * wd_pct / 100))
                muon_warmup_steps = max(1, int(steps * mw_pct / 100))
                experiments.append({
                    "warmdown_iters": warmdown_iters,
                    "warmdown_pct": wd_pct,
                    "muon_momentum_warmup_steps": muon_warmup_steps,
                    "muon_warmup_pct": mw_pct,
                    "muon_momentum_warmup_start": mw_start,
                })
    return experiments


def generate_manifest(
    model_cfg: dict,
    target_tokens: int,
    model_name: str,
    output_path: Path,
) -> None:
    steps = total_steps(model_cfg, target_tokens)
    grid = build_tuning_grid(model_cfg, target_tokens)

    experiments = []
    for i, combo in enumerate(grid):
        name = (
            f"tune-{model_name}-"
            f"wd{combo['warmdown_pct']}pct-"
            f"mw{combo['muon_warmup_pct']}pct-"
            f"mus{str(combo['muon_momentum_warmup_start']).replace('.', '')}"
        )
        experiments.append({
            "name": name,
            "trainer_path": "exp/baseline/train_gpt.py",
            "overrides": {
                "warmdown_iters": combo["warmdown_iters"],
                "muon_momentum_warmup_steps": combo["muon_momentum_warmup_steps"],
                "muon_momentum_warmup_start": combo["muon_momentum_warmup_start"],
            },
            "control": {
                "mode": "fixed_tokens",
                "target_train_tokens": target_tokens,
            },
        })

    num_combos = len(experiments)
    print(f"Tuning grid: {num_combos} experiments for {model_name}")
    print(f"  Steps: {steps}  Target tokens: {target_tokens:,}")
    print(f"  warmdown_iters range: [{experiments[0]['overrides']['warmdown_iters']}, "
          f"{experiments[-1]['overrides']['warmdown_iters']}]")
    print(f"  muon_warmup_steps range: [{min(e['overrides']['muon_momentum_warmup_steps'] for e in experiments)}, "
          f"{max(e['overrides']['muon_momentum_warmup_steps'] for e in experiments)}]")

    manifest = {
        "version": 1,
        "trainer_path": "exp/baseline/train_gpt.py",
        "launcher": {
            "nproc_per_node": 4,
            "master_port_base": 29500,
        },
        "defaults": {
            "data_path": "./data/datasets/fineweb-edu_100BT_gpt2",
            "tokenizer_path": "gpt2",
            "vocab_size": model_cfg["vocab_size"],
            "train_batch_tokens": model_cfg["train_batch_tokens"],
            "train_seq_len": 1024,
            "grad_accum_steps": model_cfg["grad_accum_steps"],
            "warmup_steps": 50,
            "enable_wandb": 1,
            "wandb_mode": "offline",
            "wandb_project": "bag-of-tricks-for-transformers",
            "wandb_entity": "Bag-of-Tricks",
            "wandb_group": f"tune-schedule-{model_name}",
            "num_layers": model_cfg["num_layers"],
            "model_dim": model_cfg["model_dim"],
            "num_heads": model_cfg["num_heads"],
            "num_kv_heads": model_cfg["num_kv_heads"],
            "mlp_mult": model_cfg["mlp_mult"],
            "tied_embed_lr": 0.05,
            "embed_lr": 0.6,
            "head_lr": 0.008,
            "matrix_lr": 0.04,
            "scalar_lr": 0.04,
            "muon_momentum": 0.95,
            "muon_backend_steps": 5,
            "grad_clip_norm": 0.0,
            "tokenizer_type": "gpt2",
        },
        "experiments": experiments,
    }

    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


def analyze_results(logs_dir: Path, model_name: str) -> None:
    """Parse result.json files from tuning runs and rank by val_loss."""
    results = []
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        result_file = run_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        metrics = data.get("metrics", {})
        hparams = data.get("hyperparameters", {})
        control = data.get("control", {})
        results.append({
            "run_id": data.get("run_id", run_dir.name),
            "val_loss": metrics.get("final_val_loss", float("inf")),
            "val_bpb": metrics.get("final_val_bpb", float("inf")),
            "train_steps": data.get("training", {}).get("final_step", 0),
            "warmdown_iters": hparams.get("warmdown_iters", "?"),
            "muon_momentum_warmup_steps": hparams.get("muon_momentum_warmup_steps", "?"),
            "muon_momentum_warmup_start": hparams.get("muon_momentum_warmup_start", "?"),
            "actual_tokens": control.get("actual_train_tokens", 0),
            "training_time_s": data.get("training", {}).get("training_time_ms", 0) / 1000,
        })

    if not results:
        print(f"No completed results found in {logs_dir}")
        return

    results.sort(key=lambda r: r["val_loss"])

    print(f"\n{'='*90}")
    print(f"  Tuning Results for {model_name} (ranked by val_loss)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'val_loss':<10} {'val_bpb':<10} {'wd_iters':<10} {'mu_wu_steps':<12} {'mu_wu_start':<12} {'steps':<8} {'time_s':<8}")
    print(f"{'-'*90}")

    for rank, r in enumerate(results, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(
            f"{rank:<5} {r['val_loss']:<10.4f} {r['val_bpb']:<10.4f} "
            f"{r['warmdown_iters']:<10} {r['muon_momentum_warmup_steps']:<12} "
            f"{r['muon_momentum_warmup_start']:<12} {r['train_steps']:<8} {r['training_time_s']:<8.0f}{marker}"
        )

    best = results[0]
    print(f"\nBest config:")
    print(f"  warmdown_iters = {best['warmdown_iters']}")
    print(f"  muon_momentum_warmup_steps = {best['muon_momentum_warmup_steps']}")
    print(f"  muon_momentum_warmup_start = {best['muon_momentum_warmup_start']}")
    print(f"  val_loss = {best['val_loss']:.4f}  val_bpb = {best['val_bpb']:.4f}")


def cmd_generate(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    target_tokens_small = int(SMALL_BASE["total_params"] * 50)
    target_tokens_medium = int(MEDIUM_BASE["total_params"] * 50)

    if args.model in ("small", "both"):
        generate_manifest(SMALL_BASE, target_tokens_small, "small",
                          root / "tune_small.json")
    if args.model in ("medium", "both"):
        generate_manifest(MEDIUM_BASE, target_tokens_medium, "medium",
                          root / "tune_medium.json")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    logs_root = root / "logs"

    if args.model in ("small", "both"):
        # Find most recent batch directory for small tuning
        batch_dirs = sorted([d for d in logs_root.iterdir()
                             if d.is_dir() and any("tune-small" in sd.name for sd in d.iterdir())],
                            reverse=True)
        if batch_dirs:
            analyze_results(batch_dirs[0], "small")
        else:
            print("No small-model tuning results found.")

    if args.model in ("medium", "both"):
        batch_dirs = sorted([d for d in logs_root.iterdir()
                             if d.is_dir() and any("tune-medium" in sd.name for sd in d.iterdir())],
                            reverse=True)
        if batch_dirs:
            analyze_results(batch_dirs[0], "medium")
        else:
            print("No medium-model tuning results found.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Schedule hyperparameter tuning for baseline")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate tuning manifest JSON files")
    gen.add_argument("--model", choices=["small", "medium", "both"], default="both")

    ana = sub.add_parser("analyze", help="Analyze tuning results and show best config")
    ana.add_argument("--model", choices=["small", "medium", "both"], default="both")
    ana.add_argument("--batch-dir", type=Path, default=None,
                     help="Specific batch directory to analyze")

    args = parser.parse_args()
    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
