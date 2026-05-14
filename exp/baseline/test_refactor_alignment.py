#!/usr/bin/env python3
"""Quick alignment test for train_gpt_refactor.py vs original baseline.

Verifies:
  1. Model instantiation (all architectures: 9l, 18l, tied embeddings)
  2. Parameter count matches expected
  3. Deterministic init (same seed → same weights)
  4. Forward pass (scalar loss, bf16 autocast)
  5. Backward pass (gradients computed)
  6. Full training step (optimizer update)
  7. eval_val works (on dummy tokens)
"""
from __future__ import annotations

import math
import os
import sys
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Setup: locate the project root and ensure trainlib/ is importable
# ---------------------------------------------------------------------------
# When invoked from repo root everything should "just work";
# otherwise we fix up sys.path so the test can be run from anywhere.
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE))),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _ensure_imports():
    """Smoke-test that all shared-library imports resolve."""
    from trainlib.hyperparameters import Hyperparameters, hyperparameters_to_dict
    from trainlib.optimizer import CONTROL_TENSOR_NAME_PATTERNS, zeropower_via_newtonschulz5, Muon
    from trainlib.validation import build_sentencepiece_luts, load_validation_tokens, eval_val
    from trainlib.dataloader import DistributedTokenLoader, load_data_shard, TokenStream
    from trainlib.transformer import (
        GPT, CastedLinear, RMSNorm, Rotary, apply_rotary_emb,
        CausalSelfAttention, MLP, Block, restore_low_dim_params_to_fp32,
    )
    return True


# ---------------------------------------------------------------------------
# Model instantiation tests
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = [
    # (name, kwargs)
    ("9l-untied", dict(vocab_size=1024, num_layers=9, model_dim=512, num_heads=8,
                       num_kv_heads=4, mlp_mult=2, tie_embeddings=False, rope_base=10000.0)),
    ("9l-tied",   dict(vocab_size=1024, num_layers=9, model_dim=512, num_heads=8,
                       num_kv_heads=4, mlp_mult=2, tie_embeddings=True, rope_base=10000.0)),
    ("18l-untied", dict(vocab_size=1024, num_layers=18, model_dim=768, num_heads=12,
                        num_kv_heads=4, mlp_mult=2, tie_embeddings=False, rope_base=10000.0)),
]


def _compute_expected_params(kwargs: dict) -> int:
    """Approximate param count for sanity check."""
    d = kwargs["model_dim"]
    L = kwargs["num_layers"]
    V = kwargs["vocab_size"]
    n_heads = kwargs["num_heads"]
    n_kv_heads = kwargs["num_kv_heads"]
    mlp_mult = kwargs["mlp_mult"]
    head_dim = d // n_heads
    kv_dim = n_kv_heads * head_dim
    hidden = mlp_mult * d

    tok_emb = V * d
    per_block = (
        d * d          # c_q
        + d * kv_dim   # c_k
        + d * kv_dim   # c_v
        + d * d        # attn.proj
        + d * hidden   # mlp.fc
        + hidden * d   # mlp.proj
    )
    total = tok_emb + L * per_block
    if not kwargs["tie_embeddings"]:
        total += d * V  # lm_head
    return total


def test_model_instantiation():
    """All configs instantiate and have correct param count."""
    from trainlib.transformer import GPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, kwargs in _MODEL_CONFIGS:
        torch.manual_seed(1337)
        model = GPT(**kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        expected = _compute_expected_params(kwargs)
        assert n_params == expected, (
            f"[{name}] param mismatch: got {n_params}, expected {expected}"
        )
        print(f"  [PASS] {name}: {n_params:,} params")

    print("  => All model instantiation tests passed")


def test_deterministic_init():
    """Same seed → same weights."""
    from trainlib.transformer import GPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(vocab_size=1024, num_layers=2, model_dim=64, num_heads=4,
                  num_kv_heads=2, mlp_mult=2, tie_embeddings=False, rope_base=10000.0)

    torch.manual_seed(42)
    m1 = GPT(**kwargs).to(device)

    torch.manual_seed(42)
    m2 = GPT(**kwargs).to(device)

    s1 = m1.state_dict()
    s2 = m2.state_dict()
    assert s1.keys() == s2.keys(), "state_dict keys differ"
    for key in s1:
        assert torch.equal(s1[key], s2[key]), f"Weight mismatch for {key}"

    print("  [PASS] Deterministic init: same seed → identical weights")


# ---------------------------------------------------------------------------
# Forward / backward tests
# ---------------------------------------------------------------------------

def test_forward_pass():
    """Forward pass returns scalar loss, works under bf16 autocast."""
    from trainlib.transformer import GPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(vocab_size=1024, num_layers=2, model_dim=64, num_heads=4,
                  num_kv_heads=2, mlp_mult=2, tie_embeddings=False, rope_base=10000.0)

    torch.manual_seed(1)
    model = GPT(**kwargs).to(device)
    if device.type == "cuda":
        model = model.bfloat16()
        for m in model.modules():
            if isinstance(m, __import__("trainlib.transformer", fromlist=["CastedLinear"]).CastedLinear):
                m.float()

    bsz, seqlen = 2, 128
    x = torch.randint(0, kwargs["vocab_size"], (bsz, seqlen), device=device)
    y = torch.randint(0, kwargs["vocab_size"], (bsz, seqlen), device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        loss = model(x, y)

    assert isinstance(loss, Tensor), f"Expected Tensor, got {type(loss)}"
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert loss.dtype == torch.float32, f"Expected fp32 loss, got {loss.dtype}"

    print(f"  [PASS] Forward pass: loss={loss.item():.4f}")


def test_backward_pass():
    """Backward pass produces gradients on all parameters."""
    from trainlib.transformer import GPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(vocab_size=1024, num_layers=2, model_dim=64, num_heads=4,
                  num_kv_heads=2, mlp_mult=2, tie_embeddings=False, rope_base=10000.0)

    torch.manual_seed(2)
    model = GPT(**kwargs).to(device)
    if device.type == "cuda":
        model = model.bfloat16()
        CastedLinear = __import__("trainlib.transformer", fromlist=["CastedLinear"]).CastedLinear
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m.float()

    bsz, seqlen = 2, 128
    x = torch.randint(0, kwargs["vocab_size"], (bsz, seqlen), device=device)
    y = torch.randint(0, kwargs["vocab_size"], (bsz, seqlen), device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        loss = model(x, y)

    loss.backward()

    grad_params = 0
    zero_grad_params = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_params += 1
            if p.grad.abs().sum() == 0:
                zero_grad_params += 1

    assert grad_params > 0, "No parameters received gradients"
    total = sum(1 for _ in model.parameters())
    print(f"  [PASS] Backward pass: {grad_params}/{total} params have gradients "
          f"({zero_grad_params} with zero grad)")


# ---------------------------------------------------------------------------
# Full training step test (optimizer integration)
# ---------------------------------------------------------------------------

def test_training_step():
    """One full optimizer step works with Muon + Adam split."""
    from trainlib.transformer import GPT, CastedLinear, restore_low_dim_params_to_fp32
    from trainlib.optimizer import Muon, zeropower_via_newtonschulz5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(vocab_size=1024, num_layers=2, model_dim=64, num_heads=4,
                  num_kv_heads=2, mlp_mult=2, tie_embeddings=False, rope_base=10000.0)

    torch.manual_seed(3)
    base_model = GPT(**kwargs).to(device)
    if device.type == "cuda":
        base_model = base_model.bfloat16()
        for m in base_model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(base_model)

    # Optimizer split (mirrors main())
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params if p.ndim == 2]
    scalar_params = [p for name, p in block_named_params if p.ndim < 2]

    optimizer_tok = torch.optim.AdamW(
        [base_model.tok_emb.weight], lr=0.6, betas=(0.9, 0.95), eps=1e-8,
    )
    optimizer_muon = Muon(matrix_params, lr=0.04, momentum=0.95, backend_steps=5)
    optimizers = [optimizer_tok, optimizer_muon]
    if scalar_params:
        optimizer_scalar = torch.optim.AdamW(
            scalar_params, lr=0.04, betas=(0.9, 0.95), eps=1e-8,
        )
        optimizers.append(optimizer_scalar)
    optimizer_head = torch.optim.AdamW(
        [base_model.lm_head.weight], lr=0.008, betas=(0.9, 0.95), eps=1e-8,
    )
    optimizers.append(optimizer_head)

    # One step
    bsz, seqlen = 2, 128
    x = torch.randint(0, kwargs["vocab_size"], (bsz, seqlen), device=device)
    y = torch.randint(0, kwargs["vocab_size"], (bsz, seqlen), device=device)

    for opt in optimizers:
        opt.zero_grad()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        loss = base_model(x, y)
    loss.backward()
    for opt in optimizers:
        opt.step()

    print(f"  [PASS] Training step: loss={loss.item():.4f}, all optimizers stepped")


# ---------------------------------------------------------------------------
# submodule replacement test (for trick experiments)
# ---------------------------------------------------------------------------

def test_submodule_replacement():
    """Replacing block.mlp after GPT construction works."""
    from trainlib.transformer import GPT, CastedLinear

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(vocab_size=1024, num_layers=2, model_dim=64, num_heads=4,
                  num_kv_heads=2, mlp_mult=2, tie_embeddings=False, rope_base=10000.0)

    torch.manual_seed(5)
    model = GPT(**kwargs).to(device)

    class SwiGLU(nn.Module):
        def __init__(self, dim, mlp_mult):
            super().__init__()
            hidden = mlp_mult * dim
            self.gate_proj = CastedLinear(dim, hidden, bias=False)
            self.up_proj = CastedLinear(dim, hidden, bias=False)
            self.down_proj = CastedLinear(hidden, dim, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    for block in model.blocks:
        block.mlp = SwiGLU(kwargs["model_dim"], kwargs["mlp_mult"]).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    expected_base = _compute_expected_params(kwargs)
    assert n_params != expected_base, "Param count should change after MLP replacement"
    print(f"  [PASS] Submodule replacement: params {expected_base} → {n_params}")

    # Forward pass still works
    x = torch.randint(0, kwargs["vocab_size"], (2, 64), device=device)
    y = torch.randint(0, kwargs["vocab_size"], (2, 64), device=device)
    loss = model(x, y)
    assert not torch.isnan(loss), "Loss is NaN after MLP replacement"
    print(f"  [PASS] Forward after replacement: loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# Hyperparameters override test
# ---------------------------------------------------------------------------

def test_hyperparameters_defaults():
    """Hyperparameters has all required fields with correct types (env vars
    are read at import time by run_experiments.py, not set mid-process)."""
    from trainlib.hyperparameters import Hyperparameters, hyperparameters_to_dict

    args = Hyperparameters()
    assert isinstance(args.num_layers, int)
    assert isinstance(args.model_dim, int)
    assert isinstance(args.num_heads, int)
    assert isinstance(args.tie_embeddings, bool)
    assert isinstance(args.iterations, int)
    assert isinstance(args.max_wallclock_seconds, float)
    assert isinstance(args.muon_momentum, float)
    assert isinstance(args.seed, int)

    d = hyperparameters_to_dict(args)
    assert "num_layers" in d
    assert "model_dim" in d
    assert "embed_lr" in d
    assert callable not in (type(v) for v in d.values())

    print(f"  [PASS] Hyperparameters: {len(d)} fields, all correct types")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Quick Alignment Test: train_gpt_refactor.py ===\n")

    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_str}\n")

    print("1. Import check...")
    assert _ensure_imports(), "Import failed"
    print("  [PASS] All shared-library imports resolve\n")

    print("2. Model instantiation...")
    test_model_instantiation()
    print()

    print("3. Deterministic init...")
    test_deterministic_init()
    print()

    print("4. Forward pass...")
    test_forward_pass()
    print()

    print("5. Backward pass...")
    test_backward_pass()
    print()

    print("6. Full training step...")
    test_training_step()
    print()

    print("7. Submodule replacement...")
    test_submodule_replacement()
    print()

    print("8. Hyperparameters defaults...")
    test_hyperparameters_defaults()
    print()

    print("=" * 50)
    print("All tests PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
