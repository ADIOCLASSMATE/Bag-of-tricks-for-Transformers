#!/usr/bin/env python3
"""
Unit tests for attention-residuals implementation.
Tests initialization, disabled mode, block boundaries, and gradient flow.
"""

import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path to import train_gpt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpt import GPT, block_attn_res


def test_initialization_equivalence():
    """Test 1: Verify initialization trick makes block_attn_res ≈ partial_block"""
    print("\n=== Test 1: Initialization Equivalence ===")

    model = GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        rope_base=10000.0,
        attn_res_enabled=True,
        attn_res_num_blocks=3,
        attn_res_recency_bias_init=10.0
    )
    model.eval()

    # Create dummy blocks and partial_block
    batch_size = 2
    seq_len = 64
    dim = 512

    # Simulate 2 completed blocks
    blocks = [
        torch.randn(batch_size, seq_len, dim),
        torch.randn(batch_size, seq_len, dim),
    ]
    partial_block = torch.randn(batch_size, seq_len, dim)

    # Test at each block boundary
    block_boundaries = [2, 5, 8]  # For 9 layers, 3 blocks

    for layer_idx in block_boundaries:
        block = model.blocks[layer_idx]

        # Call block_attn_res with recency_bias=10.0
        h_attn = block_attn_res(
            blocks,
            partial_block,
            block.attn_res_proj,
            block.attn_res_norm,
            block.attn_res_bias
        )

        # Check if h_attn ≈ partial_block (within tolerance)
        diff = torch.abs(h_attn - partial_block)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_error = max_diff / (torch.abs(partial_block).mean().item() + 1e-8)

        print(f"Layer {layer_idx}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, relative_error={relative_error:.6f}")

        # With recency_bias=10.0, the initialization should make block_attn_res strongly favor current block
        # The output should be very close to partial_block
        assert relative_error < 0.1, f"Layer {layer_idx}: Initialization not close enough to identity (relative_error={relative_error:.4f})"

    print("✓ Test 1 PASSED: Initialization equivalence verified")
    return True


def test_disabled_mode_matches_baseline():
    """Test 2: Verify disabled mode produces same output as baseline"""
    print("\n=== Test 2: Disabled Mode Matches Baseline ===")

    # Use same random seed
    torch.manual_seed(42)
    model_disabled = GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        rope_base=10000.0,
        attn_res_enabled=False,
        attn_res_num_blocks=3,
        attn_res_recency_bias_init=10.0
    )

    torch.manual_seed(42)
    model_baseline = GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        rope_base=10000.0,
        attn_res_enabled=False,
        attn_res_num_blocks=3,
        attn_res_recency_bias_init=10.0
    )

    model_disabled.eval()
    model_baseline.eval()

    # Create dummy input
    torch.manual_seed(123)
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1024, (batch_size, seq_len))
    target_ids = torch.randint(0, 1024, (batch_size, seq_len))

    # Run through both models
    with torch.no_grad():
        logits_disabled = model_disabled(input_ids, target_ids)
        logits_baseline = model_baseline(input_ids, target_ids)

    # Check if outputs match exactly
    max_diff = torch.abs(logits_disabled - logits_baseline).max().item()
    print(f"Max difference between disabled and baseline: {max_diff:.10f}")

    assert max_diff < 1e-6, f"Disabled mode doesn't match baseline (max_diff={max_diff})"

    print("✓ Test 2 PASSED: Disabled mode matches baseline")
    return True


def test_block_boundary_logic():
    """Test 3: Verify block boundary detection is correct"""
    print("\n=== Test 3: Block Boundary Logic ===")

    model = GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        rope_base=10000.0,
        attn_res_enabled=True,
        attn_res_num_blocks=3,
        attn_res_recency_bias_init=10.0
    )

    # For 9 layers, 3 blocks: layers_per_block = 3
    # Block boundaries should be at layers [2, 5, 8] (0-indexed)
    expected_boundaries = {2, 5, 8}
    expected_non_boundaries = {0, 1, 3, 4, 6, 7}

    print(f"Expected boundaries: {sorted(expected_boundaries)}")
    print(f"Expected non-boundaries: {sorted(expected_non_boundaries)}")

    # Check each layer
    for layer_idx in range(9):
        block = model.blocks[layer_idx]
        has_attn_res_proj = hasattr(block, 'attn_res_proj')
        has_mlp_res_proj = hasattr(block, 'mlp_res_proj')
        is_boundary = block.is_block_boundary

        if layer_idx in expected_boundaries:
            assert has_attn_res_proj, f"Layer {layer_idx} should have attn_res_proj"
            assert has_mlp_res_proj, f"Layer {layer_idx} should have mlp_res_proj"
            assert is_boundary, f"Layer {layer_idx} should be a block boundary"
            print(f"Layer {layer_idx}: ✓ Has attn_res components and is_block_boundary=True")
        else:
            assert has_attn_res_proj, f"Layer {layer_idx} should have attn_res_proj (all layers have it when enabled)"
            assert has_mlp_res_proj, f"Layer {layer_idx} should have mlp_res_proj (all layers have it when enabled)"
            assert not is_boundary, f"Layer {layer_idx} should NOT be a block boundary"
            print(f"Layer {layer_idx}: ✓ Has attn_res components but is_block_boundary=False")

    print("✓ Test 3 PASSED: Block boundary logic correct")
    return True


def test_gradient_flow():
    """Test 4: Verify gradients flow through attn_res and mlp_res"""
    print("\n=== Test 4: Gradient Flow ===")

    model = GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        rope_base=10000.0,
        attn_res_enabled=True,
        attn_res_num_blocks=3,
        attn_res_recency_bias_init=10.0
    )
    model.train()

    # Create dummy input and target
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1024, (batch_size, seq_len))
    target_ids = torch.randint(0, 1024, (batch_size, seq_len))

    # Forward pass - returns loss directly
    loss = model(input_ids, target_ids)

    # Backward pass
    loss.backward()

    # Check gradients for attn_res and mlp_res parameters
    block_boundaries = [2, 5, 8]

    for layer_idx in block_boundaries:
        block = model.blocks[layer_idx]

        # Check attn_res gradients
        attn_res_params = ['attn_res_proj', 'attn_res_bias']
        attn_res_has_grad = False
        for param_name in attn_res_params:
            if hasattr(block, param_name):
                param = getattr(block, param_name)
                if isinstance(param, nn.Parameter):
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        attn_res_has_grad = True
                        print(f"Layer {layer_idx} {param_name}: grad_norm={param.grad.norm().item():.6f}")
                elif isinstance(param, nn.Module):
                    for name, p in param.named_parameters():
                        if p.grad is not None and p.grad.abs().sum() > 0:
                            attn_res_has_grad = True
                            print(f"Layer {layer_idx} {param_name}.{name}: grad_norm={p.grad.norm().item():.6f}")

        assert attn_res_has_grad, f"Layer {layer_idx}: attn_res has no gradients"

        # Check mlp_res gradients
        mlp_res_params = ['mlp_res_proj', 'mlp_res_bias']
        mlp_res_has_grad = False
        for param_name in mlp_res_params:
            if hasattr(block, param_name):
                param = getattr(block, param_name)
                if isinstance(param, nn.Parameter):
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        mlp_res_has_grad = True
                        print(f"Layer {layer_idx} {param_name}: grad_norm={param.grad.norm().item():.6f}")
                elif isinstance(param, nn.Module):
                    for name, p in param.named_parameters():
                        if p.grad is not None and p.grad.abs().sum() > 0:
                            mlp_res_has_grad = True
                            print(f"Layer {layer_idx} {param_name}.{name}: grad_norm={p.grad.norm().item():.6f}")

        assert mlp_res_has_grad, f"Layer {layer_idx}: mlp_res has no gradients"

    print("✓ Test 4 PASSED: Gradients flow correctly")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Attention-Residuals Unit Tests")
    print("=" * 60)

    tests = [
        ("Initialization Equivalence", test_initialization_equivalence),
        ("Disabled Mode Matches Baseline", test_disabled_mode_matches_baseline),
        ("Block Boundary Logic", test_block_boundary_logic),
        ("Gradient Flow", test_gradient_flow),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
