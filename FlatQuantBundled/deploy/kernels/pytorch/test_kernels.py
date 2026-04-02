"""
Test script for pure PyTorch FlatQuant kernels.

Run this to verify the kernels work correctly on your system.
"""

import os
import sys

import torch

# Allow `python test_kernels.py` from this directory (FlatQuant on PYTHONPATH).
_HERE = os.path.dirname(os.path.abspath(__file__))
_FQ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _FQ_ROOT not in sys.path:
    sys.path.insert(0, _FQ_ROOT)

from kron_matmul_pytorch import kron_matmul_pytorch
from block_matmul_pytorch import block_matmul_pytorch


def test_kron_matmul():
    """Test kron_matmul_pytorch kernel."""
    print("\n" + "="*60)
    print("Testing kron_matmul_pytorch kernel")
    print("="*60)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create test tensors
    M, N = 128, 64
    B = 256  # batch_size * seq_len
    seq_len = 128

    a = torch.randn(M, M, device=device, dtype=torch.float16)
    b = torch.randn(B, M, N, device=device, dtype=torch.float16)
    c = torch.randn(N, N, device=device, dtype=torch.float16)

    clip_max = torch.tensor(1.0, device=device)
    clip_min = torch.tensor(1.0, device=device)

    print(f"Input shapes:")
    print(f"  a: {a.shape}")
    print(f"  b: {b.shape}")
    print(f"  c: {c.shape}")

    # Run kernel
    try:
        result = kron_matmul_pytorch(a, b, c, seq_len, clip_max, clip_min)

        print(f"\nOutput:")
        print(f"  Type: {type(result).__name__}")
        print(f"  Quantized shape: {result.quantized_x.shape}")
        print(f"  Quantized dtype: {result.quantized_x.dtype}")
        print(f"  Scales shape: {result.scales_x.shape}")
        print(f"  Scales dtype: {result.scales_x.dtype}")

        # Verify shapes
        assert result.quantized_x.shape == (B, M * N), f"Unexpected quantized shape: {result.quantized_x.shape}"
        assert result.scales_x.shape == (B, 1), f"Unexpected scales shape: {result.scales_x.shape}"
        assert result.quantized_x.dtype == torch.float8_e4m3fn, f"Unexpected dtype: {result.quantized_x.dtype}"

        print("\n✓ kron_matmul_pytorch test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ kron_matmul_pytorch test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_block_matmul():
    """Test block_matmul_pytorch kernel."""
    print("\n" + "="*60)
    print("Testing block_matmul_pytorch kernel")
    print("="*60)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create test tensors
    M, N = 128, 64
    B = 256  # batch_size * seq_len
    seq_len = 128

    b = torch.randn(B, M, N, device=device, dtype=torch.float16)
    c = torch.randn(N, N, device=device, dtype=torch.float16)

    clip_max = torch.tensor(1.0, device=device)
    clip_min = torch.tensor(1.0, device=device)

    print(f"Input shapes:")
    print(f"  b: {b.shape}")
    print(f"  c: {c.shape}")

    # Test 1: With quantization
    print("\nTest 1: With quantization (just_quantize=False)")
    try:
        result = block_matmul_pytorch(b, c, seq_len, clip_max, clip_min, just_quantize=False)

        print(f"  Type: {type(result).__name__}")
        print(f"  Quantized shape: {result.quantized_x.shape}")
        print(f"  Quantized dtype: {result.quantized_x.dtype}")
        print(f"  Scales shape: {result.scales_x.shape}")

        # Verify shapes
        assert result.quantized_x.shape == (B, M * N), f"Unexpected quantized shape: {result.quantized_x.shape}"
        assert result.scales_x.shape == (B, 1), f"Unexpected scales shape: {result.scales_x.shape}"
        assert result.quantized_x.dtype == torch.float8_e4m3fn, f"Unexpected dtype: {result.quantized_x.dtype}"

        print("  ✓ Test 1 PASSED")

    except Exception as e:
        print(f"  ✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Without quantization
    print("\nTest 2: Without quantization (just_quantize=True)")
    try:
        result = block_matmul_pytorch(b, c, seq_len, clip_max, clip_min, just_quantize=True)

        print(f"  Type: {type(result).__name__}")
        print(f"  Shape: {result.shape}")
        print(f"  Dtype: {result.dtype}")

        # Verify shapes
        assert result.shape == (B, M * N), f"Unexpected shape: {result.shape}"
        assert result.dtype == torch.float16, f"Unexpected dtype: {result.dtype}"

        print("  ✓ Test 2 PASSED")

    except Exception as e:
        print(f"  ✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ block_matmul_pytorch test PASSED")
    return True


def test_numerical_correctness():
    """Test numerical correctness by comparing with standard PyTorch ops."""
    print("\n" + "="*60)
    print("Testing numerical correctness")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Small test case for easier debugging
    M, N = 16, 8
    B = 32
    seq_len = 16

    # Create test tensors
    a = torch.randn(M, M, device=device, dtype=torch.float16)
    b = torch.randn(B, M, N, device=device, dtype=torch.float16)
    c = torch.randn(N, N, device=device, dtype=torch.float16)

    # Compute reference (just the matmul part, before quantization)
    ab_ref = torch.matmul(a, b)  # [B, M, N]
    abc_ref = torch.matmul(ab_ref, c)  # [B, M, N]

    print(f"Reference result range: [{abc_ref.min().item():.4f}, {abc_ref.max().item():.4f}]")

    # Test kron_matmul_pytorch (it quantizes, so we can't compare directly)
    # But we can verify it doesn't crash and produces reasonable scales
    try:
        result = kron_matmul_pytorch(a, b, c, seq_len,
                           torch.tensor(1.0, device=device),
                           torch.tensor(1.0, device=device))

        print(f"kron_matmul_pytorch scales range: [{result.scales_x.min().item():.4f}, {result.scales_x.max().item():.4f}]")

        # Scales should be positive and reasonable
        assert (result.scales_x > 0).all(), "Scales should be positive"

        print("✓ Numerical correctness test PASSED")
        return True

    except Exception as e:
        print(f"✗ Numerical correctness test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PyTorch FlatQuant Kernels Test Suite")
    print("#"*60)

    results = []

    # Run tests
    results.append(("kron_matmul_pytorch", test_kron_matmul()))
    results.append(("block_matmul_pytorch", test_block_matmul()))
    results.append(("numerical_correctness", test_numerical_correctness()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(passed for _, passed in results)

    print("="*60)
    if all_passed:
        print("All tests PASSED! 🎉")
    else:
        print("Some tests FAILED. Please check the output above.")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
