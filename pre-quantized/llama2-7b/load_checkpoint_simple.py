#!/usr/bin/env python3
"""
Simplified script to test FlatQuant installation and CUDA availability.
This validates the environment is ready for FlatQuant models.
"""
import sys

print("=" * 80)
print("FlatQuant Environment Validation")
print("=" * 80)

# Check imports
print("\n1. Checking dependencies...")
try:
    import torch
    print(f"   ✓ torch: {torch.__version__}")
except ImportError as e:
    print(f"   ✗ torch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"   ✓ transformers: {transformers.__version__}")
except ImportError as e:
    print(f"   ✗ transformers: {e}")
    sys.exit(1)

try:
    import scipy
    print(f"   ✓ scipy: {scipy.__version__}")
except ImportError as e:
    print(f"   ✗ scipy: {e}")
    sys.exit(1)

try:
    import flatquant
    print(f"   ✓ flatquant: installed")
except ImportError as e:
    print(f"   ✗ flatquant: {e}")
    sys.exit(1)

# Check CUDA
print("\n2. Checking CUDA...")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("   ✗ CUDA not available - FlatQuant requires CUDA!")
    sys.exit(1)

# Check FlatQuant modules
print("\n3. Checking FlatQuant modules...")
try:
    from flatquant.flat_linear import FlatQuantizedLinear
    print("   ✓ FlatQuantizedLinear")
except ImportError as e:
    print(f"   ✗ FlatQuantizedLinear: {e}")

try:
    from flatquant.model_utils import get_model
    print("   ✓ get_model")
except ImportError as e:
    print(f"   ✗ get_model: {e}")

try:
    from flatquant.flat_utils import load_flat_matrices
    print("   ✓ load_flat_matrices")
except ImportError as e:
    print(f"   ✗ load_flat_matrices: {e}")

# Quick tensor test
print("\n4. Testing GPU tensor operations...")
try:
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    print(f"   ✓ GPU tensor operation successful")
    print(f"   Result shape: {y.shape}, device: {y.device}")
except Exception as e:
    print(f"   ✗ GPU operation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("SUCCESS: Environment is ready for FlatQuant!")
print("=" * 80)
print("\nNext steps:")
print("1. Download LLaMA-2-7B model")
print("2. Obtain FlatQuant W4A4KV4 matrices")
print("3. Run the full checkpoint loading script")
print("=" * 80 + "\n")
