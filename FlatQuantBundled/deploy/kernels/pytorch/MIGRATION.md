# Migration from Triton to PyTorch Kernels

This document describes the migration from CUDA/Triton kernels to pure PyTorch implementations for AWS Trainium compatibility.

## Overview

The original FlatQuant implementation used Triton kernels optimized for NVIDIA GPUs. To enable deployment on AWS Trainium and other PyTorch-compatible accelerators, we've created pure PyTorch implementations that maintain the same interface while using portable PyTorch operations.

## What Changed

### 1. Quantization Format

**Before (Triton):**
- Custom int4 quantization with bit-packing
- Two 4-bit values packed into one uint8
- Range: [-8, 7] (4-bit signed integer)

**After (PyTorch):**
- Native PyTorch FP8 (`torch.float8_e4m3fn`)
- 4-bit exponent, 3-bit mantissa
- Hardware-accelerated on supported devices
- Better precision than int4

### 2. Implementation Details

**Before (Triton):**
```python
@triton.jit
def matmul_kernel(...):
    # Low-level GPU kernel code
    # Manual memory management
    # Explicit loop unrolling
    # Custom bit manipulation
```

**After (PyTorch):**
```python
def kron_matmul(a, b, c, ...):
    # High-level PyTorch operations
    ab = torch.matmul(a, b)
    abc = torch.matmul(ab, c)
    return quantize_fp8(abc, ...)
```

### 3. Performance Characteristics

| Aspect | Triton | PyTorch |
|--------|--------|---------|
| **Optimization** | Hand-tuned for CUDA | Relies on PyTorch/XLA compiler |
| **Portability** | NVIDIA GPUs only | Any PyTorch backend |
| **Maintainability** | Requires GPU expertise | Standard PyTorch code |
| **Debugging** | Difficult | Easy with standard tools |

## API Compatibility

The PyTorch implementations maintain **similar API** to the Triton versions with a naming convention to distinguish them:

```python
# Triton version (original)
from deploy.kernels.kron_matmul import kron_matmul

# PyTorch version (new)
from deploy.kernels.pytorch import kron_matmul_pytorch

# Both have the same signature
result = kron_matmul(a, b, c, seq_len, clip_max, clip_min)  # Triton
result = kron_matmul_pytorch(a, b, c, seq_len, clip_max, clip_min)  # PyTorch
```

## File Structure

```
deploy/kernels/
├── kron_matmul.py               # Original Triton implementation (reference)
├── block_matmul.py              # Original Triton implementation (reference)
└── pytorch/                     # PyTorch / Trainium implementations
    ├── __init__.py
    ├── kron_matmul_pytorch.py   # PyTorch FP8 version
    ├── block_matmul_pytorch.py  # PyTorch FP8 version
    ├── README.md                # Documentation
    ├── MIGRATION.md             # This file
    └── test_kernels.py          # Test suite
```

## Using the PyTorch Kernels

### Option 1: Direct Import

```python
from deploy.kernels.pytorch import kron_matmul_pytorch, block_matmul_pytorch

# Use the PyTorch versions
result = kron_matmul_pytorch(a, b, c, seq_len, clip_max, clip_min)
```

### Option 2: Aliased Import

Use aliasing to minimize code changes:

```python
# Old code using Triton
from deploy.kernels.kron_matmul import kron_matmul

# New code using PyTorch (with alias for compatibility)
from deploy.kernels.pytorch import kron_matmul_pytorch as kron_matmul

# Your existing code works unchanged
result = kron_matmul(a, b, c, seq_len, clip_max, clip_min)
```

### Option 3: Runtime Selection

Choose implementation based on device:

```python
import torch

if torch.cuda.is_available():
    from deploy.kernels.kron_matmul import kron_matmul  # Triton (faster on CUDA)
else:
    from deploy.kernels.pytorch import kron_matmul_pytorch as kron_matmul  # PyTorch (portable)
```

## Deployment on Trainium

### Step 1: Install Dependencies

```bash
# Install Neuron SDK (on Trainium instance)
pip install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com

# Verify installation
python -c "import torch_neuronx; print(torch_neuronx.__version__)"
```

### Step 2: Use PyTorch Kernels

```python
import torch
import torch_neuronx
from deploy.kernels.pytorch import kron_matmul_pytorch, block_matmul_pytorch

# Move tensors to Neuron device
device = 'xla'  # or torch_neuronx.device()
a = a.to(device)
b = b.to(device)
c = c.to(device)

# Use kernels normally
result = kron_matmul_pytorch(a, b, c, seq_len, clip_max, clip_min)
```

### Step 3: Trace for Trainium (Optional)

For best performance, trace your model:

```python
import torch_neuronx

# Trace the kernel
traced_kron = torch_neuronx.trace(
    kron_matmul_pytorch,
    (a_example, b_example, c_example, seq_len, clip_max, clip_min)
)

# Use traced version
result = traced_kron(a, b, c, seq_len, clip_max, clip_min)
```

## Testing

Run the test suite to verify everything works:

```bash
cd FlatQuant/deploy/kernels/pytorch
python test_kernels.py
```

Expected output:
```
############################################################
# PyTorch FlatQuant Kernels Test Suite
############################################################

============================================================
Testing kron_matmul_pytorch kernel
============================================================
Using device: cuda

✓ kron_matmul_pytorch test PASSED

============================================================
Testing block_matmul_pytorch kernel
============================================================
Using device: cuda

✓ block_matmul_pytorch test PASSED

============================================================
Test Summary
============================================================
kron_matmul_pytorch.................... ✓ PASSED
block_matmul_pytorch................... ✓ PASSED
numerical_correctness.................. ✓ PASSED
============================================================
All tests PASSED! 🎉
============================================================
```

## Performance Considerations

### Expected Performance Changes

1. **On CUDA GPUs:**
   - Triton: ~2-3x faster (hand-optimized)
   - PyTorch: Good, but not as optimized

2. **On Trainium:**
   - Triton: Not supported
   - PyTorch: Good performance with XLA compilation

3. **On CPU:**
   - Both implementations work, but slow

### Optimization Tips for Trainium

1. **Use torch.compile():**
   ```python
   kron_matmul_compiled = torch.compile(kron_matmul_pytorch)
   ```

2. **Batch operations:**
   - Process multiple sequences together
   - Larger batch sizes = better hardware utilization

3. **Keep tensors contiguous:**
   ```python
   a = a.contiguous()
   b = b.contiguous()
   c = c.contiguous()
   ```

4. **Use appropriate dtypes:**
   - FP16 for inputs (memory efficient)
   - FP8 for quantized outputs (even more efficient)

## Troubleshooting

### Issue: "torch.float8_e4m3fn not available"

**Solution:** Upgrade PyTorch to >= 2.1
```bash
pip install --upgrade torch
```

### Issue: Slow performance on Trainium

**Solutions:**
1. Enable XLA compilation
2. Use larger batch sizes
3. Trace the model with torch_neuronx.trace()

### Issue: Numerical differences vs Triton

**Explanation:**
- FP8 has different precision than int4
- This is expected and usually acceptable
- FP8 generally has better accuracy

### Issue: Out of memory

**Solutions:**
1. Reduce batch size
2. Use gradient checkpointing
3. Enable memory optimization in Neuron compiler

## Backwards Compatibility

The PyTorch implementations are designed to be **drop-in replacements**. However, note:

1. **Output format:** Same (`PackedQuantizedTensor`)
2. **Quantization format:** Different (FP8 vs int4)
3. **Numerical values:** Slightly different due to quantization format
4. **Performance:** Different depending on device

If you need bit-exact compatibility with the Triton version, you must use the original Triton kernels on CUDA.

## Future Work

Potential improvements:

1. **Automatic backend selection:** Choose Triton or PyTorch based on device
2. **Mixed precision:** Support different FP8 variants
3. **Fused kernels:** Combine quantization with other operations
4. **Trainium-specific optimizations:** Custom XLA lowerings

## Questions?

If you encounter issues or have questions about the migration:

1. Check the [README.md](README.md) for usage examples
2. Run [test_kernels.py](test_kernels.py) to verify installation
3. Review the implementation in [kron_matmul.py](kron_matmul.py) and [block_matmul.py](block_matmul.py)
