# Pure PyTorch FlatQuant Kernels

This directory contains pure PyTorch implementations of the FlatQuant kernels, designed to be compatible with AWS Trainium and other PyTorch backends.

## Overview

The original FlatQuant kernels were implemented using Triton for CUDA GPUs. These PyTorch implementations provide the same functionality using pure PyTorch operations, enabling deployment on:

- **AWS Trainium** (primary target)
- Standard PyTorch/CUDA environments
- Any PyTorch-compatible backend

## Key Differences from Original Triton Implementation

1. **Quantization Format**: Uses PyTorch native **FP8** (`torch.float8_e4m3fn`) instead of custom int4 bit-packing
2. **Performance**: Optimized for PyTorch's native operations and automatic compilation
3. **Portability**: No dependencies on Triton or CUDA-specific features
4. **Simplicity**: Easier to understand and modify

## Available Kernels

### 1. Kronecker Matrix Multiplication (`kron_matmul_pytorch`)

Performs the operation: `a @ b @ c` with FP8 quantization.

```python
from deploy.kernels.pytorch import kron_matmul_pytorch

# a: [M, M] - left transformation matrix
# b: [B, M, N] - batch of input matrices
# c: [N, N] - right transformation matrix
# Returns: PackedQuantizedTensor with FP8 quantized result

result = kron_matmul_pytorch(
    a, b, c,
    seq_len=128,
    clip_factor_a_max=1.0,  # learnable clipping factor
    clip_factor_a_min=1.0   # learnable clipping factor
)
```

**Key Features:**
- Efficient batched matrix multiplication
- Learned clipping factors for optimal quantization range
- FP8 quantization with per-batch scaling

### 2. Block Matrix Multiplication (`block_matmul_pytorch`)

Performs the operation: `b @ c` with optional FP8 quantization.

```python
from deploy.kernels.pytorch import block_matmul_pytorch

# b: [B, M, N] - batch of input matrices
# c: [N, N] - transformation matrix
# Returns: PackedQuantizedTensor with FP8 quantized result

result = block_matmul_pytorch(
    b, c,
    seq_len=128,
    clip_factor_a_max=1.0,
    clip_factor_a_min=1.0,
    just_quantize=False  # Set to True to skip quantization
)
```

**Key Features:**
- Simple batched matrix multiplication
- Optional quantization (controlled by `just_quantize` flag)
- Learned clipping factors

## Quantization Details

### FP8 Format

Both kernels use `torch.float8_e4m3fn` for quantization:
- **4-bit exponent**, **3-bit mantissa**
- Better suited for forward pass/activations
- Native PyTorch support (hardware-accelerated on supported devices)

### Learned Clipping

The quantization process uses learnable clipping factors to optimize the quantization range:

```python
sigmoid_max = 1.0 / (1.0 + exp(-clip_factor_a_max))
sigmoid_min = 1.0 / (1.0 + exp(-clip_factor_a_min))

# Apply clipping to computed min/max values
xmax = xmax * sigmoid_max
xmin = xmin * sigmoid_min

# Compute scale based on clipped range
scale = max(abs(xmin), xmax)
```

This allows the model to learn optimal quantization ranges during training.

### Per-Batch Scaling

Each batch element gets its own scale factor for better precision:
- Scales shape: `[B, 1]`
- Quantized values shape: `[B, M*N]` (flattened)

## Performance Optimization

### For Trainium

These kernels are designed to work efficiently with Trainium's XLA compiler:

1. **Pure PyTorch ops**: No custom CUDA kernels that would block XLA compilation
2. **Batched operations**: Uses `torch.matmul` with broadcasting for efficient batch processing
3. **Contiguous memory**: All inputs are verified to be contiguous for optimal memory access

### General Tips

- Use `torch.compile()` for additional speedups on supported backends
- Ensure input tensors are on the correct device before calling
- For Trainium: Use Neuron SDK and torch-neuronx for compilation

```python
# Example with torch.compile (PyTorch 2.0+)
kron_matmul_compiled = torch.compile(kron_matmul_pytorch)
result = kron_matmul_compiled(a, b, c, seq_len, clip_max, clip_min)
```

## Benchmarking

Both modules include benchmark functions compatible with the original Triton interface:

```python
from deploy.kernels.pytorch.kron_matmul_pytorch import benchmark

# Returns: (perf_ms, perf_max_ms, perf_min_ms, ms, max_ms, min_ms)
metrics = benchmark(
    B=1024,      # Total batch size
    M=128,       # Matrix dimension M
    N=128,       # Matrix dimension N
    S=128,       # Sequence length
    provider='pytorch'
)
```

## Migration Guide

If you're migrating from the Triton kernels:

### Before (Triton):
```python
from deploy.kernels.kron_matmul import kron_matmul
result = kron_matmul(a, b, c, seq_len, clip_max, clip_min)
```

### After (PyTorch):
```python
from deploy.kernels.pytorch import kron_matmul_pytorch
result = kron_matmul_pytorch(a, b, c, seq_len, clip_max, clip_min)
```

The interface is nearly identical! The main differences are:
1. Function name has `_pytorch` suffix
2. Internal quantization format (FP8 vs int4)

## Testing

To verify the kernels work correctly:

```python
import torch
from deploy.kernels.pytorch import kron_matmul_pytorch, block_matmul_pytorch

# Create test tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.randn(128, 128, device=device, dtype=torch.float16)
b = torch.randn(256, 128, 64, device=device, dtype=torch.float16)
c = torch.randn(64, 64, device=device, dtype=torch.float16)

clip_max = torch.tensor(1.0, device=device)
clip_min = torch.tensor(1.0, device=device)

# Test kron_matmul_pytorch
result = kron_matmul_pytorch(a, b, c, seq_len=128,
                             clip_factor_a_max=clip_max,
                             clip_factor_a_min=clip_min)

print(f"Result type: {type(result)}")
print(f"Quantized shape: {result.quantized_x.shape}")
print(f"Scales shape: {result.scales_x.shape}")
```

## Requirements

- PyTorch >= 2.1 (for FP8 support)
- For Trainium: torch-neuronx and Neuron SDK

## Future Improvements

Potential areas for enhancement:

1. **Mixed precision training**: Support for different FP8 formats (e4m3 vs e5m2)
2. **Fused operations**: Combine quantization with activation functions
3. **Trainium-specific optimizations**: Custom XLA lowerings if needed
4. **Block-wise quantization**: Finer-grained quantization for better accuracy
