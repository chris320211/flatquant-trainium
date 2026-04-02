"""
Pure PyTorch implementations of FlatQuant kernels.

These implementations use FP8 quantization and are compatible with:
- AWS Trainium
- Standard PyTorch/CUDA
- Any PyTorch backend

All kernels maintain similar interfaces to the original Triton implementations
but are implemented using pure PyTorch operations for maximum portability.
"""

from .kron_matmul_pytorch import kron_matmul_pytorch, quantize_fp8 as kron_quantize_fp8
from .block_matmul_pytorch import block_matmul_pytorch, quantize_fp8 as block_quantize_fp8

__all__ = [
    'kron_matmul_pytorch',
    'block_matmul_pytorch',
    'kron_quantize_fp8',
    'block_quantize_fp8',
]
