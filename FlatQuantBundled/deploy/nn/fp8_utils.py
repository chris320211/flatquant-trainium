"""
FP8 activation helpers for Trainium / PyTorch backends without CUDA INT4 matmul.

Kernels store activations as float8 with a per-row scale. This module dequantizes to
a high-precision dtype for `F.linear` while INT4 weights are expanded via `unpack_i4`.
"""

from __future__ import annotations

import torch


def fp8_dtypes() -> tuple[torch.dtype, ...]:
    out = []
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            out.append(dt)
    return tuple(out)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in fp8_dtypes()


def align_scale_for_activations(
    qx: torch.Tensor, scales_x: torch.Tensor
) -> torch.Tensor:
    """
    Broadcast scales to multiply element-wise with quantized activations.

    Handles legacy [bsz, 1, seq_len] layout (transpose to [bsz, seq_len, 1] when paired
    with [bsz, seq_len, hidden]).
    """
    sx = scales_x
    if qx.dim() == 3 and sx.dim() == 3:
        if sx.shape[0] == qx.shape[0] and sx.shape[1] == 1 and sx.shape[2] == qx.shape[1]:
            sx = sx.transpose(1, 2)
    while sx.dim() < qx.dim():
        sx = sx.unsqueeze(-1)
    return sx


def dequant_fp8_to_float(
    packed, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    packed.quantized_x: float8 tensor (last dim = full features, not packed int4)
    packed.scales_x:    fp16/fp32 scales broadcastable to qx
    """
    qx = packed.quantized_x
    sx = packed.scales_x
    if not is_fp8_dtype(qx.dtype):
        raise TypeError(f"Expected FP8 quantized_x, got {qx.dtype}")
    sx = align_scale_for_activations(qx, sx)
    x = qx.to(torch.float32) * sx.to(torch.float32)
    return x.to(out_dtype)


def int4_weight_to_float(
    weight_uint8: torch.Tensor,
    weight_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Expand packed INT4 weights to dense float for PyTorch matmul."""
    from deploy.functional import unpack_i4

    w_i = unpack_i4(weight_uint8).to(torch.float32)
    w = w_i * weight_scales.to(torch.float32)
    return w.to(out_dtype)


def dequant_int4_packed_activations(
    packed, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """INT4 packed uint8 activations × per-row scale → float (Trainium / PyTorch path)."""
    from deploy.functional import unpack_i4

    qx = unpack_i4(packed.quantized_x).to(torch.float32)
    sx = align_scale_for_activations(qx, packed.scales_x)
    return (qx * sx.to(torch.float32)).to(out_dtype)
