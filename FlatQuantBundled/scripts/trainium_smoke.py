#!/usr/bin/env python3
"""
Minimal FlatQuant checks on AWS Trainium (Neuron / XLA).

Run on a Trainium or Trainium2 instance with the Neuron PyTorch venv activated, e.g.:

  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd /path/to/flatquant-trainium/FlatQuant
  pip install -e third-party/fast-hadamard-transform
  pip install -e .
  python scripts/trainium_smoke.py

Use --cpu to sanity-check the same code path on a laptop (no Neuron required).
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _resolve_device(force_cpu: bool):
    import torch

    if force_cpu:
        return torch.device("cpu"), None

    try:
        import torch_xla.core.xla_model as xm
    except ImportError as e:
        raise SystemExit(
            "torch_xla is not available. On Trainium, activate the Neuron venv, e.g.\n"
            "  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate\n"
            "Or run with --cpu on a dev machine.\n"
            f"Original error: {e}"
        ) from e

    return xm.xla_device(), xm


def _linear4bit_fp8_smoke(device, xm) -> None:
    import torch
    import deploy
    from deploy.nn import Linear4bit

    dt = getattr(torch, "float8_e4m3fn", None)
    if dt is None:
        raise RuntimeError("torch.float8_e4m3fn required (PyTorch >= 2.1)")

    torch.manual_seed(0)
    m = Linear4bit(32, 16, bias=False, dtype=torch.float16).to(device)
    m.weight.random_()
    m.weight_scales.fill_(0.1)

    x0 = torch.randn(4, 32, dtype=torch.float16, device=device)
    s = x0.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    qx = (x0 / s).to(dt)
    pt = deploy.PackedQuantizedTensor(qx, s.to(torch.float16))
    out = m(pt)
    if xm is not None:
        xm.mark_step()
    assert out.shape == (4, 16), out.shape
    print(f"  Linear4bit FP8 forward OK, shape={tuple(out.shape)}, dtype={out.dtype}")


def _kernel_smoke(device, xm) -> None:
    """Small kron_matmul on the target device (validates FP8 kernels + deploy path)."""
    import torch
    from deploy.kernels.pytorch.kron_matmul_pytorch import kron_matmul_pytorch

    dt = getattr(torch, "float8_e4m3fn", None)
    if dt is None:
        print("  Skip kron_matmul: float8_e4m3fn not available")
        return

    torch.manual_seed(1)
    M, N = 16, 8
    B, seq_len = 32, 16
    a = torch.randn(M, M, dtype=torch.float16, device=device)
    b = torch.randn(B, M, N, dtype=torch.float16, device=device)
    c = torch.randn(N, N, dtype=torch.float16, device=device)
    clip_max = torch.tensor(1.0, device=device)
    clip_min = torch.tensor(1.0, device=device)

    result = kron_matmul_pytorch(a, b, c, seq_len, clip_max, clip_min)
    if xm is not None:
        xm.mark_step()

    assert result.quantized_x.dtype == dt, result.quantized_x.dtype
    print(
        f"  kron_matmul_pytorch OK, q.shape={tuple(result.quantized_x.shape)}, "
        f"scales.shape={tuple(result.scales_x.shape)}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="FlatQuant Trainium / XLA smoke test")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU (for local dev; default is XLA / Neuron device)",
    )
    args = p.parse_args()

    device, xm = _resolve_device(args.cpu)
    print(f"Device: {device}")

    print("1) Linear4bit FP8 (PyTorch path, no CUDA extension)")
    _linear4bit_fp8_smoke(device, xm)

    print("2) kron_matmul_pytorch (FP8 kernel)")
    _kernel_smoke(device, xm)

    print("\nAll Trainium smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
