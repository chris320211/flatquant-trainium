#!/usr/bin/env python3
"""
Minimal FlatQuant checks on AWS Trainium (Neuron / XLA).

Run on a Trainium or Trainium2 instance with the Neuron PyTorch venv activated, e.g.:

  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd /path/to/flatquant-trainium/FlatQuantBundled
  pip install -e .
  python scripts/trainium_smoke.py

Use --cpu to sanity-check on a laptop (no Neuron required).

Note on FP8: Neuron compiler currently rejects torch.float8_e4m3fn (F8E4M3FN) in the
XLA graph on trn2 (NCC_ESPP047). This script runs INT4 Linear4bit on XLA for a real
device compile, and runs FP8 kernel checks on CPU when XLA is used.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _is_xla(device) -> bool:
    import torch

    if isinstance(device, torch.device):
        return device.type == "xla"
    return str(device).startswith("xla")


def _sync(xm) -> None:
    """Prefer torch_xla.sync(); fall back to xm.mark_step()."""
    try:
        import torch_xla

        sync = getattr(torch_xla, "sync", None)
        if callable(sync):
            sync()
            return
    except ImportError:
        pass
    if xm is not None:
        xm.mark_step()


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

    try:
        import torch_xla

        dev = getattr(torch_xla, "device", None)
        if callable(dev):
            return dev(), xm
    except ImportError:
        pass
    return xm.xla_device(), xm


def _linear4bit_int4_smoke(device, xm) -> None:
    """INT4 activations on device — avoids F8E4M3FN in Neuron graph."""
    import torch
    import deploy
    from deploy.functional.quantization import pack_i4
    from deploy.nn import Linear4bit

    torch.manual_seed(0)
    m = Linear4bit(8, 4, bias=False, dtype=torch.float16).to(device)
    m.weight.random_()
    m.weight_scales.fill_(0.1)
    q = torch.randint(-4, 4, (2, 8), dtype=torch.int8, device=device)
    qx = pack_i4(q)
    sx = torch.ones(2, 1, dtype=torch.float16, device=device) * 0.05
    pt = deploy.PackedQuantizedTensor(qx, sx)
    out = m(pt)
    _sync(xm)
    assert out.shape == (2, 4), out.shape
    print(
        f"  Linear4bit INT4 forward OK, shape={tuple(out.shape)}, dtype={out.dtype}"
    )


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
    _sync(xm)
    assert out.shape == (4, 16), out.shape
    print(f"  Linear4bit FP8 forward OK, shape={tuple(out.shape)}, dtype={out.dtype}")


def _kernel_smoke(device, xm) -> None:
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
    _sync(xm)

    assert result.quantized_x.dtype == dt, result.quantized_x.dtype
    print(
        f"  kron_matmul_pytorch OK, q.shape={tuple(result.quantized_x.shape)}, "
        f"scales.shape={tuple(result.scales_x.shape)}"
    )


def main() -> int:
    import torch

    p = argparse.ArgumentParser(description="FlatQuant Trainium / XLA smoke test")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU (for local dev; default is XLA / Neuron device)",
    )
    args = p.parse_args()

    device, xm = _resolve_device(args.cpu)
    print(f"Device: {device}")

    if args.cpu or not _is_xla(device):
        print("1) Linear4bit FP8 (PyTorch path, no CUDA extension)")
        _linear4bit_fp8_smoke(device, xm)
        print("2) kron_matmul_pytorch (FP8 kernel)")
        _kernel_smoke(device, xm)
    else:
        print(
            "1) Linear4bit INT4 on XLA (Neuron does not compile F8E4M3FN in graph — NCC_ESPP047)"
        )
        _linear4bit_int4_smoke(device, xm)
        print("2) FP8 checks on CPU (same kernels as GPU/Trainium path; not lowered to NEFF as FP8)")
        cpu = torch.device("cpu")
        _linear4bit_fp8_smoke(cpu, None)
        _kernel_smoke(cpu, None)

    print("\nAll Trainium smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
