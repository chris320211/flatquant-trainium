"""
CPU / Mac smoke tests (no CUDA, no Trainium).

Run from anywhere:
  cd FlatQuant && python -m unittest tests.test_mac_cpu_smoke -v

With pytest (optional):
  cd FlatQuant && pip install pytest && pytest tests/test_mac_cpu_smoke.py -v

Tier A — always try these (only needs torch):
  - FP8 dtype check, scale alignment, FP8 dequant loaded via importlib (no full `deploy` import)

Tier B — skipped unless `import deploy` works (install fast_hadamard_transform):
  pip install -e third-party/fast-hadamard-transform
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_fp8_utils_standalone():
    """Load `deploy/nn/fp8_utils.py` without importing `deploy` (avoids fast_hadamard)."""
    path = os.path.join(_REPO_ROOT, "deploy", "nn", "fp8_utils.py")
    spec = importlib.util.spec_from_file_location("fp8_utils_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestFp8UtilsStandalone(unittest.TestCase):
    """Runs on Mac with only PyTorch + this repo checkout."""

    @classmethod
    def setUpClass(cls):
        cls.fp8 = _load_fp8_utils_standalone()
        cls.torch = __import__("torch")

    def test_align_scale_transposes_bsz_1_seq(self):
        torch = self.torch
        fp8 = self.fp8
        qx = torch.zeros(2, 4, 8)
        sx_bad = torch.ones(2, 1, 4)
        sx = fp8.align_scale_for_activations(qx, sx_bad)
        self.assertEqual(sx.shape, (2, 4, 1))

    def test_fp8_dequant_roundtrip_if_supported(self):
        torch = self.torch
        fp8 = self.fp8
        dt = getattr(torch, "float8_e4m3fn", None)
        if dt is None:
            self.skipTest("torch.float8_e4m3fn not available (upgrade PyTorch >= 2.1)")

        class Packed:
            pass

        p = Packed()
        x0 = torch.randn(3, 16, dtype=torch.float32)
        s = x0.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        p.quantized_x = (x0 / s).to(dt)
        p.scales_x = s.to(torch.float16)
        out = fp8.dequant_fp8_to_float(p, out_dtype=torch.float32)
        self.assertEqual(out.shape, x0.shape)
        self.assertLess((out - x0).abs().max().item(), 0.25)


class TestDeployOptional(unittest.TestCase):
    """Requires `pip install -e third-party/fast-hadamard-transform` and `PYTHONPATH=.` or `cd FlatQuant`."""

    deploy = None

    @classmethod
    def setUpClass(cls):
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        try:
            import deploy as d

            cls.deploy = d
        except Exception as e:
            cls.deploy = None
            cls._import_error = e

    def setUp(self):
        if self.deploy is None:
            err = getattr(self.__class__, "_import_error", Exception("unknown"))
            self.skipTest(
                f"deploy import failed ({err!r}). Install: pip install -e third-party/fast-hadamard-transform"
            )

    def test_sym_quant_pytorch_cpu(self):
        import torch

        torch.manual_seed(0)
        x = torch.randn(4, 32, dtype=torch.float16)
        scale = (torch.max(torch.abs(x), dim=-1)[0] / 7.0).to(torch.float16).unsqueeze(-1)
        q = self.deploy.sym_quant(x, scale)
        self.assertEqual(q.dtype, torch.uint8)
        self.assertEqual(q.shape, (4, 16))

    def test_linear4bit_int4_pytorch_cpu(self):
        import torch
        from deploy.nn import Linear4bit
        from deploy.functional.quantization import pack_i4

        torch.manual_seed(1)
        m = Linear4bit(8, 4, bias=False, dtype=torch.float16)
        m.weight.random_()
        m.weight_scales.fill_(0.1)
        q = torch.randint(-4, 4, (2, 8), dtype=torch.int8)
        qx = pack_i4(q)
        sx = torch.ones(2, 1, dtype=torch.float16) * 0.05
        pt = self.deploy.PackedQuantizedTensor(qx, sx)
        out = m(pt)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(out.is_floating_point())
        # CPU F.linear may promote float16 activations to float32
        self.assertIn(out.dtype, (torch.float16, torch.float32))

    def test_linear4bit_fp8_pytorch_cpu(self):
        import torch
        from deploy.nn import Linear4bit

        dt = getattr(torch, "float8_e4m3fn", None)
        if dt is None:
            self.skipTest("torch.float8_e4m3fn not available")

        torch.manual_seed(2)
        m = Linear4bit(8, 4, bias=False, dtype=torch.float16)
        m.weight.random_()
        m.weight_scales.fill_(0.1)
        x0 = torch.randn(2, 8, dtype=torch.float16)
        s = x0.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        qx = (x0 / s).to(dt)
        pt = self.deploy.PackedQuantizedTensor(qx, s.to(torch.float16))
        out = m(pt)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(out.is_floating_point())
        # CPU F.linear may promote float16 activations to float32
        self.assertIn(out.dtype, (torch.float16, torch.float32))


class TestPytorchKernelsScript(unittest.TestCase):
    def test_script_exists(self):
        script = os.path.join(
            _REPO_ROOT, "deploy", "kernels", "pytorch", "test_kernels.py"
        )
        self.assertTrue(os.path.isfile(script), msg=f"Missing {script}")


if __name__ == "__main__":
    unittest.main()
