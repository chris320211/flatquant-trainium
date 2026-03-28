"""
FlatQuant setup.

CUDA extension (deploy._CUDA) is built only when:
  - CUDA is available (`nvcc` on PATH or CUDA_HOME set), AND
  - The FORCE_PYTORCH_ONLY env var is NOT set.

On Trainium / Mac / CPU-only machines, `pip install -e .` installs the
Python packages without the CUDA extension. The PyTorch / FP8 paths in
deploy/ work without it.
"""

import os
import pathlib
import shutil
import subprocess
import sys

from setuptools import setup

HERE = pathlib.Path(__file__).absolute().parent
setup_dir = str(HERE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Return True if nvcc is on PATH or CUDA_HOME is set."""
    if os.environ.get("FORCE_PYTORCH_ONLY"):
        return False
    if shutil.which("nvcc"):
        return True
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return True
    return False


def _install_fast_hadamard(extra_pip_flags=None):
    hadamard_dir = str(HERE / "third-party" / "fast-hadamard-transform")
    if not os.path.isdir(hadamard_dir):
        print(
            f"[setup] WARNING: {hadamard_dir} not found; "
            "skipping fast-hadamard-transform install. "
            "Run: pip install -e FlatQuantBundled/third-party/fast-hadamard-transform"
        )
        return
    pip = shutil.which("pip") or sys.executable + " -m pip"
    cmd = [pip, "install", "-e", hadamard_dir]
    if extra_pip_flags:
        cmd.extend(extra_pip_flags)
    subprocess.call(cmd)


def _remove_unwanted_nvcc_flags():
    try:
        import torch.utils.cpp_extension as torch_cpp_ext
    except ImportError:
        return
    for flag in (
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ):
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except (ValueError, AttributeError):
            pass


def _cuda_arch_flags():
    return [
        "-gencode", "arch=compute_75,code=sm_75",  # Turing
        "-gencode", "arch=compute_80,code=sm_80",  # Ampere
        "-gencode", "arch=compute_86,code=sm_86",  # Ampere
    ]


def _kernel_sources():
    extra = os.environ.get("BUILD_KERNELS", "")
    default = [
        "deploy/kernels/bindings.cpp",
        "deploy/kernels/gemm.cu",
        "deploy/kernels/quant.cu",
        "deploy/kernels/flashinfer.cu",
    ]
    return extra.split() + default if extra else default


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extra_pip_flags = os.environ.get("BUILD_ARGS", "").split() or None

    # Always try to install fast-hadamard-transform
    _install_fast_hadamard(extra_pip_flags)

    cuda = _cuda_available()

    if cuda:
        print("[setup] CUDA detected — building deploy._CUDA extension.")
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        _remove_unwanted_nvcc_flags()
        ext_modules = [
            CUDAExtension(
                name="deploy._CUDA",
                sources=_kernel_sources(),
                include_dirs=[
                    os.path.join(setup_dir, "deploy/kernels/include"),
                ],
                extra_compile_args={
                    "cxx": [],
                    "nvcc": _cuda_arch_flags(),
                },
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
    else:
        print(
            "[setup] No CUDA detected — installing Python packages only.\n"
            "        The PyTorch / FP8 paths in deploy/ work without the CUDA extension.\n"
            "        To build with CUDA: ensure nvcc is on PATH or set CUDA_HOME."
        )
        ext_modules = []
        cmdclass = {}

    setup(
        name="flatquant",
        packages=["flatquant", "deploy"],
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
