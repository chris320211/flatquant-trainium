"""
Tool functions used by the LangGraph nodes.

These are plain Python callables — not LangChain tools — so they can be
called directly inside node functions without @tool decoration.
"""

import inspect
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from langchain_core.tools import tool
from transformers import AutoConfig, AutoModelForCausalLM

# Absolute path to the FlatQuant reference codebase (local clone).
REPO_ROOT = Path(__file__).resolve().parents[2]
FLATQUANT_ROOT = REPO_ROOT / "FlatQuantBundled"

# Reference files always passed to ref_reader.
FLATQUANT_REFERENCE_FILES = {
    "llama_utils": FLATQUANT_ROOT / "flatquant/model_tools/llama_utils.py",
    "deepseekv3_utils": FLATQUANT_ROOT / "flatquant/model_tools/deepseekv3_utils.py",
    "qwen_utils": FLATQUANT_ROOT / "flatquant/model_tools/qwen_utils.py",
    "llama31_utils": FLATQUANT_ROOT / "flatquant/model_tools/llama31_utils.py",
    "train_utils": FLATQUANT_ROOT / "flatquant/train_utils.py",
    "main": FLATQUANT_ROOT / "main.py",
    "deploy_modeling_llama": FLATQUANT_ROOT / "deploy/transformers/modeling_llama.py",
    "deploy_nn_quantization": FLATQUANT_ROOT / "deploy/nn/quantization.py",
    "online_trans": FLATQUANT_ROOT / "deploy/functional/online_trans.py",
    "kron_matmul_pytorch": FLATQUANT_ROOT / "deploy/kernels/pytorch/kron_matmul_pytorch.py",
    "block_matmul_pytorch": FLATQUANT_ROOT / "deploy/kernels/pytorch/block_matmul_pytorch.py",
    "flat_linear": FLATQUANT_ROOT / "flatquant/flat_linear.py",
    "trans_utils": FLATQUANT_ROOT / "flatquant/trans_utils.py",
    "quant_utils": FLATQUANT_ROOT / "flatquant/quant_utils.py",
}


@tool
def read_arch(model_name: str) -> dict:
    """
    Extract the architecture schema of a HuggingFace model without downloading weights.

    Returns a dict with:
      - model_type: str
      - model_config: dict (config fields)
      - linears: dict {layer_name: {in_features, out_features}}
      - has_moe: bool
      - modeling_source: str (full text of the modeling .py)
      - modeling_source_path: str
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Instantiate on meta device — no weights, no memory.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    linears = {}
    has_moe = False
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            linears[name] = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
            }
        # Heuristic: detect MoE by common class/attribute names.
        class_name = type(mod).__name__.lower()
        if any(kw in class_name for kw in ("moe", "expert", "router", "mixtral")):
            has_moe = True

    # Get the modeling source file.
    model_class = type(model)
    try:
        source_file = inspect.getfile(model_class)
        with open(source_file) as f:
            modeling_source = f.read()
    except (TypeError, OSError):
        source_file = ""
        modeling_source = ""

    return {
        "model_type": config.model_type,
        "model_config": config.to_dict(),
        "linears": linears,
        "has_moe": has_moe,
        "modeling_source": modeling_source,
        "modeling_source_path": source_file,
    }


@tool
def read_flatquant_files(file_keys: list[str]) -> dict:
    """
    Read one or more FlatQuant reference files by key.

    Valid keys: llama_utils, deepseekv3_utils, qwen_utils, llama31_utils,
    train_utils, main, deploy_modeling_llama, deploy_nn_quantization, online_trans,
    kron_matmul_pytorch, block_matmul_pytorch, flat_linear, trans_utils,
    quant_utils.

    Returns {key: file_contents_string}.
    """
    result = {}
    for key in file_keys:
        path = FLATQUANT_REFERENCE_FILES.get(key)
        if path is None:
            result[key] = f"ERROR: unknown key '{key}'"
        elif not path.exists():
            result[key] = f"ERROR: file not found at {path}"
        else:
            result[key] = path.read_text()
    return result


@tool
def write_output_files(model_name: str, files: dict) -> dict:
    """
    Write generated source files to outputs/<model_name>/ under the repo root.

    `files` is a dict of {filename: source_code_string}.
    Returns {filename: absolute_path_written}.
    """
    slug = model_name.replace("/", "__").replace(" ", "_")
    out_dir = REPO_ROOT / "agent-workflow" / "outputs" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for filename, content in files.items():
        dest = out_dir / filename
        dest.write_text(content)
        written[filename] = str(dest)

    return written


# Convenience: list of all tool objects for binding to LLMs.
ALL_TOOLS = [read_arch, read_flatquant_files, write_output_files]
