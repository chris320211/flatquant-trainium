"""
arch node — deterministic (no LLM call).

Extracts the target model's architecture schema from HuggingFace without
downloading weights, using torch.device("meta") and transformers AutoConfig.
"""

import inspect
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from state import AgentState


def arch_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: populate architecture fields in state.

    Reads model_name from state, writes:
      model_config, linears, modeling_source, modeling_source_path,
      model_type, has_moe
    """
    model_name: str = state["model_name"]
    print(f"[arch] Loading config for {model_name} ...")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Instantiate on meta device — zero memory, zero weights.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    linears: dict = {}
    has_moe: bool = False

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            linears[name] = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
            }
        class_name = type(mod).__name__.lower()
        if any(kw in class_name for kw in ("moe", "expert", "router", "mixtral", "sparsemoe")):
            has_moe = True

    # Find the modeling source file.
    model_class = type(model)
    try:
        source_file = inspect.getfile(model_class)
        with open(source_file) as f:
            modeling_source = f.read()
    except (TypeError, OSError):
        source_file = ""
        modeling_source = ""

    print(
        f"[arch] Found {len(linears)} linear layers, "
        f"model_type={config.model_type}, has_moe={has_moe}"
    )

    return {
        "model_config": config.to_dict(),
        "linears": linears,
        "modeling_source": modeling_source,
        "modeling_source_path": source_file,
        "model_type": config.model_type,
        "has_moe": has_moe,
        "messages": state.get("messages", []) + [
            {
                "role": "system",
                "content": (
                    f"[arch_node] Extracted architecture for {model_name}: "
                    f"{len(linears)} linears, model_type={config.model_type}, has_moe={has_moe}"
                ),
            }
        ],
    }
