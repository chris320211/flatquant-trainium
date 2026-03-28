"""
codegen node — LLM agent.

Takes the arch schema + ref patterns and generates all model-specific
FlatQuant source files: wrappers, calibration script, quant config, deploy model.
"""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from llm import anthropic_text, get_codegen_llm
from prompts import CODEGEN_PROMPT
from state import AgentState


def _model_slug(model_name: str) -> str:
    """Convert 'mistralai/Mistral-7B-v0.1' → 'mistral_7b_v0_1'."""
    base = model_name.split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")
    return slug


def codegen_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: generate FlatQuant source files for the target model.

    Reads arch fields + ref_patterns from state.
    Writes generated_files dict.
    """
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    model_type: str = state.get("model_type", "unknown")
    model_config: dict = state.get("model_config", {})
    linears: dict = state.get("linears", {})
    modeling_source: str = state.get("modeling_source", "")
    has_moe: bool = state.get("has_moe", False)
    ref_patterns: dict = state.get("ref_patterns", {})

    print(f"[codegen] Generating FlatQuant files for {model_name} (slug={slug}) ...")

    # Summarise linears to avoid hitting context limits.
    # Group by layer index prefix to show the pattern concisely.
    linears_summary = _summarise_linears(linears)

    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"model_type: {model_type}\n"
        f"has_moe: {has_moe}\n\n"
        f"model_config (selected fields):\n"
        f"{json.dumps(_selected_config_fields(model_config), indent=2)}\n\n"
        f"linears (layer name → shape):\n"
        f"{json.dumps(linears_summary, indent=2)}\n\n"
        f"ref_patterns:\n"
        f"{json.dumps(ref_patterns, indent=2)}\n\n"
        f"modeling_source (first 8000 chars):\n"
        f"{modeling_source[:8000]}\n\n"
        "Generate all four files as a JSON object {filename: source_code}."
    )

    llm = get_codegen_llm()

    messages = [
        SystemMessage(content=CODEGEN_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw_text: str = anthropic_text(response)

    generated_files = _parse_json_response(raw_text)

    print(f"[codegen] Generated {len(generated_files)} files: {list(generated_files.keys())}")

    return {
        "generated_files": generated_files,
        "messages": state.get("messages", []) + [
            {
                "role": "assistant",
                "content": f"[codegen_node] Generated files: {list(generated_files.keys())}",
            }
        ],
    }


def _selected_config_fields(config: dict) -> dict:
    """Return only the fields relevant for FlatQuant codegen."""
    keys = [
        "model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "max_position_embeddings", "vocab_size", "rms_norm_eps",
        "hidden_act", "rope_theta", "attention_bias",
    ]
    return {k: config[k] for k in keys if k in config}


def _summarise_linears(linears: dict) -> dict:
    """
    Deduplicate the linear layer list to show one representative per block.

    For a 32-layer Llama model this collapses 256 entries down to ~8 patterns.
    """
    seen_patterns: dict[str, str] = {}
    for name, shape in linears.items():
        # Strip layer index: "model.layers.7.mlp.gate_proj" → "model.layers.N.mlp.gate_proj"
        pattern = re.sub(r"\.\d+\.", ".N.", name)
        if pattern not in seen_patterns:
            seen_patterns[pattern] = f"{shape['in_features']} → {shape['out_features']}"
    return seen_patterns


def _parse_json_response(raw_text: str) -> dict:
    """Extract a JSON dict from an LLM response, stripping markdown fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # If the LLM returned multiple fenced code blocks, try to extract them.
    files = {}
    pattern = r"###?\s*`?([^`\n]+\.py)`?\s*\n```python\n(.*?)```"
    for match in re.finditer(pattern, raw_text, re.DOTALL):
        filename = match.group(1).strip()
        code = match.group(2).strip()
        files[filename] = code

    if files:
        return files

    # Last resort: store the raw text under a placeholder name.
    return {"_raw_codegen_output.txt": raw_text}
