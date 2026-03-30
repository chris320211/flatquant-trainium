"""
codegen node — LLM agent.

Takes the arch schema + ref patterns and generates all model-specific
FlatQuant source files: wrappers, calibration script, quant config, deploy model.
"""

import ast
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from llm import anthropic_text, get_codegen_llm
from prompts import CODEGEN_PROMPT
from state import AgentState

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLATQUANT_MAIN = _REPO_ROOT / "FlatQuantBundled" / "main.py"
_MAIN_SNIPPET_END_LINE = 40  # imports + main() through cali_flat_quant call
_DEPLOY_NN_INIT = _REPO_ROOT / "FlatQuantBundled" / "deploy" / "nn" / "__init__.py"
_DEPLOY_NN_QUANTIZATION = _REPO_ROOT / "FlatQuantBundled" / "deploy" / "nn" / "quantization.py"
_QUANTIZATION_SNIPPET_MAX_LINES = 70
_LLAMA_UTILS = _REPO_ROOT / "FlatQuantBundled" / "flatquant" / "model_tools" / "llama_utils.py"
_LLAMA_UTILS_ATTENTION_START = 111  # 1-based: class FlatQuantLlamaAttention through add_fq_trans
_LLAMA_UTILS_ATTENTION_END = 160


def _canonical_flatquant_main_snippet() -> str:
    """Verbatim top of FlatQuant main.py so codegen always sees real import paths."""
    try:
        text = _FLATQUANT_MAIN.read_text()
    except OSError:
        return f"(missing or unreadable: {_FLATQUANT_MAIN})"
    lines = text.splitlines()
    return "\n".join(lines[:_MAIN_SNIPPET_END_LINE])


def _canonical_deploy_quantization_snippet() -> str:
    """Verbatim deploy/nn exports + Quantizer implementation (ground truth for modeling imports)."""
    parts: list[str] = []
    for path, label in (
        (_DEPLOY_NN_INIT, "deploy/nn/__init__.py"),
        (_DEPLOY_NN_QUANTIZATION, "deploy/nn/quantization.py"),
    ):
        try:
            text = path.read_text()
        except OSError:
            parts.append(f"({label}: missing or unreadable: {path})")
            continue
        if path.name == "quantization.py":
            lines = text.splitlines()
            text = "\n".join(lines[:_QUANTIZATION_SNIPPET_MAX_LINES])
        parts.append(f"=== {label} ===\n{text.rstrip()}")
    return "\n\n".join(parts)


def _canonical_llama_utils_attention_snippet() -> str:
    """FlatQuantLlamaAttention.__init__ + add_fq_trans — use config.num_attention_heads for o_trans."""
    try:
        lines = _LLAMA_UTILS.read_text().splitlines()
    except OSError:
        return f"(missing or unreadable: {_LLAMA_UTILS})"
    i0 = max(0, _LLAMA_UTILS_ATTENTION_START - 1)
    i1 = min(len(lines), _LLAMA_UTILS_ATTENTION_END)
    return "\n".join(lines[i0:i1])


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

    print(f"[codegen] Generating FlatQuant files for {model_name} (slug={slug}) ...", flush=True)

    # Summarise linears to avoid hitting context limits.
    # Group by layer index prefix to show the pattern concisely.
    linears_summary = _summarise_linears(linears)
    forward_hints = _hint_forward_signatures(model_type, modeling_source)
    main_snippet = _canonical_flatquant_main_snippet()
    deploy_quant_snippet = _canonical_deploy_quantization_snippet()
    llama_attn_snippet = _canonical_llama_utils_attention_snippet()

    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"model_type: {model_type}\n"
        f"has_moe: {has_moe}\n\n"
        f"canonical_flatquant_main_snippet (verbatim FlatQuant/main.py — use these import paths "
        f"for calibrate_{slug}.py; cali_flat_quant lives in flatquant.train_utils):\n"
        f"```python\n{main_snippet}\n```\n\n"
        f"canonical_deploy_quantization_snippet (verbatim deploy/nn — activation quant is "
        f"`Quantizer` / `deploy.nn.Quantizer` only; do not import quantize_activation or other "
        f"invented names from deploy.nn.quantization in modeling_{slug}.py):\n"
        f"```python\n{deploy_quant_snippet}\n```\n\n"
        f"canonical_llama_utils_attention_snippet (FlatQuantLlamaAttention + add_fq_trans — "
        f"copy SingleTransMatrix(self.config.num_attention_heads) pattern for Llama; never self.num_heads):\n"
        f"```python\n{llama_attn_snippet}\n```\n\n"
        f"installed_forward_signatures (from the installed HuggingFace modeling file — "
        f"subclass forward() must include these parameter names + **kwargs):\n"
        f"{json.dumps(forward_hints, indent=2)}\n\n"
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

    print(
        "[codegen] Calling codegen LLM (Anthropic); no further logs until the response returns "
        "(often several minutes for four files).",
        flush=True,
    )
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    print(f"[codegen] LLM round-trip took {time.perf_counter() - t0:.1f}s", flush=True)
    raw_text: str = anthropic_text(response)

    generated_files = _parse_json_response(raw_text)

    print(f"[codegen] Generated {len(generated_files)} files: {list(generated_files.keys())}", flush=True)

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


def _extract_forward_signatures_from_source(source: str) -> dict[str, list[str]]:
    """Parse source and return {ClassName: [param_names]} for each forward() method."""
    sigs: dict[str, list[str]] = {}
    if not source:
        return sigs
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sigs

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                params = [arg.arg for arg in item.args.args]
                params.extend(a.arg for a in item.args.kwonlyargs)
                if item.args.vararg:
                    params.append("*" + item.args.vararg.arg)
                if item.args.kwarg:
                    params.append("**" + item.args.kwarg.arg)
                sigs[node.name] = params
    return sigs


def _hint_forward_signatures(model_type: str, modeling_source: str) -> dict[str, list[str]]:
    """
    Subset of forward() signatures from the installed HF modeling file for the codegen LLM.
    Filters by model_type title case (e.g. llama -> Llama) to keep context small.
    """
    all_sigs = _extract_forward_signatures_from_source(modeling_source)
    if not all_sigs:
        return {}
    prefix = model_type.title() if model_type else ""
    hints = {k: v for k, v in all_sigs.items() if prefix and prefix in k}
    if not hints:
        hints = dict(list(all_sigs.items())[:30])
    elif len(hints) > 35:
        priority = [
            n
            for n in hints
            if any(
                x in n
                for x in (
                    "Attention",
                    "MLP",
                    "DecoderLayer",
                    "ForCausalLM",
                    "Model",
                    "Rotary",
                )
            )
        ]
        if priority:
            hints = {k: hints[k] for k in priority[:35]}
    return hints


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
