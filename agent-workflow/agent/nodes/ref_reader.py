"""
ref_reader node — LLM agent.

Reads the FlatQuant reference implementations and extracts the patterns
that codegen needs: wrapper structure, calibration flow, kernel imports, etc.
"""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_planning_llm
from prompts import REF_READER_PROMPT
from state import AgentState
from tools import read_flatquant_files

_ALWAYS_READ = [
    "llama_utils",
    "flat_linear",
    "trans_utils",
    "quant_utils",
    "train_utils",
    "main",
    "deploy_modeling_llama",
    "deploy_nn_quantization",
    "online_trans",
    "kron_matmul_pytorch",
    "block_matmul_pytorch",
]
_MOE_EXTRA = ["deepseekv3_utils"]


def ref_reader_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: populate ref_patterns in state.

    Uses an LLM with tool-calling to read FlatQuant reference files and
    return a structured JSON summary of patterns.
    """
    has_moe: bool = state.get("has_moe", False)
    model_type: str = state.get("model_type", "unknown")

    keys_to_read = list(_ALWAYS_READ)
    if has_moe:
        keys_to_read += _MOE_EXTRA

    print(f"[ref_reader] Reading {len(keys_to_read)} reference files ...")

    # Pre-read all files deterministically — avoids tool-call round-trips for
    # file I/O and lets the LLM focus on pattern extraction.
    file_contents: dict = read_flatquant_files.invoke({"file_keys": keys_to_read})

    # Build a single message with all file contents embedded.
    files_text = "\n\n".join(
        f"=== FILE: {key} ===\n{content}" for key, content in file_contents.items()
        if not content.startswith("ERROR")
    )

    user_message = (
        f"Target model type: {model_type}\n"
        f"Has MoE routing: {has_moe}\n\n"
        f"Here are the FlatQuant reference files:\n\n{files_text}\n\n"
        "Now extract the patterns and return the JSON summary as specified in your instructions."
    )

    llm = get_planning_llm()

    messages = [
        SystemMessage(content=REF_READER_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw_text: str = anthropic_text(response)

    # Parse the JSON from the response (strip markdown fences if present).
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        json_text = re.sub(r"^```[a-z]*\n?", "", json_text)
        json_text = re.sub(r"\n?```$", "", json_text.strip())

    try:
        ref_patterns = json.loads(json_text)
    except json.JSONDecodeError:
        # Fall back: store raw text so codegen can still use it as prose.
        ref_patterns = {"raw": raw_text}

    print("[ref_reader] Pattern extraction complete.")

    return {
        "ref_patterns": ref_patterns,
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": f"[ref_reader_node] Extracted patterns for {model_type}."}
        ],
    }
