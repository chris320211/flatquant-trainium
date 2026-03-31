"""Phase 4.5: Weight mapping test generation (LLM-assisted).

Generates tests to validate state_dict key mapping and shape compatibility.
These tests verify that HF→Neuron weight mapping is correct without requiring
actual model weights.
"""

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_codegen_llm
from prompts import TRAINIUM_WEIGHT_TESTS_PROMPT
from state import AgentState
from tools import write_output_files

from nodes.trainium_io import parse_llm_json_object


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_weight_tests_node(state: AgentState) -> dict[str, Any]:
    """Generate weight mapping validation tests."""
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)

    weight_result = state.get("trainium_weight_result") or {}
    if weight_result.get("skipped"):
        print("[trainium_weight_tests] Skipped (weight mapping skipped).", flush=True)
        return {
            "trainium_weight_tests_result": {
                "skipped": True,
                "reason": "weight_mapping_skipped",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_weight_tests] Skipped."}],
        }

    print(
        f"[trainium_weight_tests] Generating weight mapping tests for {model_name} ...",
        flush=True,
    )

    generated_files: dict = state.get("generated_files", {})
    model_config = state.get("model_config") or {}
    linears: dict = state.get("linears", {})

    # Extract weight mapping source
    weight_mapping_key = next(
        (k for k in generated_files.keys() if "convert_weights" in k or "weight_map" in k),
        None,
    )
    weight_mapping_src = generated_files.get(weight_mapping_key, "")[:8000]

    system_prompt = TRAINIUM_WEIGHT_TESTS_PROMPT.replace("{slug}", slug)

    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"model_type: {state.get('model_type', 'unknown')}\n"
        f"has_moe: {state.get('has_moe', False)}\n\n"
        f"model_config (JSON):\n{json.dumps(model_config, indent=2)[:4000]}\n\n"
        f"linears (sample of layer shapes):\n"
        f"{json.dumps(dict(list(linears.items())[:10]), indent=2)}\n\n"
        f"--- Weight mapping implementation (first 8000 chars) ---\n"
        f"{weight_mapping_src}\n"
    )

    llm = get_codegen_llm()
    t0 = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    )
    print(f"[trainium_weight_tests] LLM {time.perf_counter() - t0:.1f}s", flush=True)
    raw = anthropic_text(response)
    files = parse_llm_json_object(raw)

    out: dict[str, str] = {}
    for k, v in files.items():
        if k in ("parse_error", "raw_excerpt"):
            continue
        if not isinstance(v, str):
            v = json.dumps(v)
        ks = str(k)
        if ks.startswith("tests/"):
            out[ks] = v

    if not out and files.get("parse_error"):
        out["tests/_weight_tests_parse_error.txt"] = str(
            files.get("raw_excerpt", raw)
        )[:8000]

    written = write_output_files.invoke({"model_name": model_name, "files": out})

    return {
        "trainium_weight_tests_result": {
            "written_files": written,
            "files_count": len(out),
        },
        "generated_files": {**generated_files, **out},
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_weight_tests] wrote {len(written)} file(s)",
            }
        ],
    }
