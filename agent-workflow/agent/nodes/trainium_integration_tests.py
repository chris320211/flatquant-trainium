"""Phase 3.5: Integration test generation (LLM-assisted).

Generates end-to-end tests for the full NxDI model forward pass.
These tests validate that the model can initialize and run inference.
"""

import json
import os
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_codegen_llm
from prompts import TRAINIUM_INTEGRATION_TESTS_PROMPT
from skill_loader import load_skill_markdown
from state import AgentState
from tools import write_output_files

from nodes.trainium_io import parse_llm_json_object


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_integration_tests_node(state: AgentState) -> dict[str, Any]:
    """Generate integration tests for the full NxDI model."""
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)

    # Check if test generation is disabled
    skip_tests = os.environ.get("TRAINIUM_SKIP_TEST_GENERATION", "").lower().strip() in ("1", "true", "yes")
    if skip_tests:
        print("[trainium_integration_tests] Skipped (TRAINIUM_SKIP_TEST_GENERATION=1).", flush=True)
        return {
            "trainium_integration_tests_result": {
                "skipped": True,
                "reason": "test_generation_disabled",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_integration_tests] Skipped."}],
        }

    weight_result = state.get("trainium_weight_result") or {}
    if weight_result.get("skipped"):
        print("[trainium_integration_tests] Skipped (weight mapping skipped).", flush=True)
        return {
            "trainium_integration_tests_result": {
                "skipped": True,
                "reason": "weight_mapping_skipped",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_integration_tests] Skipped."}],
        }

    print(
        f"[trainium_integration_tests] Generating integration tests for {model_name} ...",
        flush=True,
    )

    generated_files: dict = state.get("generated_files", {})
    model_config = state.get("model_config") or {}

    system_prompt = TRAINIUM_INTEGRATION_TESTS_PROMPT.replace("{slug}", slug)

    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"model_type: {state.get('model_type', 'unknown')}\n"
        f"has_moe: {state.get('has_moe', False)}\n\n"
        f"model_config (JSON):\n{json.dumps(model_config, indent=2)[:4000]}\n\n"
        f"--- Generated NxDI files (for reference) ---\n"
        f"{json.dumps([k for k in sorted(generated_files.keys()) if k.startswith('nxdi/')], indent=2)}\n"
    )

    llm = get_codegen_llm()
    t0 = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    )
    print(f"[trainium_integration_tests] LLM {time.perf_counter() - t0:.1f}s", flush=True)
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
        out["tests/_integration_tests_parse_error.txt"] = str(
            files.get("raw_excerpt", raw)
        )[:8000]

    written = write_output_files.invoke({"model_name": model_name, "files": out})

    return {
        "trainium_integration_tests_result": {
            "written_files": written,
            "files_count": len(out),
        },
        "generated_files": {**generated_files, **out},
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_integration_tests] wrote {len(written)} file(s)",
            }
        ],
    }
