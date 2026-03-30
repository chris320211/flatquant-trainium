"""Phase 1: NxDI translation plan (trainium-model-translation skill)."""

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_planning_llm
from prompts import TRAINIUM_PLAN_PROMPT
from skill_loader import is_likely_vlm, load_skill_markdown, excerpt_vlm
from state import AgentState
from tools import write_output_files

from nodes.trainium_io import parse_llm_json_object


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_plan_node(state: AgentState) -> dict[str, Any]:
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    validation = state.get("validation_result") or {}
    if not validation.get("passed"):
        print("[trainium_plan] Skipped (validation did not pass).", flush=True)
        return {
            "trainium_plan": {"skipped": True, "reason": "validation_failed"},
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_plan] Skipped."}],
        }

    print(f"[trainium_plan] Phase 1 plan for {model_name} ...", flush=True)
    generated_files: dict = state.get("generated_files", {})
    modeling_key = f"modeling_{slug}.py"
    modeling_src = generated_files.get(modeling_key, "")
    modeling_source = state.get("modeling_source", "")[:12_000]
    model_config = state.get("model_config") or {}

    skill = load_skill_markdown()
    vlm_extra = ""
    if is_likely_vlm(state.get("modeling_source", ""), model_config):
        vlm_extra = "\n\n--- reference/vlm_translation.md (excerpt) ---\n" + excerpt_vlm()

    system_prompt = TRAINIUM_PLAN_PROMPT.replace("{slug}", slug)
    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"model_type: {state.get('model_type', 'unknown')}\n"
        f"has_moe: {state.get('has_moe', False)}\n\n"
        f"model_config (JSON):\n{json.dumps(model_config, indent=2)[:8000]}\n\n"
        f"--- modeling source (truncated) ---\n{modeling_source}\n\n"
        f"--- {modeling_key} (first 6000 chars) ---\n{modeling_src[:6000]}\n\n"
        f"--- Generated filenames ---\n{json.dumps(sorted(generated_files.keys()), indent=2)}\n\n"
        f"--- SKILL.md ---\n{skill}\n"
        f"{vlm_extra}"
    )

    llm = get_planning_llm()
    t0 = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    )
    print(f"[trainium_plan] LLM {time.perf_counter() - t0:.1f}s", flush=True)
    raw = anthropic_text(response)
    plan = parse_llm_json_object(raw)

    plan_files = {f"nxdi/phase1_plan.json": json.dumps(plan, indent=2)}
    written = write_output_files.invoke({"model_name": model_name, "files": plan_files})

    return {
        "trainium_plan": plan,
        "generated_files": {**generated_files, **plan_files},
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_plan] plan keys={list(plan.keys())} wrote={list(written.keys())}",
            }
        ],
    }
