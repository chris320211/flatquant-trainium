"""Phase 3: NxDI scaffolding / integration."""

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_codegen_llm
from prompts import TRAINIUM_INTEGRATE_PROMPT
from skill_loader import excerpt_scaffolding, load_skill_markdown
from state import AgentState
from tools import write_output_files

from nodes.trainium_io import parse_llm_json_object


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_integrate_node(state: AgentState) -> dict[str, Any]:
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    print(f"[trainium_integrate] Phase 3 integrate for {model_name} ...", flush=True)

    generated_files: dict = state.get("generated_files", {})
    plan = state.get("trainium_plan") or {}
    block_files: dict = state.get("trainium_block_files") or {}

    block_preview = ""
    for path, src in list(block_files.items())[:12]:
        block_preview += f"\n--- {path} (first 3500 chars) ---\n{src[:3500]}\n"

    modeling_key = f"modeling_{slug}.py"
    modeling_src = generated_files.get(modeling_key, "")[:5000]

    system_prompt = TRAINIUM_INTEGRATE_PROMPT.replace("{slug}", slug)
    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"model_type: {state.get('model_type', 'unknown')}\n\n"
        f"--- Phase 1 plan ---\n{json.dumps(plan, indent=2)[:12_000]}\n\n"
        f"--- Phase 2 block files preview ---\n{block_preview or '(none)'}\n\n"
        f"--- {modeling_key} (prefix) ---\n{modeling_src}\n\n"
        f"--- reference/scaffolding_integration.md (excerpt) ---\n{excerpt_scaffolding()}\n\n"
        f"--- SKILL.md (truncated) ---\n{load_skill_markdown()[:8000]}\n"
    )

    llm = get_codegen_llm()
    t0 = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    )
    print(f"[trainium_integrate] LLM {time.perf_counter() - t0:.1f}s", flush=True)
    raw = anthropic_text(response)
    parsed = parse_llm_json_object(raw)

    nxdi_files: dict[str, str] = {}
    for k, v in parsed.items():
        if k in ("parse_error", "raw_excerpt"):
            continue
        if not isinstance(v, str):
            continue
        key = str(k)
        if not key.startswith("nxdi/"):
            key = f"nxdi/{key.lstrip('/')}"
        nxdi_files[key] = v

    if not nxdi_files and parsed.get("parse_error"):
        nxdi_files["nxdi/_phase3_parse_error.txt"] = str(parsed.get("raw_excerpt", raw))[
            :8000
        ]

    written = write_output_files.invoke({"model_name": model_name, "files": nxdi_files})
    merged = {**generated_files, **nxdi_files}

    return {
        "generated_files": merged,
        "trainium_integrate_result": {
            "skipped": False,
            "written_files": written,
            "filenames": list(nxdi_files.keys()),
        },
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_integrate] {list(nxdi_files.keys())}",
            }
        ],
    }
