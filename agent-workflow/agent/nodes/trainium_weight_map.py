"""Phase 4: HF → Neuron weight mapping stubs."""

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_codegen_llm
from prompts import TRAINIUM_WEIGHT_PROMPT
from skill_loader import excerpt_weight_mapping, load_skill_markdown
from state import AgentState
from tools import write_output_files

from nodes.trainium_io import parse_llm_json_object


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_weight_map_node(state: AgentState) -> dict[str, Any]:
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    print(f"[trainium_weight_map] Phase 4 weight mapping for {model_name} ...", flush=True)

    generated_files: dict = state.get("generated_files", {})
    plan = state.get("trainium_plan") or {}
    neuron_key = f"nxdi/neuron_{slug}_nxdi.py"
    neuron_src = generated_files.get(neuron_key, "")[:6000]

    system_prompt = TRAINIUM_WEIGHT_PROMPT.replace("{slug}", slug)
    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n\n"
        f"--- Phase 1 plan (truncated) ---\n{json.dumps(plan, indent=2)[:8000]}\n\n"
        f"--- {neuron_key} (prefix) ---\n{neuron_src or '(file not in generated_files yet)'}\n\n"
        f"--- reference/weight_mapping.md (excerpt) ---\n{excerpt_weight_mapping()}\n\n"
        f"--- SKILL.md (truncated) ---\n{load_skill_markdown()[:6000]}\n"
    )

    llm = get_codegen_llm()
    t0 = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    )
    print(f"[trainium_weight_map] LLM {time.perf_counter() - t0:.1f}s", flush=True)
    raw = anthropic_text(response)
    parsed = parse_llm_json_object(raw)

    out: dict[str, str] = {}
    for k, v in parsed.items():
        if k in ("parse_error", "raw_excerpt"):
            continue
        if not isinstance(v, str):
            continue
        key = str(k)
        if not key.startswith("nxdi/"):
            key = f"nxdi/{key.lstrip('/')}"
        out[key] = v

    if not out and parsed.get("parse_error"):
        out["nxdi/_phase4_parse_error.txt"] = str(parsed.get("raw_excerpt", raw))[:8000]

    written = write_output_files.invoke({"model_name": model_name, "files": out})
    merged = {**generated_files, **out}

    return {
        "generated_files": merged,
        "trainium_weight_result": {
            "skipped": False,
            "written_files": written,
            "filenames": list(out.keys()),
        },
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_weight_map] {list(out.keys())}",
            }
        ],
    }
