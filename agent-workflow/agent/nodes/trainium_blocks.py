"""Phase 2: NxDI block files + tests (LLM)."""

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_codegen_llm
from prompts import TRAINIUM_BLOCKS_PROMPT
from skill_loader import load_block_testing_utils, load_skill_markdown
from state import AgentState
from tools import write_output_files

from nodes.trainium_io import parse_llm_json_object


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_blocks_node(state: AgentState) -> dict[str, Any]:
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    plan = state.get("trainium_plan") or {}
    if plan.get("skipped"):
        print("[trainium_blocks] Skipped (no plan).", flush=True)
        return {
            "trainium_block_files": {},
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_blocks] Skipped."}],
        }

    print(f"[trainium_blocks] Phase 2 blocks for {model_name} ...", flush=True)
    generated_files: dict = state.get("generated_files", {})

    system_prompt = TRAINIUM_BLOCKS_PROMPT.replace("{slug}", slug)
    modeling_path = state.get("modeling_source_path") or "use transformers installed class sources"
    btu = load_block_testing_utils()
    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"original_modeling_source_path (for test imports — skill anti-cheat): {modeling_path}\n\n"
        f"--- Phase 1 plan (JSON) ---\n{json.dumps(plan, indent=2)[:24_000]}\n\n"
        f"--- scripts/block_testing_utils.py (FULL — you MUST call test_block_correctness) ---\n"
        f"{btu}\n\n"
        f"--- SKILL.md (Phase 2 section + context) ---\n{load_skill_markdown()[:16_000]}\n"
    )

    llm = get_codegen_llm()
    t0 = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    )
    print(f"[trainium_blocks] LLM {time.perf_counter() - t0:.1f}s", flush=True)
    raw = anthropic_text(response)
    files = parse_llm_json_object(raw)

    out: dict[str, str] = {}
    for k, v in files.items():
        if k in ("parse_error", "raw_excerpt"):
            continue
        if not isinstance(v, str):
            v = json.dumps(v)
        ks = str(k)
        if ks.startswith("nxdi/") or ks.startswith("tests/"):
            out[ks] = v

    if not out and files.get("parse_error"):
        out["nxdi/blocks/_phase2_parse_error.txt"] = str(files.get("raw_excerpt", raw))[
            :8000
        ]

    written = write_output_files.invoke({"model_name": model_name, "files": out})

    return {
        "trainium_block_files": out,
        "generated_files": {**generated_files, **out},
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_blocks] wrote {list(written.keys())}",
            }
        ],
    }
