"""
nxdi_port node — LLM-assisted NxDI scaffolding after FlatQuant validation passes.

Loads `.claude/skills/trainium-model-translation/SKILL.md` (standard skill layout)
from the repo and generates `nxdi/*` files under the model output directory.
"""

import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import anthropic_text, get_codegen_llm
from prompts import NXDI_PORT_PROMPT
from state import AgentState
from tools import write_output_files

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILL_DIR = _REPO_ROOT / ".claude" / "skills" / "trainium-model-translation"
_SKILL_MD = _SKILL_DIR / "SKILL.md"
_LEGACY_FLAT_SKILL = _REPO_ROOT / ".claude" / "skills" / "trainium-model-translation.md"


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")
    return slug


def _load_trainium_skill_text() -> str:
    for path in (_SKILL_MD, _LEGACY_FLAT_SKILL):
        try:
            body = path.read_text()
            break
        except OSError:
            pass
    else:
        return (
            f"(Skill not found. Expected {_SKILL_MD} "
            f"or legacy {_LEGACY_FLAT_SKILL}.)"
        )

    extra = (
        "\n\n--- Bundled resources (same skill directory; read on disk when needed) ---\n"
        "- reference/vlm_translation.md\n"
        "- reference/scaffolding_integration.md\n"
        "- reference/weight_mapping.md\n"
        "- scripts/block_testing_utils.py\n"
    )
    return body + extra


def _parse_nxdi_json(raw_text: str) -> dict[str, str]:
    """Extract {path: content} from LLM response."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items()}
    except json.JSONDecodeError:
        pass
    return {"nxdi/_raw_nxdi_output.txt": raw_text}


def nxdi_port_node(state: AgentState) -> dict[str, Any]:
    """
    Generate NxDI porting scaffolding under outputs/<slug>/nxdi/.

    Skips if validation did not pass (graph should not route here in that case).
    """
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    validation = state.get("validation_result") or {}

    if not validation.get("passed"):
        print("[nxdi_port] Skipped (validation did not pass).", flush=True)
        return {
            "nxdi_result": {
                "skipped": True,
                "reason": "validation_failed",
            },
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": "[nxdi_port_node] Skipped: validation failed.",
                }
            ],
        }

    print(f"[nxdi_port] Generating NxDI scaffolding for {model_name} (slug={slug}) ...", flush=True)

    generated_files: dict = state.get("generated_files", {})
    modeling_key = f"modeling_{slug}.py"
    modeling_src = generated_files.get(modeling_key, "")
    utils_key = f"{slug}_utils.py"
    utils_src = generated_files.get(utils_key, "")

    skill_text = _load_trainium_skill_text()
    system_prompt = NXDI_PORT_PROMPT.replace("{slug}", slug)

    user_message = (
        f"model_name: {model_name}\n"
        f"slug: {slug}\n"
        f"model_type: {state.get('model_type', 'unknown')}\n"
        f"has_moe: {state.get('has_moe', False)}\n\n"
        f"Generated FlatQuant filenames (for cross-reference):\n"
        f"{json.dumps(sorted(generated_files.keys()), indent=2)}\n\n"
        f"--- {modeling_key} (first 6000 chars) ---\n{modeling_src[:6000]}\n\n"
        f"--- {utils_key} (first 4000 chars) ---\n{utils_src[:4000]}\n\n"
        f"--- trainium-model-translation/SKILL.md (+ bundled paths) ---\n{skill_text}\n"
    )

    llm = get_codegen_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    print(
        "[nxdi_port] Calling codegen LLM for NxDI scaffolding (often 1-4+ minutes) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    print(f"[nxdi_port] LLM round-trip took {time.perf_counter() - t0:.1f}s", flush=True)
    raw_text = anthropic_text(response)

    nxdi_files = _parse_nxdi_json(raw_text)
    for k in list(nxdi_files.keys()):
        if not k.startswith("nxdi/"):
            nxdi_files[f"nxdi/{k}"] = nxdi_files.pop(k)

    written = write_output_files.invoke({"model_name": model_name, "files": nxdi_files})

    print(f"[nxdi_port] Wrote {len(written)} NxDI artifact(s): {list(written.keys())}", flush=True)

    merged = {**generated_files, **nxdi_files}

    return {
        "generated_files": merged,
        "nxdi_result": {
            "skipped": False,
            "written_files": written,
            "filenames": list(nxdi_files.keys()),
        },
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[nxdi_port_node] NxDI scaffolding: {list(nxdi_files.keys())}",
            }
        ],
    }
