"""
Skill Phase 2 prerequisite (trainium-model-translation SKILL.md):

Before launching block-translation work, copy `scripts/block_testing_utils.py` into
`tests/` and ensure `nxdi/` and `nxdi/blocks/` are Python packages.
"""

import re
from typing import Any

from skill_loader import load_block_testing_utils
from state import AgentState
from tools import write_output_files


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_skill_setup_node(state: AgentState) -> dict[str, Any]:
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    validation = state.get("validation_result") or {}
    if not validation.get("passed"):
        print("[trainium_skill_setup] Skipped (validation did not pass).", flush=True)
        return {
            "trainium_skill_setup_result": {"skipped": True},
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_skill_setup] Skipped."}],
        }

    print("[trainium_skill_setup] Copying block_testing_utils + nxdi package inits ...", flush=True)
    utils = load_block_testing_utils()
    files = {
        "tests/block_testing_utils.py": utils,
        "tests/__init__.py": '"""Test package for NxDI block tests (skill Phase 2)."""\n',
        "nxdi/__init__.py": '"""Generated NxDI port package."""\n',
        "nxdi/blocks/__init__.py": '"""Translated Neuron blocks (skill Phase 2)."""\n',
    }
    written = write_output_files.invoke({"model_name": model_name, "files": files})
    merged = {**state.get("generated_files", {}), **files}

    return {
        "generated_files": merged,
        "trainium_skill_setup_result": {
            "skipped": False,
            "written_files": written,
            "note": "Per SKILL.md: block_testing_utils copied before Phase 2 translators.",
        },
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_skill_setup] slug={slug} wrote={list(written.keys())}",
            }
        ],
    }
