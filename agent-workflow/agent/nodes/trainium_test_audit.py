"""
Skill Phase 2 — Auditing Subagent Test Files (Anti-Cheat Check).

See trainium-model-translation/SKILL.md after Phase 2 bullet list.
"""

import re
from typing import Any

from state import AgentState


def _audit_test_source(path: str, source: str) -> list[str]:
    flags: list[str] = []
    lower = source.lower()
    if re.search(r"^\s*from\s+pytorch_block\s+import", source, re.MULTILINE):
        flags.append(f"{path}: imports from pytorch_block (skill red flag)")
    if re.search(r"^\s*import\s+pytorch_block\b", source, re.MULTILINE):
        flags.append(f"{path}: imports pytorch_block module (skill red flag)")
    if "pytorch_block.py" in source and "do not" not in lower[:200]:
        # docstrings may mention the filename; keep heuristic light
        if re.search(r"['\"]pytorch_block\.py['\"]", source):
            flags.append(f"{path}: references pytorch_block.py as a file to use")
    return flags


def trainium_test_audit_node(state: AgentState) -> dict[str, Any]:
    generated = state.get("generated_files") or {}
    flags: list[str] = []
    checked: list[str] = []

    for path in sorted(generated.keys()):
        if path.endswith("pytorch_block.py") or path.endswith("/pytorch_block.py"):
            flags.append(f"Forbidden artifact present: {path} (skill: do not create pytorch_block.py)")
        if not (path.startswith("tests/") and path.endswith(".py") and "test" in path.lower()):
            continue
        src = generated[path]
        if not isinstance(src, str):
            continue
        checked.append(path)
        flags.extend(_audit_test_source(path, src))

    passed = len(flags) == 0
    report = {
        "passed": passed,
        "flags": flags,
        "files_checked": checked,
    }
    print(
        f"[trainium_test_audit] passed={passed} flags={len(flags)} checked={len(checked)}",
        flush=True,
    )
    return {
        "trainium_test_audit": report,
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_test_audit] passed={passed} {flags[:5]}",
            }
        ],
    }
