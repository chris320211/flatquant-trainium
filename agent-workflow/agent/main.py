"""
FlatQuant Porting Agent — entry point.

Pipeline: arch → ref_reader → codegen → registration → validation; if validation
passes, nxdi_port generates `nxdi/*` scaffolding under outputs/<model>/ using
`.claude/skills/trainium-model-translation/SKILL.md` as the NxDI workflow reference.

Usage:
    python main.py mistralai/Mistral-7B-v0.1
    python main.py  # prompts interactively

Requires:
    ANTHROPIC_API_KEY in .env or environment
    ANTHROPIC_MODEL_PLANNING (optional, default claude-opus-4-6)
    ANTHROPIC_MODEL_CODEGEN  (optional, default claude-sonnet-4-20250514)
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the agent-workflow directory.
load_dotenv(Path(__file__).parents[1] / ".env")

# Put the agent package on the path so nodes/ can import state, tools, prompts.
sys.path.insert(0, str(Path(__file__).parent))

from graph import build_graph

def main() -> None:
    if len(sys.argv) > 1:
        model_target = " ".join(sys.argv[1:])
    else:
        model_target = input("Model to port FlatQuant to: ").strip()

    if not model_target:
        print("No model specified. Exiting.")
        sys.exit(1)

    print(f"\nStarting FlatQuant port for: {model_target}\n")

    app = build_graph()

    initial_state = {
        "model_name": model_target,
        "messages": [],
    }

    final_state = app.invoke(initial_state)

    print("\n=== Validation Result ===")
    result = final_state.get("validation_result", {})
    print(f"Passed: {result.get('passed', False)}")

    if result.get("syntax_errors"):
        print("\nSyntax errors:")
        for fname, err in result["syntax_errors"].items():
            print(f"  {fname}: {err}")

    if result.get("import_errors"):
        print("\nImport errors:")
        for fname, err in result["import_errors"].items():
            print(f"  {fname}: {err}")

    if result.get("signature_errors"):
        print("\nSignature errors:")
        for cls, err in result["signature_errors"].items():
            print(f"  {cls}: {err}")

    print("\n=== Files Written (validation) ===")
    for fname, fpath in result.get("written_files", {}).items():
        print(f"  {fname} → {fpath}")

    vr_passed = (final_state.get("validation_result") or {}).get("passed")
    nxdi = final_state.get("nxdi_result") or {}
    print("\n=== NxDI porting ===")
    if not vr_passed:
        print("  Not run (validation did not pass).")
    elif nxdi.get("skipped"):
        print(f"  Skipped: {nxdi.get('reason', 'unknown')}")
    elif nxdi.get("written_files"):
        print("  Scaffolding written under outputs/.../nxdi/:")
        for fname, fpath in nxdi["written_files"].items():
            print(f"  {fname} → {fpath}")
    else:
        print("  (no nxdi artifacts recorded)")

    # Print the LLM validation summary from the message log.
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict) and "[validation_node]" in msg.get("content", ""):
            summary = msg["content"].replace("[validation_node] ", "", 1)
            print(f"\n=== Agent Summary ===\n{summary}")
            break


if __name__ == "__main__":
    main()
