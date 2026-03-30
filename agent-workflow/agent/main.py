"""
FlatQuant Porting Agent — entry point.

Pipeline: arch → ref_reader → codegen → registration → validation → **flatquant_calibrate**
  (runs only if FLATQUANT_CALIBRATE=smoke|full) → Trainium path → **trainium_compile_smoke**
  (only if TRAINIUM_COMPILE_CMD / TRAINIUM_SMOKE_CMD set).

  - TRAINIUM_SKILL_MODE=full (default): skill phases + verify.
  - TRAINIUM_SKILL_MODE=fast: nxdi_port only, then compile_smoke.

Env (see agent-workflow/agent/graph.py docstring for full list):
  FLATQUANT_CALIBRATE=smoke|full — auto-run calibrate_{slug}.py (needs HF_TOKEN, weights).
  FLATQUANT_CALIBRATE_MODEL — local path override for --model.
  TRAINIUM_COMPILE_CMD / TRAINIUM_SMOKE_CMD — shell commands from output dir (Neuron).
  TRAINIUM_RUN_BLOCK_TESTS=1, TRAINIUM_SKIP_VERIFY=1, TRAINIUM_SKIP_COMPILE_SMOKE=1

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

    cal = final_state.get("flatquant_calibrate_result") or {}
    print("\n=== FlatQuant calibration (optional) ===")
    if cal.get("skipped"):
        print(f"  Skipped: {cal.get('reason', 'unknown')}")
    else:
        print(f"  returncode={cal.get('returncode')} mode={cal.get('mode')}")
        if cal.get("stderr_tail"):
            print(f"  stderr (tail):\n{cal['stderr_tail'][-2000:]}")
        if cal.get("returncode") not in (0, None):
            print(
                "\n  *** PIPELINE NOTE: Calibration did not succeed (rc != 0). "
                "Static validation may still show \"passed\" — that only checks generated files, "
                "not calibration. Run calibrate_*.py manually with `python ... --help` flags. ***"
            )

    vr_passed = (final_state.get("validation_result") or {}).get("passed")
    _mode = os.environ.get("TRAINIUM_SKILL_MODE", "full").lower().strip()
    print("\n=== Trainium / NxDI ===")
    if not vr_passed:
        print("  Not run (validation did not pass).")
    elif _mode == "fast":
        nxdi = final_state.get("nxdi_result") or {}
        if nxdi.get("skipped"):
            print(f"  fast mode skipped: {nxdi.get('reason', 'unknown')}")
        elif nxdi.get("written_files"):
            print("  nxdi_port (fast) wrote:")
            for fname, fpath in nxdi["written_files"].items():
                print(f"    {fname} → {fpath}")
        else:
            print("  (no nxdi artifacts recorded)")
    else:
        plan = final_state.get("trainium_plan") or {}
        if plan.get("skipped"):
            print(f"  Plan skipped: {plan.get('reason')}")
        else:
            print(
                f"  Phase 1 plan keys: {list(plan.keys())[:12]}"
                f"{'...' if len(plan) > 12 else ''}"
            )
            notes = plan.get("_agent_plan_notes")
            if notes:
                print(f"  Phase 1 notes: {notes}")
        setup = final_state.get("trainium_skill_setup_result") or {}
        if setup.get("written_files"):
            print("  Skill setup (pre-Phase 2) wrote:")
            for fname, fpath in setup["written_files"].items():
                print(f"    {fname} → {fpath}")
        blocks = final_state.get("trainium_block_files") or {}
        print(f"  Phase 2 block files: {len(blocks)} path(s)")
        audit = final_state.get("trainium_test_audit") or {}
        if audit.get("files_checked") is not None:
            print(
                f"  Phase 2 test audit: passed={audit.get('passed')} "
                f"flags={len(audit.get('flags') or [])}"
            )
            for fl in (audit.get("flags") or [])[:5]:
                print(f"    ! {fl}")
        trep = final_state.get("trainium_test_report") or {}
        if trep.get("skipped"):
            print(f"  Phase 2 tests: skipped — {trep.get('reason', 'unknown')}")
        else:
            print(
                f"  Phase 2 tests: rc={trep.get('returncode')} "
                f"(TRAINIUM_RUN_BLOCK_TESTS)"
            )
        integ = final_state.get("trainium_integrate_result") or {}
        if integ.get("written_files"):
            print("  Phase 3 integrate wrote:")
            for fname, fpath in integ["written_files"].items():
                print(f"    {fname} → {fpath}")
        wres = final_state.get("trainium_weight_result") or {}
        if wres.get("written_files"):
            print("  Phase 4 weight map wrote:")
            for fname, fpath in wres["written_files"].items():
                print(f"    {fname} → {fpath}")
        ver = final_state.get("trainium_nxdi_verify") or {}
        if ver.get("skipped"):
            print(f"  Verify: skipped ({ver.get('reason')})")
        elif ver:
            print(
                f"  Verify: neuronx_distributed import OK={ver.get('neuron_pkg_import_ok')} "
                f"generated nxdi module exec OK={ver.get('neuron_module_exec_ok')}"
            )
            if ver.get("error"):
                print(f"    error: {ver['error']}")

    cps = final_state.get("trainium_compile_smoke_result") or {}
    if vr_passed:
        print("\n=== Neuron compile / smoke (optional) ===")
        if cps.get("skipped"):
            print(f"  Skipped: {cps.get('reason', 'unknown')}")
        else:
            if cps.get("compile"):
                c = cps["compile"]
                print(f"  compile rc={c.get('returncode')}")
                if c.get("stderr_tail"):
                    print(f"    stderr tail: {c['stderr_tail'][-1500:]}")
            if cps.get("smoke"):
                s = cps["smoke"]
                print(f"  smoke rc={s.get('returncode')}")
                if s.get("stderr_tail"):
                    print(f"    stderr tail: {s['stderr_tail'][-1500:]}")
            print(f"  overall_ok={cps.get('overall_ok')}")

    # Print the LLM validation summary from the message log.
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict) and "[validation_node]" in msg.get("content", ""):
            summary = msg["content"].replace("[validation_node] ", "", 1)
            print(f"\n=== Agent Summary ===\n{summary}")
            break


if __name__ == "__main__":
    main()
