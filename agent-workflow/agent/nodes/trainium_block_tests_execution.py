"""Phase 2.5: Execute block tests with Trainium2 support.

Runs pytest on generated block tests. Gated by TRAINIUM_RUN_TESTS env var.
Supports both CPU (fallback) and Trainium XLA execution.
"""

import os
import re
import subprocess
from typing import Any

from state import AgentState


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_block_tests_execution_node(state: AgentState) -> dict[str, Any]:
    """Run pytest on Phase 2 block tests."""
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)

    run_tests = os.environ.get("TRAINIUM_RUN_TESTS", "").lower().strip() in ("1", "true", "yes")

    if not run_tests:
        print("[trainium_block_tests_execution] Skipped (TRAINIUM_RUN_TESTS not set).", flush=True)
        return {
            "trainium_block_tests_execution_result": {
                "skipped": True,
                "reason": "not_requested",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_block_tests_execution] Skipped."}],
        }

    block_files = state.get("trainium_block_files") or {}
    if not block_files:
        print("[trainium_block_tests_execution] Skipped (no block files).", flush=True)
        return {
            "trainium_block_tests_execution_result": {
                "skipped": True,
                "reason": "no_block_files",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_block_tests_execution] Skipped."}],
        }

    print(f"[trainium_block_tests_execution] Running block tests for {model_name} ...", flush=True)

    # Build pytest command
    output_dir = f"outputs/{slug}"
    test_files = [k for k in block_files.keys() if k.startswith("tests/") and k.endswith("_test.py")]

    if not test_files:
        print("[trainium_block_tests_execution] No test files found.", flush=True)
        return {
            "trainium_block_tests_execution_result": {
                "skipped": True,
                "reason": "no_test_files",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_block_tests_execution] No test files."}],
        }

    # Prepare pytest args
    pytest_args = [
        "pytest",
        "-v",
        "--tb=short",
        "-x",  # stop on first failure
        f"{output_dir}/tests/",
    ]

    # Optional: use XLA device if on Trainium
    env = os.environ.copy()
    if os.environ.get("TRAINIUM_USE_XLA", "").lower() in ("1", "true", "yes"):
        env["XLA_ACCELERATOR_TYPE"] = "neuron"
        print("  Using XLA Trainium accelerator...", flush=True)
    else:
        print("  Using CPU/fallback device (no XLA)...", flush=True)

    try:
        result = subprocess.run(
            pytest_args,
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
        )

        return {
            "trainium_block_tests_execution_result": {
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "stdout": result.stdout[-4000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            },
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": f"[trainium_block_tests_execution] rc={result.returncode} success={result.returncode == 0}",
                }
            ],
        }
    except subprocess.TimeoutExpired:
        print("[trainium_block_tests_execution] Timeout (10 minutes).", flush=True)
        return {
            "trainium_block_tests_execution_result": {
                "returncode": -1,
                "success": False,
                "error": "timeout_10m",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_block_tests_execution] Timeout."}],
        }
    except Exception as e:
        print(f"[trainium_block_tests_execution] Error: {e}", flush=True)
        return {
            "trainium_block_tests_execution_result": {
                "returncode": -1,
                "success": False,
                "error": str(e),
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": f"[trainium_block_tests_execution] Error: {e}"}],
        }
