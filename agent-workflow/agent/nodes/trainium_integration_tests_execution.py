"""Phase 3.5 execution: Run integration tests.

Tests full model initialization and forward pass.
Gated by TRAINIUM_RUN_TESTS env var.
"""

import os
import re
import subprocess
from typing import Any

from state import AgentState


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_integration_tests_execution_node(state: AgentState) -> dict[str, Any]:
    """Run integration tests."""
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)

    run_tests = os.environ.get("TRAINIUM_RUN_TESTS", "").lower().strip() in ("1", "true", "yes")

    if not run_tests:
        print("[trainium_integration_tests_execution] Skipped (TRAINIUM_RUN_TESTS not set).", flush=True)
        return {
            "trainium_integration_tests_execution_result": {
                "skipped": True,
                "reason": "not_requested",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_integration_tests_execution] Skipped."}],
        }

    test_result = state.get("trainium_integration_tests_result") or {}
    if test_result.get("skipped"):
        print("[trainium_integration_tests_execution] Skipped (no integration tests generated).", flush=True)
        return {
            "trainium_integration_tests_execution_result": {
                "skipped": True,
                "reason": "no_tests_generated",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_integration_tests_execution] Skipped."}],
        }

    print(f"[trainium_integration_tests_execution] Running integration tests for {model_name} ...", flush=True)

    output_dir = f"outputs/{slug}"
    pytest_args = [
        "pytest",
        "-v",
        "--tb=short",
        "-x",
        f"{output_dir}/tests/",
        "-k",
        "integration",  # only run integration tests
    ]

    env = os.environ.copy()
    if os.environ.get("TRAINIUM_USE_XLA", "").lower() in ("1", "true", "yes"):
        env["XLA_ACCELERATOR_TYPE"] = "neuron"
        print("  Using XLA Trainium accelerator...", flush=True)
    else:
        print("  Using CPU/fallback device...", flush=True)

    try:
        result = subprocess.run(
            pytest_args,
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

        return {
            "trainium_integration_tests_execution_result": {
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "stdout": result.stdout[-4000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            },
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": f"[trainium_integration_tests_execution] rc={result.returncode}",
                }
            ],
        }
    except subprocess.TimeoutExpired:
        print("[trainium_integration_tests_execution] Timeout.", flush=True)
        return {
            "trainium_integration_tests_execution_result": {
                "returncode": -1,
                "success": False,
                "error": "timeout",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_integration_tests_execution] Timeout."}],
        }
    except Exception as e:
        print(f"[trainium_integration_tests_execution] Error: {e}", flush=True)
        return {
            "trainium_integration_tests_execution_result": {
                "returncode": -1,
                "success": False,
                "error": str(e),
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": f"[trainium_integration_tests_execution] Error: {e}"}],
        }
