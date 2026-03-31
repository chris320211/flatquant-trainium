"""Trainium2-optimized Neuron compilation node.

Compiles NxDI model with neuronx_compiler.
Gated by TRAINIUM_COMPILE env var.
Replaces legacy trainium_compile_smoke_node for Trainium2 instances.
"""

import os
import re
import subprocess
from typing import Any

from state import AgentState


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_neuron_compile_node(state: AgentState) -> dict[str, Any]:
    """Compile NxDI model with neuronx_compiler on Trainium2."""
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)

    should_compile = os.environ.get("TRAINIUM_COMPILE", "").lower().strip() in (
        "1",
        "true",
        "yes",
    )

    if not should_compile:
        print("[trainium_neuron_compile] Skipped (TRAINIUM_COMPILE not set).", flush=True)
        return {
            "trainium_neuron_compile_result": {
                "skipped": True,
                "reason": "not_requested",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_neuron_compile] Skipped."}],
        }

    validation = state.get("validation_result") or {}
    if not validation.get("passed"):
        print("[trainium_neuron_compile] Skipped (validation did not pass).", flush=True)
        return {
            "trainium_neuron_compile_result": {
                "skipped": True,
                "reason": "validation_failed",
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_neuron_compile] Skipped."}],
        }

    print(f"[trainium_neuron_compile] Compiling NxDI model for {model_name} with neuronx_compiler ...", flush=True)

    output_dir = f"outputs/{slug}"
    nxdi_model_path = f"{output_dir}/nxdi/neuron_causal_lm.py"

    if not os.path.exists(nxdi_model_path):
        print(f"[trainium_neuron_compile] Model file not found: {nxdi_model_path}", flush=True)
        return {
            "trainium_neuron_compile_result": {
                "skipped": True,
                "reason": "model_file_not_found",
                "path": nxdi_model_path,
            },
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_neuron_compile] Model file not found."}],
        }

    # Check if neuronx_compiler is available
    result_check = subprocess.run(
        ["python", "-c", "import neuronx_distributed_inference; print('neuronx available')"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result_check.returncode != 0:
        print(
            "[trainium_neuron_compile] Warning: neuronx_distributed_inference not available. "
            "Compile will likely fail. Make sure you're on a Trainium instance.",
            flush=True,
        )

    # Optional custom compile command from env var
    custom_compile_cmd = os.environ.get("TRAINIUM_COMPILE_CMD")

    if custom_compile_cmd:
        print(f"[trainium_neuron_compile] Running custom compile command: {custom_compile_cmd}", flush=True)
        try:
            result = subprocess.run(
                custom_compile_cmd,
                shell=True,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes for compilation
            )

            return {
                "trainium_neuron_compile_result": {
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                    "command": custom_compile_cmd,
                    "stdout": result.stdout[-4000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else "",
                },
                "messages": state.get("messages", [])
                + [
                    {
                        "role": "assistant",
                        "content": f"[trainium_neuron_compile] custom compile rc={result.returncode}",
                    }
                ],
            }
        except subprocess.TimeoutExpired:
            print("[trainium_neuron_compile] Compile timeout (30 minutes).", flush=True)
            return {
                "trainium_neuron_compile_result": {
                    "returncode": -1,
                    "success": False,
                    "error": "timeout_30m",
                },
                "messages": state.get("messages", [])
                + [{"role": "assistant", "content": "[trainium_neuron_compile] Timeout."}],
            }
        except Exception as e:
            print(f"[trainium_neuron_compile] Error: {e}", flush=True)
            return {
                "trainium_neuron_compile_result": {
                    "returncode": -1,
                    "success": False,
                    "error": str(e),
                },
                "messages": state.get("messages", [])
                + [{"role": "assistant", "content": f"[trainium_neuron_compile] Error: {e}"}],
            }
    else:
        # Default: just report that compile would happen here
        print(
            "[trainium_neuron_compile] No custom compile command set. "
            "Set TRAINIUM_COMPILE_CMD to run compilation.",
            flush=True,
        )
        return {
            "trainium_neuron_compile_result": {
                "skipped": True,
                "reason": "no_compile_command",
                "note": "Set TRAINIUM_COMPILE_CMD env var to enable compilation",
            },
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": "[trainium_neuron_compile] No compile command; set TRAINIUM_COMPILE_CMD.",
                }
            ],
        }
