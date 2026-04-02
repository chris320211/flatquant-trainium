"""
Optional Neuron compile + smoke commands after nxdi verify.

Set TRAINIUM_COMPILE_CMD and/or TRAINIUM_SMOKE_CMD (shell strings, run from output dir).
Example:
  export TRAINIUM_COMPILE_CMD='python nxdi/compile_llm.py --config nxdi/compile_config.json'
  export TRAINIUM_SMOKE_CMD='python nxdi/smoke_infer.py --compiled-dir ./compiled'
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from state import AgentState

_REPO = Path(__file__).resolve().parents[3]


def _output_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def _run_shell(
    label: str,
    cmd: str,
    cwd: Path,
    env: dict,
    timeout: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"cmd": cmd, "returncode": None, "stdout_tail": "", "stderr_tail": ""}
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out["returncode"] = proc.returncode
        out["stdout_tail"] = (proc.stdout or "")[-8000:]
        out["stderr_tail"] = (proc.stderr or "")[-8000:]
    except subprocess.TimeoutExpired:
        out["returncode"] = -1
        out["stderr_tail"] = f"{label} timeout ({timeout}s)"
    except Exception as e:
        out["returncode"] = -1
        out["stderr_tail"] = f"{type(e).__name__}: {e}"
    return out


def trainium_compile_smoke_node(state: AgentState) -> dict[str, Any]:
    if os.environ.get("TRAINIUM_SKIP_COMPILE_SMOKE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        print("[trainium_compile_smoke] Skipped (TRAINIUM_SKIP_COMPILE_SMOKE=1).", flush=True)
        return {
            "trainium_compile_smoke_result": {"skipped": True, "reason": "TRAINIUM_SKIP_COMPILE_SMOKE"},
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_compile_smoke] Skipped."}],
        }

    model_name: str = state["model_name"]
    out_dir = _REPO / "agent-workflow" / "outputs" / _output_slug(model_name)

    compile_cmd = os.environ.get("TRAINIUM_COMPILE_CMD", "").strip()
    smoke_cmd = os.environ.get("TRAINIUM_SMOKE_CMD", "").strip()
    report: dict[str, Any] = {
        "skipped": True,
        "compile": None,
        "smoke": None,
        "reason": None,
    }

    if not compile_cmd and not smoke_cmd:
        report["reason"] = (
            "Set TRAINIUM_COMPILE_CMD and/or TRAINIUM_SMOKE_CMD to run post-verify Neuron steps"
        )
        print(f"[trainium_compile_smoke] Skip: {report['reason']}", flush=True)
        return {
            "trainium_compile_smoke_result": report,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_compile_smoke] Skipped (no cmds)."}],
        }

    if not out_dir.is_dir():
        report["reason"] = "output directory missing"
        return {
            "trainium_compile_smoke_result": report,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_compile_smoke] No output dir."}],
        }

    env = os.environ.copy()
    flat = str(_REPO / "FlatQuantBundled")
    env["PYTHONPATH"] = os.pathsep.join([str(out_dir), flat, env.get("PYTHONPATH", "")])

    compile_timeout = int(os.environ.get("TRAINIUM_COMPILE_TIMEOUT_S", "14400"))
    smoke_timeout = int(os.environ.get("TRAINIUM_SMOKE_TIMEOUT_S", "3600"))

    report["skipped"] = False
    if compile_cmd:
        print("[trainium_compile_smoke] Running TRAINIUM_COMPILE_CMD ...", flush=True)
        report["compile"] = _run_shell("compile", compile_cmd, out_dir, env, compile_timeout)
    if smoke_cmd:
        print("[trainium_compile_smoke] Running TRAINIUM_SMOKE_CMD ...", flush=True)
        report["smoke"] = _run_shell("smoke", smoke_cmd, out_dir, env, smoke_timeout)

    ok_c = report.get("compile") is None or report["compile"].get("returncode") == 0
    ok_s = report.get("smoke") is None or report["smoke"].get("returncode") == 0
    report["overall_ok"] = ok_c and ok_s

    return {
        "trainium_compile_smoke_result": report,
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_compile_smoke] overall_ok={report.get('overall_ok')}",
            }
        ],
    }
