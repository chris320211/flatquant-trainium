"""
Run generated `calibrate_{slug}.py` in the output directory (optional, env-gated).

Set FLATQUANT_CALIBRATE=smoke|full to enable after validation. Requires weights at
--model (HF id or local path); use FLATQUANT_CALIBRATE_MODEL to override.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from state import AgentState

_REPO = Path(__file__).resolve().parents[3]
_FLATQUANT = _REPO / "FlatQuantBundled"


def _output_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def _utils_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def flatquant_calibrate_node(state: AgentState) -> dict[str, Any]:
    model_name: str = state["model_name"]
    slug = _utils_slug(model_name)
    out_slug = _output_slug(model_name)
    out_dir = _REPO / "agent-workflow" / "outputs" / out_slug
    calibrate_name = f"calibrate_{slug}.py"
    calibrate_path = out_dir / calibrate_name

    mode = os.environ.get("FLATQUANT_CALIBRATE", "").strip().lower()
    report: dict[str, Any] = {
        "skipped": True,
        "mode": mode or None,
        "reason": None,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }

    if mode not in ("smoke", "full"):
        report["reason"] = (
            "FLATQUANT_CALIBRATE not set to smoke|full (unset = skip; avoids heavy work on dev machines)"
        )
        print(f"[flatquant_calibrate] Skip: {report['reason']}", flush=True)
        return {
            "flatquant_calibrate_result": report,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[flatquant_calibrate] Skipped."}],
        }

    if not calibrate_path.is_file():
        report["reason"] = f"missing {calibrate_path}"
        print(f"[flatquant_calibrate] Skip: {report['reason']}", flush=True)
        return {
            "flatquant_calibrate_result": report,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": f"[flatquant_calibrate] {report['reason']}"}],
        }

    model_path = os.environ.get("FLATQUANT_CALIBRATE_MODEL", "").strip() or model_name
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    # Must match flags defined on generated calibrate_{slug}.py (often a subset of
    # flatquant.args_utils — no --w_groupsize / --output_dir / --exp_name unless codegen adds them).
    if mode == "smoke":
        extra = [
            "--quantize",
            "--w_bits",
            "4",
            "--a_bits",
            "16",
            "--cali_dataset",
            "wikitext2",
            "--nsamples",
            "8",
            "--cali_bsz",
            "1",
            "--cali_trans",
            "--epochs",
            "2",
        ]
        timeout = int(os.environ.get("FLATQUANT_CALIBRATE_TIMEOUT_SMOKE_S", "7200"))
    else:
        extra = [
            "--quantize",
            "--w_bits",
            "4",
            "--a_bits",
            "16",
            "--cali_dataset",
            "wikitext2",
            "--nsamples",
            "128",
            "--cali_bsz",
            "1",
            "--cali_trans",
            "--epochs",
            "15",
        ]
        timeout = int(os.environ.get("FLATQUANT_CALIBRATE_TIMEOUT_FULL_S", "86400"))

    cmd: list[str] = [
        sys.executable,
        str(calibrate_path.name),
        "--model",
        model_path,
        *extra,
    ]
    if hf_token:
        cmd.extend(["--hf_token", hf_token])

    user_extra = os.environ.get("FLATQUANT_CALIBRATE_EXTRA_ARGS", "").strip()
    if user_extra:
        cmd.extend(shlex.split(user_extra))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_FLATQUANT), str(out_dir), env.get("PYTHONPATH", "")]
    )

    print(
        f"[flatquant_calibrate] Running {mode} calibration in {out_dir} (timeout={timeout}s) ...",
        flush=True,
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(out_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        report["skipped"] = False
        report["returncode"] = proc.returncode
        report["stdout_tail"] = (proc.stdout or "")[-12_000:]
        report["stderr_tail"] = (proc.stderr or "")[-8000:]
        report["reason"] = None if proc.returncode == 0 else "non_zero_exit"
    except subprocess.TimeoutExpired:
        report["reason"] = f"timeout ({timeout}s)"
        report["stderr_tail"] = "timeout"
    except Exception as e:
        report["reason"] = f"{type(e).__name__}: {e}"

    print(
        f"[flatquant_calibrate] done rc={report.get('returncode')} skipped={report['skipped']}",
        flush=True,
    )
    return {
        "flatquant_calibrate_result": report,
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[flatquant_calibrate] mode={mode} rc={report.get('returncode')}",
            }
        ],
    }
