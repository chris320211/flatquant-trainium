"""Copy block_testing_utils and optionally run pytest (Phase 2 execution)."""

import os
import sys
import subprocess
from pathlib import Path
from typing import Any

from skill_loader import load_block_testing_utils
from state import AgentState
from tools import REPO_ROOT, write_output_files


def trainium_block_tests_node(state: AgentState) -> dict[str, Any]:
    model_name: str = state["model_name"]
    out_slug = model_name.replace("/", "__").replace(" ", "_")
    out_dir = REPO_ROOT / "agent-workflow" / "outputs" / out_slug

    generated_files = dict(state.get("generated_files", {}))
    extra: dict[str, str] = {}
    if "tests/block_testing_utils.py" not in generated_files:
        extra["tests/block_testing_utils.py"] = load_block_testing_utils()
    if extra:
        written_utils = write_output_files.invoke({"model_name": model_name, "files": extra})
        generated_files.update(extra)
    else:
        written_utils = {}

    report: dict[str, Any] = {
        "written_block_testing_utils": written_utils.get("tests/block_testing_utils.py"),
        "skipped": True,
        "reason": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }

    run = os.environ.get("TRAINIUM_RUN_BLOCK_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not run:
        report["reason"] = "TRAINIUM_RUN_BLOCK_TESTS not set (set to 1 to run pytest)"
        print("[trainium_block_tests] Skipping pytest (TRAINIUM_RUN_BLOCK_TESTS unset).", flush=True)
        return {
            "trainium_test_report": report,
            "generated_files": generated_files,
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": "[trainium_block_tests] Copied block_testing_utils; pytest skipped.",
                }
            ],
        }

    if not out_dir.is_dir():
        report["reason"] = "output directory missing"
        return {
            "trainium_test_report": report,
            "generated_files": generated_files,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_block_tests] No output dir."}],
        }

    flatquant_root = str(REPO_ROOT / "FlatQuantBundled")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(out_dir), flatquant_root, env.get("PYTHONPATH", "")]
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        report["skipped"] = False
        report["returncode"] = proc.returncode
        report["stdout"] = (proc.stdout or "")[-8000:]
        report["stderr"] = (proc.stderr or "")[-4000:]
    except subprocess.TimeoutExpired:
        report["reason"] = "pytest timeout (600s)"
        report["stderr"] = "timeout"
    except FileNotFoundError:
        report["reason"] = "python/pytest not found on PATH"
    except Exception as e:
        report["reason"] = f"{type(e).__name__}: {e}"

    print(
        f"[trainium_block_tests] pytest rc={report.get('returncode')} skipped={report['skipped']}",
        flush=True,
    )
    return {
        "trainium_test_report": report,
        "generated_files": generated_files,
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_block_tests] {report.get('reason') or report}",
            }
        ],
    }
