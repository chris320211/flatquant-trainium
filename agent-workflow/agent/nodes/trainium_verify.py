"""
Post-pipeline checks: neuronx_distributed_inference import + exec of generated nxdi module.

Runs on the host that executes the agent (Trainium AMI should pass package import;
generated code may still fail until hand-fixed).
"""

import os
import re
import subprocess
import sys
import textwrap
from typing import Any

from state import AgentState
from tools import REPO_ROOT


def trainium_verify_node(state: AgentState) -> dict[str, Any]:
    if os.environ.get("TRAINIUM_SKIP_VERIFY", "").strip().lower() in ("1", "true", "yes"):
        print("[trainium_verify] Skipped (TRAINIUM_SKIP_VERIFY=1).", flush=True)
        return {
            "trainium_nxdi_verify": {"skipped": True, "reason": "TRAINIUM_SKIP_VERIFY"},
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_verify] Skipped."}],
        }

    model_name: str = state["model_name"]
    out_slug = model_name.replace("/", "__").replace(" ", "_")
    out_dir = REPO_ROOT / "agent-workflow" / "outputs" / out_slug
    base = model_name.split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")
    neuron_rel = f"nxdi/neuron_{slug}_nxdi.py"
    neuron_path = out_dir / neuron_rel.replace("/", os.sep)

    report: dict[str, Any] = {
        "output_dir": str(out_dir),
        "neuron_module_path": str(neuron_path),
        "neuron_pkg_import_ok": None,
        "neuron_module_exec_ok": None,
        "stderr": "",
    }

    if not out_dir.is_dir():
        report["error"] = "output directory missing"
        return {
            "trainium_nxdi_verify": report,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "[trainium_verify] No output dir."}],
        }

    script = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        from pathlib import Path

        root = Path({str(out_dir)!r})
        sys.path.insert(0, str(root))
        try:
            import neuronx_distributed_inference  # noqa: F401
            print("NEURON_PKG_OK True")
        except Exception as e:
            print("NEURON_PKG_OK False")
            print("NEURON_PKG_ERR", repr(e))

        p = Path({str(neuron_path)!r})
        if not p.is_file():
            print("NEURON_MOD_OK False")
            print("NEURON_MOD_ERR missing_file")
        else:
            spec = importlib.util.spec_from_file_location("_nxdi_generated_check", p)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                print("NEURON_MOD_OK True")
            except Exception as e:
                print("NEURON_MOD_OK False")
                print("NEURON_MOD_ERR", repr(e))
        """
    ).strip()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(out_dir), str(REPO_ROOT / "FlatQuantBundled"), env.get("PYTHONPATH", "")]
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        report["raw_output"] = out[-6000:]
        report["neuron_pkg_import_ok"] = "NEURON_PKG_OK True" in out
        report["neuron_module_exec_ok"] = "NEURON_MOD_OK True" in out
        report["returncode"] = proc.returncode
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"

    ok = report.get("neuron_module_exec_ok") is True
    print(
        f"[trainium_verify] pkg={report.get('neuron_pkg_import_ok')} "
        f"mod_exec={report.get('neuron_module_exec_ok')}",
        flush=True,
    )
    return {
        "trainium_nxdi_verify": report,
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_verify] neuron_pkg={report.get('neuron_pkg_import_ok')} "
                f"neuron_mod_exec={report.get('neuron_module_exec_ok')}",
            }
        ],
    }
