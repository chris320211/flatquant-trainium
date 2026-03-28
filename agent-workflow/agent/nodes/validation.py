"""
validation node — deterministic tool node + optional LLM summary.

Writes generated files to disk, then:
  1. Syntax-checks each .py file via compile()
  2. Attempts importlib import for each file
  3. Checks that FlatQuant wrapper forward() signatures are supersets of
     the original model's forward() signature
Reports {passed, import_errors, signature_errors, syntax_errors, written_files}.
"""

import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from llm import anthropic_text, get_planning_llm
from prompts import VALIDATION_PROMPT
from state import AgentState
from tools import write_output_files

REPO_ROOT = Path(__file__).resolve().parents[3]


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")
    return slug


def validation_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: validate generated files and write them to outputs/.
    """
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)
    generated_files: dict = state.get("generated_files", {})
    modeling_source: str = state.get("modeling_source", "")

    print(f"[validation] Validating {len(generated_files)} generated files ...")

    # Write files to disk first.
    written: dict = write_output_files.invoke({
        "model_name": model_name,
        "files": generated_files,
    })

    syntax_errors: dict = {}
    import_errors: dict = {}
    signature_errors: dict = {}

    out_dir = Path(next(iter(written.values()))).parent if written else None

    # Add output dir to sys.path for import checks.
    original_sys_path = list(sys.path)
    if out_dir and str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    # Also add FlatQuant root so deploy.* imports resolve.
    flatquant_root = str(REPO_ROOT / "FlatQuant")
    if flatquant_root not in sys.path:
        sys.path.insert(0, flatquant_root)

    for filename, source_code in generated_files.items():
        if not filename.endswith(".py"):
            continue

        # 1. Syntax check via compile().
        try:
            compile(source_code, filename, "exec")
        except SyntaxError as e:
            syntax_errors[filename] = f"SyntaxError at line {e.lineno}: {e.msg}"
            continue

        # 2. Import check.
        file_path = written.get(filename)
        if file_path:
            try:
                spec = importlib.util.spec_from_file_location(
                    filename.replace(".py", "").replace("/", "."),
                    file_path,
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except ImportError as e:
                import_errors[filename] = f"ImportError: {e}"
            except Exception as e:
                import_errors[filename] = f"{type(e).__name__}: {e}"

    # 3. Forward-signature check: compare FlatQuant wrapper forward() args
    #    against the original model's forward() args from modeling_source.
    original_sigs = _extract_forward_signatures_from_source(modeling_source)
    for filename, source_code in generated_files.items():
        if not filename.endswith(".py"):
            continue
        wrapper_sigs = _extract_forward_signatures_from_source(source_code)
        for cls_name, wrapper_params in wrapper_sigs.items():
            # Find the best-matching original class (strip "FlatQuant" prefix).
            stripped = cls_name.replace("FlatQuant", "")
            for orig_cls, orig_params in original_sigs.items():
                if stripped.lower() in orig_cls.lower() or orig_cls.lower() in stripped.lower():
                    missing = set(orig_params) - set(wrapper_params) - {"self"}
                    if missing:
                        signature_errors[cls_name] = (
                            f"forward() missing params from {orig_cls}: {sorted(missing)}"
                        )

    sys.path = original_sys_path

    passed = not syntax_errors and not import_errors and not signature_errors

    validation_result = {
        "passed": passed,
        "syntax_errors": syntax_errors,
        "import_errors": import_errors,
        "signature_errors": signature_errors,
        "written_files": written,
    }

    print(
        f"[validation] passed={passed} | "
        f"syntax_errors={len(syntax_errors)} | "
        f"import_errors={len(import_errors)} | "
        f"signature_errors={len(signature_errors)}"
    )

    # Optional LLM summary for human-readable output.
    llm_summary = _llm_summary(validation_result)

    return {
        "validation_result": validation_result,
        "messages": state.get("messages", []) + [
            {
                "role": "assistant",
                "content": f"[validation_node] {llm_summary}",
            }
        ],
    }


def _extract_forward_signatures_from_source(source: str) -> dict[str, list[str]]:
    """
    Parse source code and return {ClassName: [param_names]} for every
    class that defines a forward() method.
    """
    sigs: dict[str, list[str]] = {}
    if not source:
        return sigs
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sigs

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                params = [arg.arg for arg in item.args.args]
                sigs[node.name] = params
    return sigs


def _llm_summary(validation_result: dict) -> str:
    """Use an LLM to produce a concise human-readable validation summary."""
    try:
        llm = get_planning_llm()
        messages = [
            SystemMessage(content=VALIDATION_PROMPT),
            HumanMessage(content=f"validation_result:\n{validation_result}"),
        ]
        response = llm.invoke(messages)
        return anthropic_text(response)
    except Exception as e:
        return f"(LLM summary unavailable: {e})\nRaw result: {validation_result}"
