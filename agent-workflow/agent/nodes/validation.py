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


def _calibrate_forbidden_import_message(filename: str, source: str) -> str | None:
    """
    Block hallucinated FlatQuant calibration modules (no flatquant.cali_utils in repo).
    """
    if not (filename.startswith("calibrate_") and filename.endswith(".py")):
        return None
    if "flatquant.cali_utils" in source:
        return (
            "Static import error: flatquant.cali_utils does not exist. "
            "Use flatquant.train_utils (cali_flat_quant) per FlatQuantBundled/main.py."
        )
    if re.search(r"from\s+flatquant\s+import\s+cali_utils\b", source):
        return (
            "Static import error: do not import cali_utils from flatquant. "
            "Use flatquant.train_utils for cali_flat_quant."
        )
    return None


def _forbidden_cuda_message(filename: str, source: str) -> str | None:
    """
    Trainium / CPU-safe calibration: no .cuda() or torch.cuda.* in training wrappers
    and entry scripts.
    """
    if not filename.endswith(".py"):
        return None
    is_utils = filename.endswith("_utils.py")
    is_patch = filename.startswith("patch_")
    is_calibrate = filename.startswith("calibrate_")
    if not (is_utils or is_patch or is_calibrate):
        return None
    if re.search(r"\.cuda\s*\(", source):
        return (
            "Static check: do not use .cuda(); use flatquant.utils.DEV and .to(DEV) "
            "for buffers (Trainium has no CUDA). Import: from flatquant.utils import DEV."
        )
    if re.search(r"\btorch\.cuda\.", source):
        return (
            "Static check: do not use torch.cuda.*; use flatquant.utils.DEV for device placement."
        )
    return None


def _utils_attention_num_heads_message(filename: str, source: str) -> str | None:
    """Llama-style FlatQuant attention must use config.num_attention_heads in add_fq_trans."""
    if not (filename.endswith("_utils.py") and "add_fq_trans" in source):
        return None
    if re.search(
        r"(SingleTransMatrix|SVDSingleTransMatrix|InvSingleTransMatrix)\s*\(\s*self\.num_heads\s*\)",
        source,
    ):
        return (
            "Static check: in add_fq_trans use self.config.num_attention_heads for "
            "SingleTransMatrix/SVDSingleTransMatrix (not self.num_heads). "
            "See FlatQuantBundled/flatquant/model_tools/llama_utils.py."
        )
    return None


def _deploy_quantization_public_names() -> set[str]:
    """Top-level public class/function names in FlatQuantBundled deploy/nn/quantization.py."""
    path = REPO_ROOT / "FlatQuantBundled" / "deploy" / "nn" / "quantization.py"
    try:
        qsrc = path.read_text()
    except OSError:
        return {"Quantizer"}
    try:
        tree = ast.parse(qsrc)
    except SyntaxError:
        return {"Quantizer"}
    return {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")
    }


def _modeling_deploy_quantization_import_error(filename: str, source: str) -> str | None:
    """
    Block hallucinated imports from deploy.nn.quantization (only real symbols allowed).
    """
    if not (filename.startswith("modeling_") and filename.endswith(".py")):
        return None
    allowed = _deploy_quantization_public_names()
    if not allowed:
        allowed = {"Quantizer"}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "deploy.nn.quantization":
            continue
        if any(alias.name == "*" for alias in node.names):
            return (
                "Static check: do not use `from deploy.nn.quantization import *`; "
                f"import only: {sorted(allowed)}"
            )
        bad = [alias.name for alias in node.names if alias.name not in allowed]
        if bad:
            return (
                f"Static check: invalid name(s) imported from deploy.nn.quantization: {bad}. "
                f"Allowed: {sorted(allowed)}"
            )
    return None


def _quant_config_get_quantization_args_message(filename: str, source: str) -> str | None:
    """run_{slug}.py imports get_quantization_args from quant_config_{slug}.py."""
    if not (filename.startswith("quant_config_") and filename.endswith(".py")):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    has = any(
        isinstance(n, ast.FunctionDef) and n.name == "get_quantization_args"
        for n in tree.body
    )
    if not has:
        return (
            "Static check: define def get_quantization_args(...) at module level; "
            "run script imports it from this module."
        )
    return None


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
    flatquant_root = str(REPO_ROOT / "FlatQuantBundled")
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

        bad_cali = _calibrate_forbidden_import_message(filename, source_code)
        if bad_cali:
            import_errors[filename] = bad_cali
            continue

        bad_heads = _utils_attention_num_heads_message(filename, source_code)
        if bad_heads:
            import_errors[filename] = bad_heads
            continue

        bad_cuda = _forbidden_cuda_message(filename, source_code)
        if bad_cuda:
            import_errors[filename] = bad_cuda
            continue

        bad_quant = _quant_config_get_quantization_args_message(filename, source_code)
        if bad_quant:
            import_errors[filename] = bad_quant
            continue

        bad_deploy_q = _modeling_deploy_quantization_import_error(filename, source_code)
        if bad_deploy_q:
            import_errors[filename] = bad_deploy_q
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
                    if any(p.startswith("**") for p in wrapper_params):
                        missing = set()
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
                params.extend(a.arg for a in item.args.kwonlyargs)
                if item.args.vararg:
                    params.append("*" + item.args.vararg.arg)
                if item.args.kwarg:
                    params.append("**" + item.args.kwarg.arg)
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
