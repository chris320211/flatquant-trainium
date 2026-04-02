"""Final verification: Ensure all required output files are present and valid.

Checks that code generation produced all expected files without execution.
"""

import os
import re
from typing import Any

from state import AgentState


def _model_slug(model_name: str) -> str:
    base = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", base).lower().strip("_")


def trainium_verify_outputs_node(state: AgentState) -> dict[str, Any]:
    """Verify all expected output files exist."""
    model_name: str = state["model_name"]
    slug = _model_slug(model_name)

    output_dir = f"outputs/{slug}"

    print(f"[trainium_verify_outputs] Verifying outputs for {model_name} ...", flush=True)

    # Expected files
    expected_flatquant = [
        f"{slug}_utils.py",
        f"calibrate_{slug}.py",
        f"quant_config_{slug}.py",
        f"modeling_{slug}.py",
        f"patch_{slug}.py",
    ]

    expected_nxdi = [
        "nxdi/phase1_plan.json",
        "nxdi/neuron_config.py",
        "nxdi/inference_config.py",
        "nxdi/neuron_model.py",
        "nxdi/neuron_causal_lm.py",
        "nxdi/convert_weights.py",
    ]

    expected_tests = [
        "tests/__init__.py",
        "tests/block_testing_utils.py",
    ]

    missing_flatquant = []
    missing_nxdi = []
    missing_tests = []

    # Check FlatQuant files
    for fname in expected_flatquant:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            missing_flatquant.append(fname)
        else:
            print(f"  ✓ {fname}", flush=True)

    # Check NxDI files
    for fname in expected_nxdi:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            missing_nxdi.append(fname)
        else:
            print(f"  ✓ {fname}", flush=True)

    # Check test files
    for fname in expected_tests:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            missing_tests.append(fname)
        else:
            print(f"  ✓ {fname}", flush=True)

    # Count generated test files
    tests_dir = os.path.join(output_dir, "tests")
    test_files = []
    if os.path.exists(tests_dir):
        test_files = [f for f in os.listdir(tests_dir) if f.endswith("_test.py")]
        print(f"  ✓ Found {len(test_files)} generated test file(s)", flush=True)

    all_missing = missing_flatquant + missing_nxdi + missing_tests
    passed = len(all_missing) == 0

    result = {
        "passed": passed,
        "missing_flatquant": missing_flatquant,
        "missing_nxdi": missing_nxdi,
        "missing_tests": missing_tests,
        "generated_test_count": len(test_files),
        "output_directory": os.path.abspath(output_dir),
    }

    if passed:
        print(
            f"[trainium_verify_outputs] ✓ All expected outputs present! Ready for Trainium deployment.",
            flush=True,
        )
    else:
        print(f"[trainium_verify_outputs] ✗ Missing files: {all_missing}", flush=True)

    return {
        "trainium_verify_outputs_result": result,
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": f"[trainium_verify_outputs] passed={passed} missing={len(all_missing)}",
            }
        ],
    }
