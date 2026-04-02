"""
LangGraph StateGraph definition (Trainium2-optimized).

Code Generation Path (ALWAYS):
  arch → ref_reader → codegen → registration → validation
  → flatquant_calibrate (optional, gated by FLATQUANT_CALIBRATE)
  → if TRAINIUM_SKILL_MODE=fast → nxdi_port
  → else full skill chain:
        trainium_plan → trainium_skill_setup → trainium_blocks
        → trainium_integrate → trainium_weight_map

Test Generation (ALWAYS after code generation):
  → trainium_integration_tests → trainium_weight_tests

Test Execution (OPTIONAL, gated by TRAINIUM_RUN_TESTS):
  → trainium_block_tests_execution → trainium_integration_tests_execution
  → trainium_weight_tests_execution

Compilation (OPTIONAL, gated by TRAINIUM_COMPILE):
  → trainium_neuron_compile

Output Verification (ALWAYS at end):
  → trainium_verify_outputs → END

Env Variables (Trainium2):
  FLATQUANT_CALIBRATE=smoke|full — run generated calibrate_{slug}.py
  TRAINIUM_SKILL_MODE=fast|full — use simple or detailed NxDI generation (default: full)
  TRAINIUM_RUN_TESTS=1 — execute generated tests (default: skip)
  TRAINIUM_USE_XLA=1 — use XLA Trainium accelerator for tests (default: CPU)
  TRAINIUM_COMPILE=1 — compile with neuronx_compiler (default: skip)
  TRAINIUM_COMPILE_CMD — custom shell command to run for compilation
"""

import os

from langgraph.graph import END, START, StateGraph

from nodes.arch import arch_node
from nodes.codegen import codegen_node
from nodes.flatquant_calibrate import flatquant_calibrate_node
from nodes.nxdi_port import nxdi_port_node
from nodes.ref_reader import ref_reader_node
from nodes.registration import registration_node
from nodes.trainium_block_tests import trainium_block_tests_node
from nodes.trainium_blocks import trainium_blocks_node
from nodes.trainium_integrate import trainium_integrate_node
from nodes.trainium_plan import trainium_plan_node
from nodes.trainium_skill_setup import trainium_skill_setup_node
from nodes.trainium_test_audit import trainium_test_audit_node
from nodes.trainium_verify import trainium_verify_node
from nodes.trainium_weight_map import trainium_weight_map_node
from nodes.validation import validation_node
from nodes.trainium_integration_tests import trainium_integration_tests_node
from nodes.trainium_weight_tests import trainium_weight_tests_node
from nodes.trainium_block_tests_execution import trainium_block_tests_execution_node
from nodes.trainium_integration_tests_execution import trainium_integration_tests_execution_node
from nodes.trainium_weight_tests_execution import trainium_weight_tests_execution_node
from nodes.trainium_neuron_compile import trainium_neuron_compile_node
from nodes.trainium_verify_outputs import trainium_verify_outputs_node
from state import AgentState


def _route_after_validation(state: AgentState) -> str:
    vr = state.get("validation_result") or {}
    if not vr.get("passed"):
        return "end"
    return "flatquant_calibrate"


def _route_after_calibrate(state: AgentState) -> str:
    mode = os.environ.get("TRAINIUM_SKILL_MODE", "full").lower().strip()
    if mode == "fast":
        return "nxdi_port"
    return "trainium_plan"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Code generation nodes
    graph.add_node("arch", arch_node)
    graph.add_node("ref_reader", ref_reader_node)
    graph.add_node("codegen", codegen_node)
    graph.add_node("registration", registration_node)
    graph.add_node("validation", validation_node)
    graph.add_node("flatquant_calibrate", flatquant_calibrate_node)
    graph.add_node("nxdi_port", nxdi_port_node)
    graph.add_node("trainium_plan", trainium_plan_node)
    graph.add_node("trainium_skill_setup", trainium_skill_setup_node)
    graph.add_node("trainium_blocks", trainium_blocks_node)
    graph.add_node("trainium_test_audit", trainium_test_audit_node)
    graph.add_node("trainium_integrate", trainium_integrate_node)
    graph.add_node("trainium_weight_map", trainium_weight_map_node)
    graph.add_node("trainium_verify", trainium_verify_node)

    # Test generation nodes (new)
    graph.add_node("trainium_integration_tests", trainium_integration_tests_node)
    graph.add_node("trainium_weight_tests", trainium_weight_tests_node)

    # Test execution nodes (new, optional)
    graph.add_node("trainium_block_tests_execution", trainium_block_tests_execution_node)
    graph.add_node("trainium_integration_tests_execution", trainium_integration_tests_execution_node)
    graph.add_node("trainium_weight_tests_execution", trainium_weight_tests_execution_node)

    # Compilation node (new, optional, replaces trainium_compile_smoke)
    graph.add_node("trainium_neuron_compile", trainium_neuron_compile_node)

    # Output verification node (new)
    graph.add_node("trainium_verify_outputs", trainium_verify_outputs_node)

    # Code generation edges
    graph.add_edge(START, "arch")
    graph.add_edge("arch", "ref_reader")
    graph.add_edge("ref_reader", "codegen")
    graph.add_edge("codegen", "registration")
    graph.add_edge("registration", "validation")
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {
            "flatquant_calibrate": "flatquant_calibrate",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "flatquant_calibrate",
        _route_after_calibrate,
        {
            "trainium_plan": "trainium_plan",
            "nxdi_port": "nxdi_port",
        },
    )
    graph.add_edge("trainium_plan", "trainium_skill_setup")
    graph.add_edge("trainium_skill_setup", "trainium_blocks")
    # Skip test_audit and directly go to integrate (no test execution in codegen path)
    graph.add_edge("trainium_blocks", "trainium_integrate")
    graph.add_edge("trainium_integrate", "trainium_weight_map")
    graph.add_edge("trainium_weight_map", "trainium_verify")

    # nxdi_port path also converges to test generation
    graph.add_edge("nxdi_port", "trainium_integration_tests")

    # Test generation path (always run after code generation)
    graph.add_edge("trainium_verify", "trainium_integration_tests")
    graph.add_edge("trainium_integration_tests", "trainium_weight_tests")

    # Test execution path (optional, gated by env var)
    graph.add_edge("trainium_weight_tests", "trainium_block_tests_execution")
    graph.add_edge("trainium_block_tests_execution", "trainium_integration_tests_execution")
    graph.add_edge("trainium_integration_tests_execution", "trainium_weight_tests_execution")

    # Compilation (optional, gated by env var)
    graph.add_edge("trainium_weight_tests_execution", "trainium_neuron_compile")

    # Output verification (always at end)
    graph.add_edge("trainium_neuron_compile", "trainium_verify_outputs")
    graph.add_edge("trainium_verify_outputs", END)

    return graph.compile()
