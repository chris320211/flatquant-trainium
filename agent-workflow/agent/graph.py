"""
LangGraph StateGraph definition.

Wires nodes:
  arch → ref_reader → codegen → registration → validation
  → if validation failed → END
  → else flatquant_calibrate (no-op unless FLATQUANT_CALIBRATE=smoke|full)
  → if TRAINIUM_SKILL_MODE=fast → nxdi_port
  → else full skill chain:
        trainium_plan → trainium_skill_setup → trainium_blocks → trainium_test_audit
        → trainium_block_tests → trainium_integrate → trainium_weight_map
        → trainium_verify
  → trainium_compile_smoke (no-op unless TRAINIUM_COMPILE_CMD / TRAINIUM_SMOKE_CMD set)
  → END

Env (high level):
  FLATQUANT_CALIBRATE=smoke|full — run generated calibrate_{slug}.py on Trainium
  TRAINIUM_COMPILE_CMD / TRAINIUM_SMOKE_CMD — optional Neuron compile/infer after nxdi verify
  TRAINIUM_SKILL_MODE — full (default) | fast
  TRAINIUM_RUN_BLOCK_TESTS — 1 to run pytest after Phase 2
  TRAINIUM_SKIP_VERIFY / TRAINIUM_SKIP_COMPILE_SMOKE — skip gates
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
from nodes.trainium_compile_smoke import trainium_compile_smoke_node
from nodes.trainium_blocks import trainium_blocks_node
from nodes.trainium_integrate import trainium_integrate_node
from nodes.trainium_plan import trainium_plan_node
from nodes.trainium_skill_setup import trainium_skill_setup_node
from nodes.trainium_test_audit import trainium_test_audit_node
from nodes.trainium_verify import trainium_verify_node
from nodes.trainium_weight_map import trainium_weight_map_node
from nodes.validation import validation_node
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
    graph.add_node("trainium_block_tests", trainium_block_tests_node)
    graph.add_node("trainium_integrate", trainium_integrate_node)
    graph.add_node("trainium_weight_map", trainium_weight_map_node)
    graph.add_node("trainium_verify", trainium_verify_node)
    graph.add_node("trainium_compile_smoke", trainium_compile_smoke_node)

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
    graph.add_edge("trainium_blocks", "trainium_test_audit")
    graph.add_edge("trainium_test_audit", "trainium_block_tests")
    graph.add_edge("trainium_block_tests", "trainium_integrate")
    graph.add_edge("trainium_integrate", "trainium_weight_map")
    graph.add_edge("trainium_weight_map", "trainium_verify")
    graph.add_edge("trainium_verify", "trainium_compile_smoke")
    graph.add_edge("nxdi_port", "trainium_compile_smoke")
    graph.add_edge("trainium_compile_smoke", END)

    return graph.compile()
