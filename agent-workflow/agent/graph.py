"""
LangGraph StateGraph definition.

Wires nodes:
  arch → ref_reader → codegen → registration → validation
  → if validation passed:
      TRAINIUM_SKILL_MODE=fast → nxdi_port → END
      else (default full) → trainium_plan → trainium_blocks → trainium_block_tests
      → trainium_integrate → trainium_weight_map → END
  → else END

Env:
  TRAINIUM_SKILL_MODE — full (default) | fast (single nxdi_port LLM call only)
  TRAINIUM_RUN_BLOCK_TESTS — set to 1 to run pytest after Phase 2
"""

import os

from langgraph.graph import END, START, StateGraph

from nodes.arch import arch_node
from nodes.codegen import codegen_node
from nodes.nxdi_port import nxdi_port_node
from nodes.ref_reader import ref_reader_node
from nodes.registration import registration_node
from nodes.trainium_block_tests import trainium_block_tests_node
from nodes.trainium_blocks import trainium_blocks_node
from nodes.trainium_integrate import trainium_integrate_node
from nodes.trainium_plan import trainium_plan_node
from nodes.trainium_weight_map import trainium_weight_map_node
from nodes.validation import validation_node
from state import AgentState


def _route_after_validation(state: AgentState) -> str:
    vr = state.get("validation_result") or {}
    if not vr.get("passed"):
        return "end"
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
    graph.add_node("nxdi_port", nxdi_port_node)
    graph.add_node("trainium_plan", trainium_plan_node)
    graph.add_node("trainium_blocks", trainium_blocks_node)
    graph.add_node("trainium_block_tests", trainium_block_tests_node)
    graph.add_node("trainium_integrate", trainium_integrate_node)
    graph.add_node("trainium_weight_map", trainium_weight_map_node)

    graph.add_edge(START, "arch")
    graph.add_edge("arch", "ref_reader")
    graph.add_edge("ref_reader", "codegen")
    graph.add_edge("codegen", "registration")
    graph.add_edge("registration", "validation")
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {
            "trainium_plan": "trainium_plan",
            "nxdi_port": "nxdi_port",
            "end": END,
        },
    )
    graph.add_edge("trainium_plan", "trainium_blocks")
    graph.add_edge("trainium_blocks", "trainium_block_tests")
    graph.add_edge("trainium_block_tests", "trainium_integrate")
    graph.add_edge("trainium_integrate", "trainium_weight_map")
    graph.add_edge("trainium_weight_map", END)
    graph.add_edge("nxdi_port", END)

    return graph.compile()
