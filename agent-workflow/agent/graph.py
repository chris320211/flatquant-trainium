"""
LangGraph StateGraph definition.

Wires nodes:
  arch → ref_reader → codegen → registration → validation
  → (if validation passed) nxdi_port → END
  → (else) END
"""

from langgraph.graph import END, START, StateGraph

from nodes.arch import arch_node
from nodes.codegen import codegen_node
from nodes.nxdi_port import nxdi_port_node
from nodes.ref_reader import ref_reader_node
from nodes.registration import registration_node
from nodes.validation import validation_node
from state import AgentState


def _route_after_validation(state: AgentState) -> str:
    if (state.get("validation_result") or {}).get("passed"):
        return "nxdi_port"
    return "end"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("arch", arch_node)
    graph.add_node("ref_reader", ref_reader_node)
    graph.add_node("codegen", codegen_node)
    graph.add_node("registration", registration_node)
    graph.add_node("validation", validation_node)
    graph.add_node("nxdi_port", nxdi_port_node)

    graph.add_edge(START, "arch")
    graph.add_edge("arch", "ref_reader")
    graph.add_edge("ref_reader", "codegen")
    graph.add_edge("codegen", "registration")
    graph.add_edge("registration", "validation")
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {
            "nxdi_port": "nxdi_port",
            "end": END,
        },
    )
    graph.add_edge("nxdi_port", END)

    return graph.compile()
