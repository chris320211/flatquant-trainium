"""
LangGraph StateGraph definition.

Wires the five nodes in sequence:
  arch_node → ref_reader_node → codegen_node → registration_node → validation_node
"""

from langgraph.graph import END, START, StateGraph

from nodes.arch import arch_node
from nodes.codegen import codegen_node
from nodes.ref_reader import ref_reader_node
from nodes.registration import registration_node
from nodes.validation import validation_node
from state import AgentState


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("arch", arch_node)
    graph.add_node("ref_reader", ref_reader_node)
    graph.add_node("codegen", codegen_node)
    graph.add_node("registration", registration_node)
    graph.add_node("validation", validation_node)

    graph.add_edge(START, "arch")
    graph.add_edge("arch", "ref_reader")
    graph.add_edge("ref_reader", "codegen")
    graph.add_edge("codegen", "registration")
    graph.add_edge("registration", "validation")
    graph.add_edge("validation", END)

    return graph.compile()
