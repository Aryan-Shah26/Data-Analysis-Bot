import logging
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.graph.nodes import (
    node_load,
    node_profile,
    node_plan,
    node_human_approval,
    node_execute,
    node_analyze,
    node_visualize,
)
from src.graph.edges import (
    edge_after_load,
    edge_after_plan,
    edge_after_execute,
    edge_after_analyze,
)

logger = logging.getLogger(__name__)


def build_graph():
    """
    Assembles and compiles the LangGraph agent graph.
    No checkpointer — state is managed by Streamlit session state.
    Human-in-the-loop is handled by splitting the graph into
    two runs: run_until_approval() and run_after_approval().
    """
    graph = StateGraph(AgentState)

    graph.add_node("load", node_load)
    graph.add_node("profile", node_profile)
    graph.add_node("plan", node_plan)
    graph.add_node("human_approval", node_human_approval)
    graph.add_node("execute", node_execute)
    graph.add_node("analyze", node_analyze)
    graph.add_node("visualize", node_visualize)

    graph.set_entry_point("load")

    graph.add_conditional_edges("load", edge_after_load, {
        "profile": "profile"
    })
    graph.add_edge("profile", "plan")
    graph.add_conditional_edges("plan", edge_after_plan, {
        "human_approval": "human_approval"
    })
    graph.add_edge("human_approval", "execute")
    graph.add_conditional_edges("execute", edge_after_execute, {
        "analyze": "analyze"
    })
    graph.add_conditional_edges("analyze", edge_after_analyze, {
        "visualize": "visualize"
    })
    graph.add_edge("visualize", END)

    # No checkpointer — avoids DataFrame serialization issues
    compiled = graph.compile()

    logger.info("Graph compiled successfully.")
    return compiled


def build_approval_graph():
    """
    Second graph — runs from human_approval onwards.
    Called after user approves the plan in Streamlit.
    """
    graph = StateGraph(AgentState)

    graph.add_node("human_approval", node_human_approval)
    graph.add_node("execute", node_execute)
    graph.add_node("analyze", node_analyze)
    graph.add_node("visualize", node_visualize)

    graph.set_entry_point("human_approval")
    graph.add_edge("human_approval", "execute")
    graph.add_conditional_edges("execute", edge_after_execute, {
        "analyze": "analyze"
    })
    graph.add_conditional_edges("analyze", edge_after_analyze, {
        "visualize": "visualize"
    })
    graph.add_edge("visualize", END)

    compiled = graph.compile()
    return compiled


# Singletons
agent_graph = build_graph()
approval_graph = build_approval_graph()