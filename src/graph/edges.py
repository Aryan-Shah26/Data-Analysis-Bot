from src.graph.state import AgentState


def edge_after_load(state: AgentState) -> str:
    return "profile"


def edge_after_plan(state: AgentState) -> str:
    return "human_approval"


def edge_after_execute(state: AgentState) -> str:
    return "analyze"


def edge_after_analyze(state: AgentState) -> str:
    return "visualize"