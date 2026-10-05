import time
from backend.app.observability.integration import TracedGraph
from backend.app.agent.guards.limits import RuntimeLimits
from backend.app.agent.guards.runtime import guarded_node, GuardedGateway, GuardedRegistry
from langgraph.graph import END, START, StateGraph

from backend.app.agent.nodes.plan import make_plan_node
from backend.app.agent.nodes.select_tool import make_select_node
from backend.app.agent.nodes.understand import make_understand_node
from backend.app.agent.nodes.execute_tool import make_execute_node
from backend.app.agent.nodes.review_progress import make_review_node
from backend.app.agent.nodes.synthesize import make_synthesize_node
from backend.app.agent.state import AgentState
from backend.app.agent.tooling import ToolCatalog
from backend.app.llm.gateway import ModelGateway


def route_after_understanding(state):
    return "plan" if state["status"] == "GOAL_UNDERSTOOD" else END


def build_agent_graph(model_gateway: ModelGateway, *, catalog=None, raise_unexpected_errors=False, runtime_limits=None, sleeper=time.sleep, trace_collector=None):
    """Task 03 planning-only entry point, preserved for isolated regression."""
    catalog = catalog if catalog is not None else ToolCatalog()
    model_gateway = GuardedGateway(model_gateway)
    builder = StateGraph(AgentState)
    def add_node(name, function):
        builder.add_node(name, guarded_node(name, function, limits=runtime_limits, sleeper=sleeper))
    add_node("understand_goal", make_understand_node(model_gateway, strict=raise_unexpected_errors))
    add_node("plan", make_plan_node(model_gateway, catalog, strict=raise_unexpected_errors))
    add_node("select_tool", make_select_node(model_gateway, catalog, strict=raise_unexpected_errors))
    builder.add_edge(START, "understand_goal")
    builder.add_conditional_edges("understand_goal", route_after_understanding, {"plan": "plan", END: END})
    builder.add_conditional_edges("plan", lambda state: "select_tool" if state["status"] == "PLANNED" else END,
                                  {"select_tool": "select_tool", END: END})
    builder.add_edge("select_tool", END)
    return TracedGraph(builder.compile().with_config(recursion_limit=max(50, (runtime_limits or RuntimeLimits()).max_agent_steps + 5)), trace_collector)


def build_investigation_graph(model_gateway, registry, *, raise_unexpected_errors=False, runtime_limits=None, sleeper=time.sleep,
                              action_service=None, approval_token_sink=None, trace_collector=None):
    """Caller owns the registry lifecycle; all business nodes share run-local guards."""
    catalog = ToolCatalog(registry)
    model_gateway = GuardedGateway(model_gateway)
    builder = StateGraph(AgentState)
    def add_node(name, function):
        builder.add_node(name, guarded_node(name, function, limits=runtime_limits, sleeper=sleeper))
    add_node("understand_goal", make_understand_node(model_gateway, strict=raise_unexpected_errors))
    add_node("plan", make_plan_node(model_gateway, catalog, strict=raise_unexpected_errors))
    add_node("select_tool", make_select_node(model_gateway, catalog, strict=raise_unexpected_errors, investigation=True))
    add_node("execute_tool", make_execute_node(GuardedRegistry(registry), catalog))
    add_node("review_progress", make_review_node(model_gateway, catalog, strict=raise_unexpected_errors))
    add_node("synthesize", make_synthesize_node(model_gateway, strict=raise_unexpected_errors))
    builder.add_edge(START, "understand_goal")
    builder.add_conditional_edges("understand_goal", route_after_understanding, {"plan": "plan", END: END})
    builder.add_conditional_edges("plan", lambda s: "select_tool" if s["status"] == "PLANNED" else END,
                                 {"select_tool": "select_tool", END: END})
    builder.add_conditional_edges("select_tool", lambda s: "execute_tool" if s["status"] == "TOOL_SELECTED" else END,
                                 {"execute_tool": "execute_tool", END: END})
    builder.add_conditional_edges("execute_tool", lambda s: "review_progress" if s["status"] == "TOOL_EXECUTED" else END,
                                 {"review_progress": "review_progress", END: END})
    builder.add_conditional_edges("review_progress", lambda s: (
        "select_tool" if s["status"] == "INVESTIGATING" else
        "synthesize" if s["status"] == "INVESTIGATION_COMPLETE" else END),
        {"select_tool": "select_tool", "synthesize": "synthesize", END: END})
    if action_service is not None:
        if not callable(approval_token_sink):
            raise ValueError("HITL requires a trusted application token delivery callback")
        from backend.app.agent.nodes.propose_action import make_propose_action_node
        add_node("propose_action", make_propose_action_node(model_gateway, action_service, approval_token_sink,
                                                           strict=raise_unexpected_errors))
        builder.add_conditional_edges("synthesize", lambda s: "propose_action" if s["status"] == "DRAFT_READY" else END,
                                     {"propose_action": "propose_action", END: END})
        builder.add_edge("propose_action", END)
    else:
        builder.add_edge("synthesize", END)
    return TracedGraph(builder.compile().with_config(recursion_limit=max(50, (runtime_limits or RuntimeLimits()).max_agent_steps + 5)), trace_collector)
