from backend.app.agent.guards.runtime import runtime_defaults
from typing import TypedDict
from uuid import UUID, uuid4

from backend.app.agent.schemas import AgentStatus


class AgentState(TypedDict):
    context_manifests: list[dict]
    evidence_usage: dict
    run_id: str
    user_query: str
    goal: dict | None
    plan: dict | None
    current_step_index: int
    selected_tool: dict | None
    status: str
    errors: list[dict]
    llm_call_count: int
    visited_nodes: list[str]
    tool_call_count: int
    tool_proposal_count: int
    deferred_tool_proposals: list[dict]
    latest_tool_result: dict | None
    tool_history: list[dict]
    observations: list[dict]
    completed_steps: list[dict]
    analysis_draft: dict | None
    agent_step_count: int
    retry_count: int
    repeated_action_counts: dict[str, int]
    stagnant_cycle_count: int
    stop_reason: str | None
    runtime_events: list[dict]
    token_usage: dict
    last_action_signature: str | None
    last_progress_fingerprint: str | None
    pending_action: dict | None
    approval_status: str | None
    action_result: dict | None


def initial_state(user_query: str, *, run_id: str | None = None) -> AgentState:
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be nonempty text")
    return AgentState(context_manifests=[], evidence_usage={}, run_id=str(UUID(run_id)) if run_id else str(uuid4()), user_query=user_query,
                      goal=None, plan=None, current_step_index=0, selected_tool=None,
                      status=AgentStatus.RECEIVED.value, errors=[], llm_call_count=0, visited_nodes=[],
                      tool_call_count=0, tool_proposal_count=0, deferred_tool_proposals=[],
                      latest_tool_result=None, tool_history=[], observations=[],
                      completed_steps=[], analysis_draft=None, pending_action=None,
                      approval_status=None, action_result=None, **runtime_defaults())
