from pydantic import ValidationError
from backend.app.actions.schemas import ActionDecision, ActionError
from backend.app.actions.validation import proposal_context
from backend.app.agent.guards.context import active_runtime
from backend.app.agent.nodes.common import model_node
from backend.app.llm.gateway import ErrorCode, GatewayError
from backend.app.actions.diagnostics import MISSING, pydantic_diagnostic


def make_propose_action_node(gateway, service, token_sink, *, strict=False):
    @model_node("propose_action", validation_code=ErrorCode.ACTION_PROPOSAL_VALIDATION_ERROR, strict=strict)
    def propose_action(state):
        if state["status"] != "DRAFT_READY" or not state["analysis_draft"]:
            raise GatewayError(ErrorCode.ACTION_PROPOSAL_VALIDATION_ERROR)
        raw = MISSING
        try:
            raw = gateway.propose_action(proposal_context(state, service.registry))
            decision = ActionDecision.model_validate(raw)
            if not decision.action_required:
                return {"status": "DRAFT_READY"}
            issued = service.create_pending(decision.proposal, state)
        except (ActionError, ValidationError) as error:
            wrapped = GatewayError(ErrorCode.ACTION_PROPOSAL_VALIDATION_ERROR)
            detail = getattr(error, "action_proposal_diagnostic", None)
            if detail is None and isinstance(error, ValidationError):
                detail = pydantic_diagnostic(error, raw, schema=ActionDecision)
            wrapped.action_proposal_diagnostic = detail
            raise wrapped from None
        # Explicit trusted application callback, never State, ToolResult or model context.
        try:
            token_sink(issued.pending.action_id, issued.token)
        except Exception:
            service.reject(issued.pending.action_id, issued.token)
            raise GatewayError(ErrorCode.INTERNAL_ERROR) from None
        runtime = active_runtime.get()
        if runtime:
            runtime.event("ACTION_PROPOSED", action_id=issued.pending.action_id, tool_name=issued.pending.tool_name)
        return dict(pending_action=issued.pending.model_dump(mode="json"), approval_status="PENDING",
                    action_result=None, status="AWAITING_APPROVAL", stop_reason="AWAITING_APPROVAL")
    return propose_action
