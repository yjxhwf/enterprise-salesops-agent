from backend.app.actions.store import ApprovalStore
from backend.app.actions.validation import validate_proposal, validate_tool
from backend.app.actions.execution import execute_write
from collections import OrderedDict
from time import perf_counter
from uuid import uuid4
from backend.app.observability.integration import bind_action, traced_decision, tool_start, tool_end, write_completed


class ApprovalService:
    def __init__(self, registry, write_session_factory, *, store=None, trace_approval_source="HUMAN"):
        self.registry = registry
        self.store = store if store is not None else ApprovalStore()
        self._write_session_factory = write_session_factory
        self._trace_collectors = OrderedDict()
        self.trace_approval_source = trace_approval_source if trace_approval_source in {"HUMAN", "TEST_HARNESS"} else "HUMAN"

    def create_pending(self, proposal, state):
        validated = validate_proposal(proposal, state, self.registry)
        scope = {k: v for k, v in state["goal"].items() if k in {"start_date", "end_date", "normalized_goal"}}
        issued = self.store.create_pending(validated, state["run_id"], scope)
        bind_action(self, issued)
        return issued

    @traced_decision
    def execute_approved_action(self, action_id):
        def execute(pending):
            validate_tool(self.registry, pending.tool_name, pending.arguments)
            call_id, started = str(uuid4()), perf_counter()
            tool_start("write", pending.tool_name, pending.arguments, "WRITE_APPROVAL_REQUIRED", call_id,
                       approval_existed=True, fingerprint_validated=True, write_attempts=1)
            try:
                result = execute_write(pending, self.registry, self._write_session_factory, self.store.clock)
            except Exception as error:
                tool_end("write", pending.tool_name, "WRITE_APPROVAL_REQUIRED", call_id, started, error=error)
                raise
            tool_end("write", pending.tool_name, "WRITE_APPROVAL_REQUIRED", call_id, started, result=result)
            write_completed(pending, result)
            return result
        return self.store.mark_executed(action_id, execute)

    @traced_decision
    def approve(self, action_id, token):
        self.store.approve(action_id, token)
        return self.execute_approved_action(action_id)

    @traced_decision
    def reject(self, action_id, token):
        return self.store.reject(action_id, token)

    def state_update(self, action_id):
        pending, result = self.store.snapshot(action_id)
        status = pending.approval_status
        return dict(pending_action=pending.model_dump(mode="json"), approval_status=status,
                    action_result=result, status={"PENDING":"AWAITING_APPROVAL", "APPROVED":"ACTION_APPROVED",
                    "REJECTED":"ACTION_REJECTED", "EXECUTED":"ACTION_EXECUTED", "FAILED":"ACTION_FAILED"}[status])
