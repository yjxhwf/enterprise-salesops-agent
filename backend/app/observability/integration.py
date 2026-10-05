"""Observation-only adapters. No callbacks, trace data, or summaries enter model state."""
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from uuid import uuid4

from backend.app.observability.collector import TraceCollector, active_trace, observe, trace_scope

_decision_active = ContextVar("salesops_trace_decision", default=False)


def observation(function):
    @wraps(function)
    def safe(*args, **kwargs):
        collector = active_trace.get()
        if collector is None:
            return None
        try:
            return function(collector, *args, **kwargs)
        except Exception:
            collector.trace.observability_errors += 1
            return None
    return safe


@observation
def runtime_event(c, node, event_type, fields):
    if event_type == "CONTEXT_MANIFEST":
        c.emit("CONTEXT_BUILT", node=node, context_id=fields.get("context_id"), fingerprint=fields.get("context_fingerprint"),
               chars=fields.get("projected_char_count"), included_evidence_count=len(fields.get("included_evidence_ids", [])),
               business_evidence_count=len(fields.get("included_business_evidence_ids", [])),
               policy_evidence_count=len(fields.get("included_policy_evidence_ids", [])),
               omitted_count=sum(fields.get("omitted_item_counts", {}).values()))
    elif event_type in {"LLM_RETRY", "TOOL_RETRY"}:
        c.emit(event_type, node=node, **fields)
    elif event_type in {"LOOP_DETECTED", "GUARD_STOP", "TOOL_TIMEOUT"}:
        c.emit("GUARD_STOP", node=node, status="RUNTIME_STOPPED", **fields)


@observation
def provider_identity(c, settings, model, schema_type=None):
    c.protect(getattr(model, "openai_api_key", None))
    if settings is not None:
        c.protect(settings.LLM_API_KEY, settings.EMBEDDING_API_KEY)
    c._model = getattr(model, "model_name", None) or (settings.LLM_MODEL if settings else None) or "OFFLINE"
    c._schema_type = schema_type


def error_fields(error, code=None):
    from backend.app.agent.guards.retry import transient_code
    from openai import APIError
    cause = error.__cause__ or error.__context__
    return dict(error_code=str(code or transient_code(error) or getattr(error, "code", None) or getattr(error, "reason", None)
                               or ("PROVIDER_REQUEST_ERROR" if isinstance(error, APIError) else "INTERNAL_ERROR")),
                exception_class=type(error).__name__, cause_class=type(cause).__name__ if cause else None,
                cause_error_code=str(cause.code) if cause and getattr(cause, "code", None) is not None else None,
                retryable=transient_code(error) is not None, safe_message="Operation failed; diagnostic codes only.")


@observation
def llm_start(c, runtime, attempt):
    c.emit("LLM_REQUEST_STARTED", node=runtime.node, attempt=attempt, model=getattr(c, "_model", "OFFLINE"),
           context_id=(runtime.current_manifest or {}).get("context_id"), status="STARTED")


@observation
def llm_end(c, runtime, attempt, started, response=None, error=None):
    elapsed = (perf_counter() - started) * 1000
    fields = dict(attempt=attempt, model=getattr(c, "_model", "OFFLINE"),
                  context_id=(runtime.current_manifest or {}).get("context_id"), latency_ms=elapsed)
    usage = getattr(response, "usage_metadata", None)
    keys = ("input_tokens", "output_tokens", "total_tokens")
    valid = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys)
    valid = valid and usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    fields["usage_available"] = bool(valid)
    if valid:
        fields.update({key: usage[key] for key in keys})
    if error:
        fields.update(error_fields(error))
    else:
        fields.update(schema_type=getattr(c, "_schema_type", None) or type(response).__name__, result_status="RECEIVED")
    c.emit("LLM_REQUEST_FAILED" if error else "LLM_REQUEST_SUCCEEDED", node=runtime.node,
           duration_ms=elapsed, status="FAILED" if error else "SUCCEEDED", **fields)


@observation
def tool_start(c, node, name, arguments, permission, call_id, attempt=1, **fields):
    if permission == "WRITE_APPROVAL_REQUIRED":
        fields["approval_source"] = getattr(c, "_approval_source", "HUMAN")
    else:
        fields["selection_id"] = getattr(c, "_selection_id", None)
    c.emit("TOOL_STARTED", node=node, tool_name=name, arguments=arguments, permission=permission,
           tool_call_id=call_id, attempt=attempt, **fields)


@observation
def tool_end(c, node, name, permission, call_id, started, result=None, error=None, **fields):
    elapsed = (perf_counter() - started) * 1000
    success = bool(result.success) if result is not None else False
    detail = error_fields(error) if error else dict(error_code=getattr(getattr(result, "error", None), "code", None),
        retryable=bool(getattr(getattr(result, "error", None), "retryable", False)))
    timeout = getattr(error, "reason", None) == "TOOL_TIMEOUT"
    c.emit("TOOL_TIMEOUT" if timeout else "TOOL_SUCCEEDED" if success else "TOOL_FAILED", node=node,
           duration_ms=elapsed, latency_ms=elapsed, status="SUCCEEDED" if success else "FAILED", success=success,
           permission=permission, tool_name=name, tool_call_id=call_id,
           evidence_count=len(getattr(result, "evidence_ids", [])), **detail, **fields)


@observation
def validation_error(c, node, error, code):
    from pydantic import ValidationError
    diagnostic = getattr(error, "diagnostic", None) or getattr(error, "structured_output_diagnostic", None) or {}
    fields = {k: diagnostic[k] for k in ("rule_id", "field_path", "reason") if k in diagnostic}
    if isinstance(error, ValidationError):
        item = error.errors(include_url=False, include_context=False, include_input=False)[0]
        # Unknown model-generated field names must never be retained.
        fields.update(rule_id=item["type"], field_path="schema", reason="SCHEMA_VALIDATION_FAILED")
    for item in diagnostic.get("violations", [])[:1]:
        fields.update({k: item[k] for k in ("rule_id", "field_path") if k in item})
    # Some frozen business validators expose only a contract-level code. Do not
    # invent an offending field or copy exception text to make diagnostics richer.
    fields.setdefault("rule_id", str(code))
    fields.setdefault("field_path", "NOT_PROVIDED")
    fields.setdefault("reason", "CONTRACT_REJECTED_DETAILS_NOT_PROVIDED")
    action_detail = getattr(error, "action_proposal_diagnostic", None)
    if action_detail:
        fields.update({k: action_detail[k] for k in ("field_path", "reason", "validation_rule", "validation_source", "expected_type", "actual_type", "safe_actual_shape", "evidence_related")})
        fields["rule_id"] = action_detail["validation_rule"]
    context = getattr(error, "context_diagnostic", None)
    if context:
        fields.update({k: context[k] for k in ("budget_chars", "mandatory_chars")})
    c.emit("CONTEXT_REJECTED" if context else "VALIDATION_FAILED", node=node, status="FAILED",
           **error_fields(error, code), **fields)


@observation
def node_finished(c, name, before, after, started, error=None):
    c._last_status, c._last_stop = after.get("status"), after.get("stop_reason")
    c.emit("NODE_FINISHED", node=name, status="FAILED" if error else after.get("status"),
           duration_ms=(perf_counter()-started)*1000, **(error_fields(error) if error else {}))
    if error:
        return
    selected = after.get("selected_tool")
    if name == "select_tool" and selected:
        c._selection_id = str(uuid4())
        c.emit("TOOL_SELECTED", node=name, selection_id=c._selection_id, tool_name=selected["tool_name"], arguments=selected["arguments"], permission="READ")
    if name == "execute_tool" and len(after.get("tool_history", [])) > len(before.get("tool_history", [])):
        from backend.app.agent.evidence import extract_evidence
        from backend.app.agent.policy_evidence import policy_catalog
        history = after["tool_history"][-1:]
        for kind, catalog in (("BUSINESS_DATA", extract_evidence(history)), ("POLICY", policy_catalog(history))):
            for eid in catalog:
                c.evidence(eid, kind, history[0]["tool_name"], node=name)
    if name == "review_progress" and len(after.get("observations", [])) > len(before.get("observations", [])):
        record = after["observations"][-1]
        for eid in record["evidence_ids"]:
            prior = c._evidence.get(eid, {})
            c.evidence(eid, "POLICY" if eid.startswith("POLICY::") else "BUSINESS_DATA", prior.get("source_tool", record["source_tool"]), node=name, selected=True)
        c.emit("PLAN_UPDATED", node=name, decision=record["decision"], reason="EVIDENCE_GAP_REPORTED" if record["remaining_evidence_gap"] else "SUFFICIENT_EVIDENCE",
               gap_chars=len(record["remaining_evidence_gap"] or ""), pending_step_count=sum(s["status"] == "PENDING" for s in after["plan"]["steps"]))
    if name == "synthesize" and after.get("analysis_draft"):
        draft = after["analysis_draft"]
        c.emit("SYNTHESIS_COMPLETED", node=name, finding_count=len(draft["findings"]), recommendation_count=len(draft["recommendations"]))
    for eid, usage in after.get("evidence_usage", {}).items():
        used = name == "synthesize" and usage.get("used_in_synthesis")
        proposed = name == "propose_action" and usage.get("used_in_action_proposal")
        if used or proposed:
            prior = c._evidence.get(eid, {})
            c.evidence(eid, "POLICY" if eid.startswith("POLICY::") else "BUSINESS_DATA", prior.get("source_tool"), node=name,
                       used_in_synthesis=bool(used), used_in_action_proposal=bool(proposed))


@observation
def bind_action(c, service, issued):
    c.protect(issued.token)
    service._trace_collectors[issued.pending.action_id] = c
    while len(service._trace_collectors) > 128:
        service._trace_collectors.popitem(last=False)


@observation
def approval_event(c, kind, pending):
    fields = dict(action_id=pending.action_id, tool_name=pending.tool_name,
                  permission="WRITE_APPROVAL_REQUIRED", approval_source=getattr(c, "_approval_source", "HUMAN"))
    c.emit("ACTION_STATUS" if kind == "ACTION_EXECUTED" else kind, node="approval", status=kind, **fields)
    if kind == "ACTION_PROPOSED":
        c.emit("APPROVAL_PENDING", node="approval", status="AWAITING_APPROVAL", **fields)
    if kind in {"ACTION_REJECTED", "ACTION_FAILED", "ACTION_EXECUTED"}:
        from backend.app.agent.guards.limits import StopReason
        c._last_status = kind
        c._last_stop = StopReason.TOOL_ERROR.value if kind == "ACTION_FAILED" else StopReason.COMPLETED.value


def traced_decision(function):
    @wraps(function)
    def call(service, action_id, *args, **kwargs):
        if _decision_active.get():
            return function(service, action_id, *args, **kwargs)
        try:
            collector = service._trace_collectors.get(action_id) or active_trace.get()
        except Exception:
            collector = None
        with trace_scope(collector):
            decision_token = _decision_active.set(True)
            observe("protect", *args)
            if collector:
                collector._approval_source = service.trace_approval_source
            try:
                return function(service, action_id, *args, **kwargs)
            except Exception as error:
                decision_failed(action_id, error)
                raise
            finally:
                if collector:
                    observe("finish")
                _decision_active.reset(decision_token)
    return call


@observation
def decision_failed(c, action_id, error):
    c.emit("WRITE_BLOCKED", node="approval", action_id=action_id, permission="WRITE_APPROVAL_REQUIRED",
           write_executed=False, **error_fields(error))


@observation
def write_completed(c, pending, result):
    # Called only when the store actually invoked its gated executor, not for cached results.
    if result.success:
        c.emit("ACTION_EXECUTED", node="write", action_id=pending.action_id, tool_name=pending.tool_name,
               permission="WRITE_APPROVAL_REQUIRED", approval_existed=True, fingerprint_validated=True,
               approval_source=getattr(c, "_approval_source", "HUMAN"), write_attempts=1, write_executed=True)
        for eid in result.evidence_ids:
            c.evidence(eid, "ACTION", pending.tool_name, node="write")


class TracedGraph:
    """Same graph inputs/outputs; retain at most 32 traces per graph instance.

    Callers keeping pending approvals should retain their collector explicitly.
    No global run registry, persistence, or model-visible state.
    """
    def __init__(self, graph, collector=None):
        self.graph, self.collector = graph, collector
        self.traces = OrderedDict()
        self.last_trace = None

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def with_config(self, *args, **kwargs):
        return TracedGraph(self.graph.with_config(*args, **kwargs), self.collector)

    def _new(self, state):
        try:
            collector = self.collector or TraceCollector(state.get("run_id"))
            if not collector.claim(state.get("run_id")):
                collector = TraceCollector(state.get("run_id"))
                collector.claim(state.get("run_id"))
            self.last_trace = collector
            self.traces[collector.trace.run_id] = collector
            while len(self.traces) > 32:
                self.traces.popitem(last=False)
            return collector
        except Exception:
            return None

    def invoke(self, input, *args, **kwargs):
        collector = self._new(input)
        with trace_scope(collector):
            try:
                return self.graph.invoke(input, *args, **kwargs)
            except BaseException:
                observe("finish", "ERROR", "MODEL_ERROR")
                raise
            finally:
                if collector and collector.trace.finished_at is None:
                    observe("finish")

    def stream(self, input, *args, **kwargs):
        collector = self._new(input)
        iterator = self.graph.stream(input, *args, **kwargs)
        completed = False
        failed = False
        try:
            while True:
                with trace_scope(collector):
                    try:
                        item = next(iterator)
                    except StopIteration:
                        completed = True
                        break
                yield item
        except Exception:
            failed = True
            raise
        finally:
            with trace_scope(collector):
                iterator.close()
                observe("finish", "ERROR" if failed else None if completed else "INTERRUPTED", "MODEL_ERROR" if failed else None)

    async def ainvoke(self, input, *args, **kwargs):
        collector = self._new(input)
        with trace_scope(collector):
            try:
                return await self.graph.ainvoke(input, *args, **kwargs)
            except BaseException:
                observe("finish", "ERROR", "MODEL_ERROR")
                raise
            finally:
                if collector and collector.trace.finished_at is None:
                    observe("finish")

    async def astream(self, input, *args, **kwargs):
        collector = self._new(input)
        iterator = self.graph.astream(input, *args, **kwargs)
        completed = False
        failed = False
        try:
            while True:
                with trace_scope(collector):
                    try:
                        item = await anext(iterator)
                    except StopAsyncIteration:
                        completed = True
                        break
                yield item
        except Exception:
            failed = True
            raise
        finally:
            with trace_scope(collector):
                await iterator.aclose()
                observe("finish", "ERROR" if failed else None if completed else "INTERRUPTED", "MODEL_ERROR" if failed else None)
