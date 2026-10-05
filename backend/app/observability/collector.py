"""Bounded process-local metadata, with fail-open collection and fail-closed export."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from threading import RLock
from time import perf_counter
from uuid import UUID, uuid4

from backend.app.observability.schemas import RunTrace, TraceEvent

MAX_TRACE_EVENTS = 500
active_trace = ContextVar("salesops_trace", default=None)
EVENTS = frozenset("""RUN_STARTED RUN_FINISHED NODE_STARTED NODE_FINISHED
LLM_REQUEST_STARTED LLM_REQUEST_SUCCEEDED LLM_REQUEST_FAILED LLM_RETRY
CONTEXT_BUILT CONTEXT_REJECTED TOOL_SELECTED TOOL_STARTED TOOL_SUCCEEDED TOOL_FAILED
TOOL_TIMEOUT TOOL_RETRY EVIDENCE_REGISTERED EVIDENCE_SELECTED EVIDENCE_USED PLAN_UPDATED
SYNTHESIS_COMPLETED ACTION_PROPOSED APPROVAL_PENDING ACTION_APPROVED ACTION_REJECTED
ACTION_EXECUTED ACTION_FAILED ACTION_STATUS WRITE_BLOCKED GUARD_STOP VALIDATION_FAILED""".split())
CRITICAL = EVENTS - {"EVIDENCE_REGISTERED", "EVIDENCE_SELECTED", "EVIDENCE_USED", "CONTEXT_BUILT", "TOOL_SELECTED"}
NODES = frozenset("understand_goal plan select_tool execute_tool review_progress synthesize propose_action approval write".split())
TOOLS = frozenset("get_sales_overview analyze_region_performance analyze_product_performance analyze_customer_performance query_orders search_sales_policy create_crm_task create_business_alert".split())
NUMBERS = frozenset("""attempt input_tokens output_tokens total_tokens latency_ms chars
included_evidence_count business_evidence_count policy_evidence_count omitted_count
evidence_count delay_ms write_attempts pending_step_count gap_chars finding_count
recommendation_count budget_chars mandatory_chars""".split())
FLAGS = frozenset("usage_available success retryable approval_existed fingerprint_validated selected used_in_synthesis used_in_action_proposal write_executed evidence_related".split())
CODES = frozenset("error_code cause_error_code exception_class cause_class schema_type result_status permission kind reason rule_id field_path operation approval_source decision stop_reason validation_rule validation_source expected_type actual_type".split())
IDS = frozenset("context_id tool_call_id action_id selection_id".split())
SENSITIVE = re.compile(r"(?i)authorization|bearer\s|basic\s|sk-[\w-]+|api[_ -]?key|approval[_ -]?token|token[_ -]?hash|password|private.?key")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def observe(method, *args, collector=None, **kwargs):
    """All observer failures stay on this side of the business boundary."""
    target = collector if collector is not None else active_trace.get()
    if target is None:
        return None
    try:
        return getattr(target, method)(*args, **kwargs)
    except Exception:
        try:
            target.trace.observability_errors += 1
        except Exception:
            pass
        return None


@contextmanager
def trace_scope(collector):
    token = active_trace.set(collector)
    try:
        yield collector
    finally:
        active_trace.reset(token)


class TraceCollector:
    def __init__(self, run_id=None, *, max_events=MAX_TRACE_EVENTS):
        if type(max_events) is not int or max_events < 2:
            raise ValueError("At least two lifecycle events are required")
        self.max_events = min(max_events, MAX_TRACE_EVENTS)
        self.trace = RunTrace(str(UUID(run_id)) if run_id else str(uuid4()), utc_now())
        self._start = perf_counter()
        self._sequence = 0
        self._lock = RLock()
        self._secrets = set()
        self._evidence = {}  # bounded metadata only, never evidence bodies
        self._nodes = []
        self._llm = dict(attempts=0, retries=0, input_tokens=0, output_tokens=0, total_tokens=0, missing_usage=0)
        self._tools = dict(read_tool_calls=0, write_tool_calls=0)
        self._approval = dict(action_proposed=False, approval_required=False, write_executed=False)
        self._errors = dict(count=0, last_error_code=None)
        self._last_status = None
        self._last_stop = None
        self._claimed = False
        self.emit("RUN_STARTED", status="RUNNING")

    def claim(self, run_id):
        with self._lock:
            if self._claimed or self.trace.finished_at is not None:
                return False
            self._claimed = True
            self.trace.run_id = str(UUID(run_id))
            self._start = perf_counter()
            self.trace.started_at = utc_now()
            for event in self.trace.events:
                event.run_id = self.trace.run_id
                event.timestamp = self.trace.started_at
            return True

    def protect(self, *values):
        for value in values:
            if hasattr(value, "get_secret_value"):
                value = value.get_secret_value()
            if isinstance(value, str) and value:
                self._secrets.update((value, hashlib.sha256(value.encode()).hexdigest()))

    def _text(self, value, limit=160):
        if not isinstance(value, str) or len(value) > limit or SENSITIVE.search(value):
            return None
        if any(secret in value for secret in self._secrets):
            return None
        return value

    def _code(self, value):
        text = value if isinstance(value, str) and len(value) <= 160 and not any(s in value for s in self._secrets) else None
        return text if text and re.fullmatch(r"[A-Za-z0-9_.$\[\]<>:-]+", text) else None

    def _metadata(self, metadata):
        result = {}
        for key, value in metadata.items():
            if key == "safe_actual_shape":
                from backend.app.actions.diagnostics import validate_shape
                result[key] = json.loads(json.dumps(validate_shape(value), allow_nan=False))
            elif key in NUMBERS and type(value) in {int, float} and math.isfinite(value) and value >= 0:
                result[key] = value
            elif key in FLAGS and type(value) is bool:
                result[key] = value
            elif key in CODES:
                result[key] = self._code(value)
            elif key in IDS:
                try:
                    result[key] = str(UUID(value))
                except (ValueError, TypeError, AttributeError):
                    pass
            elif key in {"tool_name", "source_tool"} and value in TOOLS:
                result[key] = value
            elif key == "model":
                text = self._text(value, 120)
                if text and re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
                    result[key] = text
            elif key == "fingerprint" and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                result[key] = self._text(value)
            elif key == "evidence_id":
                text = self._text(value, 240)
                if text and re.fullmatch(r"[\w:./-]+", text):
                    result[key] = text
            elif key == "arguments" and isinstance(value, dict):
                # No free-form query, reason, suggested action, headers, or arbitrary keys.
                safe = {}
                for name in ("start_date", "end_date"):
                    v = value.get(name)
                    if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                        safe[name] = v
                for name in ("limit", "top_k", "offset"):
                    v = value.get(name)
                    if type(v) is int and 0 <= v <= 10000:
                        safe[name] = v
                previous = value.get("omitted_field_count")
                safe["omitted_field_count"] = previous if type(previous) is int and previous >= 0 else len(value) - len(safe)
                result[key] = safe
            elif key == "safe_message" and value == "Operation failed; diagnostic codes only.":
                result[key] = value
        return result

    def emit(self, event_type, *, node=None, duration_ms=None, status=None, **metadata):
        if event_type not in EVENTS:
            raise ValueError("Unknown trace event")
        safe = self._metadata(metadata)  # validate before retaining anything
        duration = duration_ms if type(duration_ms) in {int, float} and math.isfinite(duration_ms) and duration_ms >= 0 else None
        with self._lock:
            self._sequence += 1
            event = TraceEvent(str(uuid4()), self.trace.run_id, self._sequence, utc_now(), event_type,
                               node if node in NODES else None, duration, self._code(status), safe)
            self._aggregate(event)
            if len(self.trace.events) >= self.max_events:
                self.trace.trace_truncated = True
                self.trace.dropped_events += 1
                # Always preserve start and the latest finish; prefer dropping verbose evidence.
                removable = next((i for i, e in enumerate(self.trace.events) if e.event_type not in CRITICAL), None)
                if removable is None and event_type in CRITICAL:
                    removable = next((i for i, e in enumerate(self.trace.events) if e.event_type not in {"RUN_STARTED", "RUN_FINISHED"}), None)
                if removable is None:
                    return
                self.trace.events.pop(removable)
            self.trace.events.append(event)

    def _aggregate(self, event):
        kind, m = event.event_type, event.safe_metadata
        if kind == "NODE_STARTED" and len(self._nodes) < MAX_TRACE_EVENTS:
            self._nodes.append(event.node)
        if kind == "LLM_REQUEST_STARTED":
            self._llm["attempts"] += 1
        if kind == "LLM_RETRY":
            self._llm["retries"] += 1
        if kind in {"LLM_REQUEST_SUCCEEDED", "LLM_REQUEST_FAILED"}:
            if m.get("usage_available"):
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    self._llm[key] += m.get(key, 0)
            else:
                self._llm["missing_usage"] += 1
        if kind == "TOOL_STARTED":
            self._tools["write_tool_calls" if m.get("permission") == "WRITE_APPROVAL_REQUIRED" else "read_tool_calls"] += 1
        if kind == "ACTION_PROPOSED":
            self._approval["action_proposed"] = True
        if kind == "APPROVAL_PENDING":
            self._approval["approval_required"] = True
        if kind == "ACTION_EXECUTED" and m.get("write_executed"):
            self._approval["write_executed"] = True
        if kind in {"LLM_REQUEST_FAILED", "TOOL_FAILED", "TOOL_TIMEOUT", "VALIDATION_FAILED", "ACTION_FAILED", "WRITE_BLOCKED", "CONTEXT_REJECTED"}:
            self._errors["count"] += 1
            self._errors["last_error_code"] = m.get("error_code")

    def evidence(self, eid, kind, source, *, node=None, selected=False, used_in_synthesis=False, used_in_action_proposal=False):
        safe = self._metadata(dict(evidence_id=eid, kind=kind, source_tool=source))
        eid = safe.get("evidence_id")
        if not eid:
            return
        if eid not in self._evidence:
            if len(self._evidence) >= MAX_TRACE_EVENTS:
                self.trace.trace_truncated = True
                return
            self._evidence[eid] = safe
            self.emit("EVIDENCE_REGISTERED", node=node, **safe)
        if selected or used_in_synthesis or used_in_action_proposal:
            self.emit("EVIDENCE_SELECTED" if selected else "EVIDENCE_USED", node=node, **safe,
                      selected=selected, used_in_synthesis=used_in_synthesis, used_in_action_proposal=used_in_action_proposal)

    def finish(self, status=None, stop_reason=None):
        with self._lock:
            self.trace.final_status = self._code(status or self._last_status) or "UNKNOWN"
            self.trace.stop_reason = self._code(stop_reason or self._last_stop)
            self.trace.finished_at = utc_now()
            self.trace.duration_ms = (perf_counter() - self._start) * 1000
            self.trace.events = [e for e in self.trace.events if e.event_type != "RUN_FINISHED"]
            self.emit("RUN_FINISHED", status=self.trace.final_status, stop_reason=self.trace.stop_reason, duration_ms=self.trace.duration_ms)

    def to_dict(self):
        # Export is intentionally NOT fail-open: no raw fallback on sanitization failure.
        with self._lock:
            self.trace.llm_totals = dict(self._llm)
            self.trace.tool_totals = dict(self._tools)
            self.trace.evidence_totals = {f"{k.lower()}_count": sum(v.get("kind") == k for v in self._evidence.values()) for k in ("BUSINESS_DATA", "POLICY", "ACTION")}
            self.trace.approval_summary = dict(self._approval)
            self.trace.error_summary = dict(self._errors)
            self.trace.summary = dict(nodes_visited=list(self._nodes), llm_attempts=self._llm["attempts"], llm_retries=self._llm["retries"],
                **{k: self._llm[k] for k in ("input_tokens", "output_tokens", "total_tokens")}, usage_complete=self._llm["missing_usage"] == 0,
                **self._tools, evidence_count=len(self._evidence), policy_evidence_count=self.trace.evidence_totals["policy_count"],
                business_evidence_count=self.trace.evidence_totals["business_data_count"], **self._approval,
                final_status=self.trace.final_status, stop_reason=self.trace.stop_reason, duration_ms=self.trace.duration_ms)
            data = asdict(self.trace)
            for event in data["events"]:
                event["safe_metadata"] = self._metadata(event["safe_metadata"])
            text = json.dumps(data, ensure_ascii=False, allow_nan=False)
            if any(secret in text for secret in self._secrets):
                raise ValueError("Trace export rejected sensitive data")
            return json.loads(text)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def safe_summary(self):
        return self.to_dict()["summary"]
