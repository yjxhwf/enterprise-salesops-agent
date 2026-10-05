from copy import deepcopy
from datetime import date
import json
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, APIStatusError, AuthenticationError, BadRequestError

from backend.app.agent.graph import build_agent_graph, build_investigation_graph
from backend.app.agent.guards.context import RuntimeStop, active_runtime
from backend.app.agent.guards.limits import RuntimeLimits, StopReason
from backend.app.agent.guards.loops import action_signature
from backend.app.agent.guards.retry import transient_code
from backend.app.agent.guards.runtime import Runtime, guarded_node, soft_call
from backend.app.agent.state import initial_state
from backend.app.data.database import session_factory
from backend.app.llm.gateway import GatewayError, ErrorCode
from backend.app.llm.provider import ProductionGateway
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import QUERY, goal, plan, selection, FakeGateway
from tests.test_agent_investigation import scripted, offline
from tests.test_llm_gateway import inject, native, settings


def connection():
    error = APIConnectionError(request=httpx.Request("POST", "https://provider.invalid"))
    error.__cause__ = httpx.RemoteProtocolError("secret sentinel")
    return error


def status_error(code):
    response = httpx.Response(code, request=httpx.Request("POST", "https://provider.invalid"))
    return APIStatusError("secret sentinel", response=response, body={"Authorization": "secret sentinel"})


def runtime(**limits):
    state, delays = initial_state(QUERY), []
    return Runtime(state, "select_tool", RuntimeLimits(**limits), delays.append), delays


@pytest.mark.parametrize("failures", [1, 2])
def test_connection_recovery_counts_each_attempt_and_no_real_sleep(failures):
    rt, delays = runtime()
    calls = []
    def operation():
        calls.append(1)
        if len(calls) <= failures:
            raise connection()
        return SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    rt.llm(operation)
    assert rt.state["llm_call_count"] == len(calls) == failures + 1
    assert rt.state["retry_count"] == failures
    assert delays == [0.5, 1.0][:failures]
    assert rt.state["token_usage"]["total_tokens"] == 15
    assert "secret sentinel" not in json.dumps(rt.state)


@pytest.mark.parametrize("error,expected", [(connection(), "PROVIDER_CONNECTION_ERROR"),
    (APITimeoutError(request=httpx.Request("POST", "https://provider.invalid")), "PROVIDER_TIMEOUT"),
    (status_error(429), "PROVIDER_RATE_LIMIT"), (status_error(500), "PROVIDER_SERVER_ERROR"),
    (status_error(503), "PROVIDER_SERVER_ERROR")])
def test_transient_classification_and_exhaustion(error, expected):
    assert transient_code(error) == expected
    rt, delays = runtime()
    def fail():
        raise error
    with pytest.raises(RuntimeStop) as caught:
        rt.llm(fail)
    assert caught.value.reason == "MODEL_RETRY_EXHAUSTED"
    assert rt.state["llm_call_count"] == 3 and rt.state["retry_count"] == 2
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize("error", [status_error(400), status_error(401), status_error(403), status_error(404),
    GatewayError(ErrorCode.MODEL_CONFIGURATION_ERROR), GatewayError(ErrorCode.MODEL_CAPABILITY_ERROR),
    GatewayError(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR), GatewayError(ErrorCode.TOOL_SELECTION_VALIDATION_ERROR),
    ValueError("semantic sentinel")])
def test_nontransient_never_retried(error):
    rt, delays = runtime()
    def fail():
        raise error
    with pytest.raises(type(error)):
        rt.llm(fail)
    assert rt.state["llm_call_count"] == 1 and rt.state["retry_count"] == 0 and delays == []


def test_llm_and_retry_budgets_prevent_extra_attempt():
    rt, delays = runtime(max_llm_calls=1)
    with pytest.raises(RuntimeStop, match="MAX_LLM_CALLS_EXCEEDED"):
        rt.llm(lambda: (_ for _ in ()).throw(connection()))
    assert rt.state["llm_call_count"] == 1 and rt.state["retry_count"] == 0 and not delays
    rt, _ = runtime(max_total_retries=1)
    with pytest.raises(RuntimeStop, match="MAX_RETRIES_EXCEEDED"):
        rt.llm(lambda: (_ for _ in ()).throw(connection()))
    assert rt.state["llm_call_count"] == 2 and rt.state["retry_count"] == 1


def test_total_retry_budget_accumulates_across_operations():
    rt, delays = runtime(max_total_retries=3)
    for _ in range(3):
        responses = iter([connection(), SimpleNamespace(usage_metadata=None)])
        def call():
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response
        rt.llm(call)
    with pytest.raises(RuntimeStop, match="MAX_RETRIES_EXCEEDED"):
        rt.llm(lambda: (_ for _ in ()).throw(connection()))
    assert rt.state["llm_call_count"] == 7 and rt.state["retry_count"] == 3


def test_usage_accounting_soft_limit_and_missing_usage():
    rt, _ = runtime(soft_token_budget=10)
    rt.llm(lambda: SimpleNamespace(usage_metadata=None))
    rt.llm(lambda: SimpleNamespace(usage_metadata={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}))
    assert rt.state["token_usage"] == dict(input_tokens=8, output_tokens=4, total_tokens=12,
                                          usage_available=False, reported_responses=1, missing_responses=1)
    with pytest.raises(RuntimeStop, match="TOKEN_BUDGET_EXCEEDED"):
        rt.llm(lambda: pytest.fail("No extra request"))
    assert rt.state["llm_call_count"] == 2


@pytest.mark.parametrize("usage", [{}, {"input_tokens": -1, "output_tokens": 4, "total_tokens": 3},
    {"input_tokens": 1, "output_tokens": 4, "total_tokens": 9}, {"input_tokens": True, "output_tokens": 4, "total_tokens": 5}])
def test_unreliable_usage_not_guessed(usage):
    rt, _ = runtime()
    rt.llm(lambda: SimpleNamespace(usage_metadata=usage))
    assert not rt.state["token_usage"]["usage_available"] and rt.state["token_usage"]["total_tokens"] == 0


class Registry:
    def __init__(self, *, permission="READ", errors=0, retryable=True, code="DATABASE_ERROR"):
        self.calls, self.errors, self.retryable, self.code = [], errors, retryable, code
        self.metadata = SimpleNamespace(permission_level=permission, timeout_seconds=8)
    def get_tool(self, name):
        return self.metadata
    def invoke(self, name, arguments):
        self.calls.append((name, arguments))
        error = SimpleNamespace(code=self.code, retryable=self.retryable) if len(self.calls) <= self.errors else None
        return SimpleNamespace(success=error is None, error=error)


def test_read_retry_counts_physical_invocations():
    rt, _ = runtime()
    registry = Registry(errors=1)
    assert rt.tool(registry, "read", {}).success
    assert len(registry.calls) == rt.state["tool_call_count"] == 2 and rt.state["retry_count"] == 1
    registry = Registry(errors=10)
    rt, _ = runtime()
    assert not rt.tool(registry, "read", {}).success
    assert len(registry.calls) == 2


@pytest.mark.parametrize("kwargs", [dict(permission="WRITE_APPROVAL_REQUIRED"), dict(retryable=False), dict(code="INTERNAL_ERROR")])
def test_tool_retry_permission_and_error_boundary(kwargs):
    rt, _ = runtime()
    registry = Registry(errors=10, **kwargs)
    assert not rt.tool(registry, "tool", {}).success
    assert len(registry.calls) == 1 and rt.state["retry_count"] == 0


def test_tool_budget_prevents_execution_and_retry():
    rt, _ = runtime(max_tool_calls=1)
    registry = Registry(errors=1)
    with pytest.raises(RuntimeStop, match="MAX_TOOL_CALLS_EXCEEDED"):
        rt.tool(registry, "read", {})
    assert len(registry.calls) == 1 and rt.state["retry_count"] == 0


def test_signature_normalization_and_repeated_action_cumulative():
    assert action_signature("read", {"date": date(2026, 8, 1), "a": 1}) == action_signature("read", {"a": 1, "date": "2026-08-01"})
    rt, _ = runtime()
    registry = Registry()
    for name, args in [("read", {}), ("other", {}), ("read", {"filter": 1}), ("read", {})]:
        rt.tool(registry, name, args)
    with pytest.raises(RuntimeStop, match="REPEATED_ACTION_LOOP"):
        rt.tool(registry, "read", {})
    assert len(registry.calls) == 4


def test_soft_timeout_returns_without_waiting_for_worker_or_retry():
    started, release, finished = Event(), Event(), Event()
    registry = Registry()
    registry.metadata.timeout_seconds = 0.02
    def blocked(name, args):
        registry.calls.append((name, args))
        started.set()
        release.wait(5)
        finished.set()
        return SimpleNamespace(success=True, error=None)
    registry.invoke = blocked
    rt, _ = runtime()
    try:
        with pytest.raises(RuntimeStop, match="TOOL_TIMEOUT"):
            rt.tool(registry, "read", {})
        assert started.is_set() and not finished.is_set()
        assert rt.state["tool_call_count"] == 1 and rt.state["retry_count"] == 0
    finally:
        release.set()
        assert finished.wait(2)


def test_node_budget_stops_before_gateway_and_preserves_state():
    state = initial_state(QUERY)
    fake = FakeGateway(goal(), plan(), selection())
    result = build_agent_graph(fake, runtime_limits=RuntimeLimits(max_agent_steps=1)).invoke(state)
    assert result["status"] == "RUNTIME_STOPPED" and result["stop_reason"] == "MAX_AGENT_STEPS_EXCEEDED"
    assert len(fake.calls) == 1 and result["agent_step_count"] == 1
    assert result["goal"] and result["run_id"] == state["run_id"] and result["analysis_draft"] is None
    assert json.loads(json.dumps(result)) == result


def test_stagnation_ignores_control_ids_but_resets_with_evidence(seeded_engine):
    with build_default_registry(session_factory(seeded_engine)) as registry:
        result = build_investigation_graph(scripted(), registry).invoke(initial_state(QUERY), config={"recursion_limit": 50})
    state = deepcopy(result)
    state.update(status="INVESTIGATING", last_progress_fingerprint=None, stagnant_cycle_count=0)
    state["plan"] = {"steps": [dict(step_id="one", objective="Investigate", expected_output="Evidence", status="PENDING")]}
    def unchanged(s):
        s["plan"]["steps"][0]["step_id"] += "x"
        return {"status": "INVESTIGATING"}
    node = guarded_node("review_progress", unchanged)
    state = node(state)
    state = node(state)
    assert state["stagnant_cycle_count"] == 1
    # Remove one record, then add it back: the new evidence resets stagnation.
    extra = state["tool_history"].pop()
    state = node(state)
    state["tool_history"].append(extra)
    state = node(state)
    assert state["stagnant_cycle_count"] == 0
    state = node(node(state))
    assert state["status"] == "RUNTIME_STOPPED" and state["stop_reason"] == "STAGNANT_LOOP"
    assert state["tool_history"] and state["observations"] and state["analysis_draft"] is None


def test_fault_injection_full_e2e_only_current_operation_retried(seeded_engine):
    fake = scripted()
    original = fake.select_investigation_tool
    calls, delays = [], []
    def select(*args):
        calls.append(1)
        if len(calls) == 3:
            raise connection()
        return original(*args)
    fake.select_investigation_tool = select
    with build_default_registry(session_factory(seeded_engine)) as registry:
        state = build_investigation_graph(fake, registry, sleeper=delays.append).invoke(initial_state(QUERY), config={"recursion_limit": 50})
    assert state["status"] == "DRAFT_READY" and state["stop_reason"] == "COMPLETED"
    assert state["llm_call_count"] == 12 and state["retry_count"] == 1 and delays == [0.5]
    assert state["tool_call_count"] == 4 and state["agent_step_count"] == 15
    assert [r["tool_name"] for r in state["tool_history"]] == ["get_sales_overview", "analyze_region_performance", "analyze_product_performance", "query_orders"]
    assert state["visited_nodes"].count("understand_goal") == 1
    assert next(e for e in state["runtime_events"] if e["event_type"] == "LLM_RETRY")["node"] == "select_tool"


def test_actual_production_transport_retry_and_usage_before_schema(monkeypatch):
    responses = [connection(), native("GoalUnderstanding", goal()), native("Plan", {"steps": []})]
    responses[1].usage_metadata = {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
    responses[2].usage_metadata = {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12}
    client, _ = inject(monkeypatch, responses)
    delays = []
    state = build_agent_graph(ProductionGateway(settings()), sleeper=delays.append).invoke(initial_state(QUERY))
    assert state["status"] == "ERROR" and state["errors"][-1]["code"] == "MODEL_OUTPUT_VALIDATION_ERROR"
    assert state["llm_call_count"] == len(client.requests) == 3 and state["retry_count"] == 1
    assert state["token_usage"]["total_tokens"] == 27 and delays == [0.5]


def test_production_budget_zero_means_zero_http(monkeypatch):
    client, _ = inject(monkeypatch, [])
    state = build_agent_graph(ProductionGateway(settings()), runtime_limits=RuntimeLimits(max_llm_calls=0)).invoke(initial_state(QUERY))
    assert state["stop_reason"] == "MAX_LLM_CALLS_EXCEEDED" and not client.requests


@pytest.mark.parametrize("method", ["understand_goal", "create_plan", "select_investigation_tool", "review_progress", "synthesize"])
def test_every_llm_node_uses_shared_retry(seeded_engine, method):
    fake = scripted()
    original = getattr(fake, method)
    attempts, delays = [], []
    def operation(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise connection()
        return original(*args, **kwargs)
    setattr(fake, method, operation)
    with build_default_registry(session_factory(seeded_engine)) as registry:
        state = build_investigation_graph(fake, registry, sleeper=delays.append).invoke(initial_state(QUERY))
    assert state["status"] == "DRAFT_READY" and state["llm_call_count"] == 12
    assert state["tool_call_count"] == 4 and state["retry_count"] == 1 and delays == [0.5]


@pytest.mark.parametrize("limit,reason", [(dict(max_llm_calls=6), "MAX_LLM_CALLS_EXCEEDED"),
    (dict(max_tool_calls=2), "MAX_TOOL_CALLS_EXCEEDED"), (dict(max_total_retries=0), "MAX_RETRIES_EXCEEDED")])
def test_guard_stops_graph_and_preserves_collected_evidence(seeded_engine, limit, reason):
    fake = scripted()
    if "max_total_retries" in limit:
        original = fake.select_investigation_tool
        calls = []
        def selection_with_failure(*args):
            calls.append(1)
            if len(calls) == 3:
                raise connection()
            return original(*args)
        fake.select_investigation_tool = selection_with_failure
    with build_default_registry(session_factory(seeded_engine)) as registry:
        state = build_investigation_graph(fake, registry, runtime_limits=RuntimeLimits(**limit),
                                         sleeper=lambda delay: pytest.fail("No retry allowed")).invoke(initial_state(QUERY))
    assert state["status"] == "RUNTIME_STOPPED" and state["stop_reason"] == reason
    assert state["tool_history"] and state["observations"] and state["plan"]
    assert state["analysis_draft"] is None and json.loads(json.dumps(state)) == state
    assert state["tool_call_count"] == 2


def test_step_guard_precedes_default_graph_recursion_limit(seeded_engine):
    from tests.test_agent_investigation import review_at
    fake = scripted(selections=[selection()] * 20, reviews=[review_at(0)] * 20)
    limits = RuntimeLimits(max_llm_calls=100, max_tool_calls=100, max_repeated_tool_args=100, max_stagnant_cycles=100)
    with build_default_registry(session_factory(seeded_engine)) as registry:
        state = build_investigation_graph(fake, registry, runtime_limits=limits).invoke(initial_state(QUERY))
    assert state["stop_reason"] == "MAX_AGENT_STEPS_EXCEEDED" and state["agent_step_count"] == 30


def test_repeated_and_stagnant_guards_in_real_graph(seeded_engine):
    from tests.test_agent_investigation import review_at
    for limits, reason, tool_count in [(RuntimeLimits(), "REPEATED_ACTION_LOOP", 2),
                                       (RuntimeLimits(max_repeated_tool_args=8), "STAGNANT_LOOP", 3)]:
        fake = scripted(selections=[selection()] * 10, reviews=[review_at(0)] * 10)
        with build_default_registry(session_factory(seeded_engine)) as registry:
            state = build_investigation_graph(fake, registry, runtime_limits=limits).invoke(initial_state(QUERY))
        assert state["stop_reason"] == reason and state["tool_call_count"] == tool_count
        assert state["analysis_draft"] is None


def test_deferred_proposals_not_counted_as_actions(seeded_engine):
    fake = scripted(selections=[{"candidates": [dict(**selection(), provider_order=0),
                                                dict(**selection("analyze_region_performance"), provider_order=1)]}])
    with build_default_registry(session_factory(seeded_engine)) as registry:
        state = build_investigation_graph(fake, registry, runtime_limits=RuntimeLimits(max_agent_steps=5)).invoke(initial_state(QUERY))
    assert state["tool_call_count"] == 1 and sum(state["repeated_action_counts"].values()) == 1


def test_semantic_and_evidence_failures_never_retry(seeded_engine):
    from tests.test_agent_investigation import review_at
    for invalid in ("semantic", "evidence"):
        def review(**context):
            result = review_at(0)(**context)
            if invalid == "semantic":
                result["next_objective"] = "Revenue increased YoY"
            else:
                result["selected_evidence_ids"] = ["invented"]
            return result
        delays = []
        with build_default_registry(session_factory(seeded_engine)) as registry:
            state = build_investigation_graph(scripted(reviews=[review]), registry, sleeper=delays.append).invoke(initial_state(QUERY))
        assert state["status"] == "ERROR" and state["stop_reason"] == "MODEL_ERROR"
        assert state["llm_call_count"] == 4 and state["retry_count"] == 0 and not delays


def test_configuration_failure_is_not_a_provider_attempt(monkeypatch):
    client, _ = inject(monkeypatch, [])
    state = build_agent_graph(ProductionGateway(settings(LLM_API_KEY=""))).invoke(initial_state(QUERY))
    assert state["stop_reason"] == "MODEL_ERROR" and state["llm_call_count"] == 0 and not client.requests


def test_timeout_stopped_state_preserves_history():
    state = initial_state(QUERY)
    state["observations"] = [{"evidence_ids": ["existing"]}]
    def node(s):
        active_runtime.get().stop(StopReason.TOOL_TIMEOUT)
    result = guarded_node("execute_tool", node)(state)
    assert result["stop_reason"] == "TOOL_TIMEOUT" and result["observations"] == state["observations"]
    assert result["runtime_events"][-1]["event_type"] == "TOOL_TIMEOUT"
