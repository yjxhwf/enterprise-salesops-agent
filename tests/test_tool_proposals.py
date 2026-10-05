from copy import deepcopy
import json
import socket
import ssl
from http.client import RemoteDisconnected

import pytest
from langchain_core.messages import AIMessage
from openai import APIConnectionError
import httpx

from backend.app.agent.nodes.select_tool import make_select_node
from backend.app.agent.nodes.execute_tool import make_execute_node
from backend.app.agent.state import initial_state
from backend.app.agent.tooling import ToolCatalog
from backend.app.data.database import session_factory
from backend.app.llm.diagnostics import connection_diagnostics
from backend.app.llm.provider import ProductionGateway
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import DATES, QUERY, goal, plan, selection
from tests.test_llm_gateway import inject, settings, offline
from tests.test_agent_investigation import scripted, run


def native_candidates(candidates):
    return AIMessage(content="", tool_calls=[{"name": c["tool_name"], "args": c["arguments"],
                                             "id": f"call-{i}", "type": "tool_call"}
                                            for i, c in enumerate(candidates)])


def planned_state():
    return {**initial_state(QUERY), "goal": goal(), "plan": plan(), "status": "PLANNED"}


@pytest.mark.parametrize("count", [1, 3])
def test_native_proposals_validate_then_execute_once(monkeypatch, seeded_engine, count):
    proposed = [selection(), selection("analyze_region_performance"), selection("analyze_product_performance")][:count]
    client, _ = inject(monkeypatch, [native_candidates(proposed)])
    with build_default_registry(session_factory(seeded_engine)) as registry:
        catalog = ToolCatalog(registry)
        calls = []
        original = registry.invoke
        def invoke(name, arguments):
            calls.append(name)
            return original(name, arguments)
        monkeypatch.setattr(registry, "invoke", invoke)
        state = planned_state()
        state.update(make_select_node(ProductionGateway(settings()), catalog, investigation=True)(state))
        assert state["status"] == "TOOL_SELECTED"
        assert state["tool_proposal_count"] == count
        assert state["selected_tool"]["tool_name"] == proposed[0]["tool_name"]
        assert [p["provider_order"] for p in state["deferred_tool_proposals"]] == list(range(1, count))
        assert calls == []
        state.update(make_execute_node(registry, catalog)(state))
        assert calls == [proposed[0]["tool_name"]]
        assert state["tool_call_count"] == 1
        assert json.loads(json.dumps(state)) == state
        assert len(client.requests) == 1


@pytest.mark.parametrize("bad", ["unknown", "write", "sql", "session", "date", "invalid_schema"])
def test_invalid_secondary_candidate_rejects_all(monkeypatch, seeded_engine, bad):
    second = selection("analyze_region_performance")
    if bad == "unknown": second["tool_name"] = "unknown_tool"
    elif bad in {"sql", "session"}: second["arguments"][bad] = "malicious-private-marker"
    elif bad == "date": second["arguments"]["start_date"] = "2026-07-01"
    elif bad == "invalid_schema": second["arguments"]["end_date"] = "not-a-date"
    inject(monkeypatch, [native_candidates([selection(), second])])
    with build_default_registry(session_factory(seeded_engine)) as registry:
        catalog = ToolCatalog(registry)
        if bad == "write":
            metadata, model = catalog._entries[second["tool_name"]]
            catalog._entries[second["tool_name"]] = (metadata.model_copy(update={"permission_level": "WRITE"}), model)
        def forbidden(*args, **kwargs):
            raise AssertionError("No candidate may execute")
        monkeypatch.setattr(registry, "invoke", forbidden)
        state = planned_state()
        state.update(make_select_node(ProductionGateway(settings()), catalog, investigation=True)(state))
        assert state["status"] == "ERROR"
        assert state["errors"][-1]["code"] == "TOOL_SELECTION_VALIDATION_ERROR"
        assert state["selected_tool"] is None and state["tool_call_count"] == 0
        assert "malicious-private-marker" not in json.dumps(state)


def test_deferred_proposals_are_not_a_queue(seeded_engine):
    candidates = [selection(), selection("query_orders"), selection("analyze_customer_performance")]
    fake = scripted(selections=[{"candidates": [{**c, "provider_order": i} for i, c in enumerate(candidates)]},
                                selection("analyze_region_performance"),
                                selection("analyze_product_performance", region="EAST_CHINA"),
                                selection("query_orders", region="EAST_CHINA", product_id="SKU-A12", limit=2)])
    state = run(fake, seeded_engine)
    assert state["status"] == "DRAFT_READY"
    assert [r["tool_name"] for r in state["tool_history"]][:2] == ["get_sales_overview", "analyze_region_performance"]
    selectors = [context for name, context in fake.calls if name == "select_tool"]
    assert len(selectors) == state["tool_call_count"] == 4
    assert selectors[1]["observations"] and selectors[1]["plan"] != selectors[0]["plan"]
    assert state["deferred_tool_proposals"] == []


@pytest.mark.parametrize("cause,category", [(TimeoutError("sensitive"), "timeout"),
    (socket.gaierror("sensitive"), "DNS"), (ConnectionResetError("sensitive"), "connection reset"),
    (ssl.SSLError("sensitive"), "TLS"), (RemoteDisconnected("sensitive"), "remote disconnect"),
    (RuntimeError("sensitive"), "unknown")])
def test_safe_connection_diagnostics(cause, category):
    error = APIConnectionError(request=httpx.Request("POST", "https://private.invalid", headers={"Authorization": "sensitive"}))
    error.__cause__ = cause
    result = connection_diagnostics(error)
    assert result["category"] == category
    assert result["underlying_cause_classes"] == [type(cause).__name__]
    assert "sensitive" not in json.dumps(result) and "private.invalid" not in json.dumps(result)
