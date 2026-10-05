from copy import deepcopy
import json

from langchain_core.messages import AIMessage

from backend.app.agent.guards.context import build_llm_context, compact_facts, active_runtime
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.facts import canonical_facts
from backend.app.agent.graph import build_investigation_graph
from backend.app.agent.state import initial_state
from backend.app.data.database import session_factory
from backend.app.llm.provider import ProductionGateway
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import QUERY, goal, plan, selection
from tests.test_agent_investigation import scripted, run, review_at, draft_from_evidence, offline
from tests.test_canonical_evidence import history
from tests.test_runtime_guards import connection


def test_understanding_and_planning_are_minimal():
    assert build_llm_context("understand_goal", user_query=QUERY, history=[{"private": 1}]) == {"user_query": QUERY}
    payload = build_llm_context("plan", goal=goal(), capabilities=[dict(name="read", description="Read", input_schema={"private": 1}, permission_level="READ")])
    assert set(payload) == {"goal", "capabilities"}
    assert "input_schema" not in str(payload)


def test_compact_fact_deduplication_and_business_labels(history):
    facts = canonical_facts(extract_evidence(history))
    compact = compact_facts(facts + facts)
    assert sum(len(b["facts"]) for b in compact) == len(facts)
    text = json.dumps(compact, ensure_ascii=False)
    for forbidden in ("internal_reference", "current_value", "display_value", "avg_discount", "discount_rate", "json_pointer"):
        assert forbidden not in text
    assert "price_realization_ratio" in text and "discount_depth" in text


def test_review_latest_all_but_prior_only_selected(history):
    catalog = extract_evidence(history)
    selected = list(extract_evidence(history[:1]))
    latest = extract_evidence(history[-1:])
    observation = dict(evidence_ids=selected, facts=[], plan_before={"private": "sentinel"})
    original = deepcopy(history)
    payload = build_llm_context("review_progress", goal=goal(), plan=plan(), step=plan()["steps"][0], history=history, observations=[observation])
    visible = list(latest)[:10]
    assert {b["evidence_id"] for b in payload["latest_evidence"]} == set(visible)
    assert {b["evidence_id"] for b in payload["prior_selected_evidence"]} == set(selected)
    assert sum(len(b["facts"]) for b in payload["latest_evidence"]) == len(canonical_facts(latest, visible))
    assert set(catalog) - latest.keys() - set(selected)
    assert history == original
    for forbidden in ("internal_reference", "tool_history", "runtime_events", "current_value", "plan_before", "private"):
        assert forbidden not in json.dumps(payload)
    assert payload == build_llm_context("review_progress", goal=goal(), plan=plan(), step=plan()["steps"][0], history=history, observations=[observation])


def test_selector_and_synthesis_keep_selected_facts_without_history(seeded_engine):
    state = run(scripted(), seeded_engine)
    original = deepcopy(state)
    current = dict(objective="Next", expected_output="Evidence", status="PENDING", step_id="current")
    view = build_llm_context("select_tool", goal=state["goal"], plan={"steps": state["plan"]["steps"] + [current]}, step=current, observations=state["observations"])
    assert view["pending_objectives"] == [] and "status" not in view["current_step"]
    assert {b["evidence_id"] for b in view["selected_evidence"]} == set(state["observations"][-1]["evidence_ids"])
    synthesis = build_llm_context("synthesize", goal=state["goal"], observations=state["observations"], history=state["tool_history"])
    ids = {b["evidence_id"] for b in synthesis["selected_evidence"] + synthesis["executive_facts"]}
    expected = {eid for o in state["observations"] for eid in o["evidence_ids"]}
    assert expected <= ids
    rendered = [f["rendered_text"] for b in synthesis["selected_evidence"] + synthesis["executive_facts"] for f in b["facts"]]
    assert len(rendered) == len(set(rendered))
    assert len(synthesis["completed_objectives"]) == 4
    assert state == original and state["tool_history"][0]["data"] and state["observations"][0]["facts"]
    assert json.loads(json.dumps(state)) == state


class ProjectedClient:
    """Installed ProductionGateway path with deterministic native responses, no network."""
    def __init__(self, fail_selection=False):
        self.payloads, self.selections, self.fail_selection = [], 0, fail_selection
    def bind_tools(self, tools, *, tool_choice, **kwargs):
        self.choice = tool_choice
        return self
    def invoke(self, messages, config):
        payload = json.loads(messages[1][1])
        state = active_runtime.get().state
        self.payloads.append((active_runtime.get().node, deepcopy(payload)))
        if self.choice == "GoalUnderstanding":
            name, args = self.choice, goal()
        elif self.choice == "Plan":
            name, args = self.choice, plan()
        elif self.choice == "required":
            self.selections += 1
            if self.fail_selection and self.selections == 3:
                raise connection()
            index = state["tool_call_count"]
            choice = [selection(), selection("analyze_region_performance"), selection("analyze_product_performance"), selection("query_orders", limit=2)][index]
            name, args = choice["tool_name"], choice["arguments"]
        elif self.choice == "ProgressReview":
            index = state["tool_call_count"] - 1
            args = review_at(index, complete=index == 3)(latest_result=state["latest_tool_result"], history=state["tool_history"])
            if index == 2:
                catalog = extract_evidence(state["tool_history"][-1:])
                args["selected_evidence_ids"] = [eid for eid, e in catalog.items() if e["fact_key"] in {"SKU-A12", "SKU-B07", "SKU-C03"}]
            name = self.choice
        else:
            name, args = self.choice, draft_from_evidence(observations=state["observations"])
        return AIMessage(content="", tool_calls=[dict(name=name, args=args, id="native")],
                         usage_metadata=dict(input_tokens=100, output_tokens=20, total_tokens=120))


def projected_run(engine, fail_selection=False):
    client, delays = ProjectedClient(fail_selection), []
    gateway = ProductionGateway()
    gateway._model = client
    with build_default_registry(session_factory(engine)) as registry:
        state = build_investigation_graph(gateway, registry, sleeper=delays.append).invoke(initial_state(QUERY))
    return state, client, delays


def test_four_cycle_projection_growth_and_golden_retention(seeded_engine):
    state, client, _ = projected_run(seeded_engine)
    assert state["status"] == "DRAFT_READY"
    reviews = [p for n, p in client.payloads if n == "review_progress"]
    assert len(reviews) == 4
    for i, p in enumerate(reviews):
        latest = extract_evidence(state["tool_history"][i:i+1])
        prior = {eid for o in state["observations"][:i] for eid in o["evidence_ids"]} - latest.keys()
        assert {b["evidence_id"] for b in p["prior_selected_evidence"]} == prior
        assert {b["evidence_id"] for b in p["latest_evidence"]} == set(latest)
        assert set(p) == {"goal", "current_step", "pending_objectives", "latest_source_tool", "latest_evidence", "prior_selected_evidence", "result_digest", "runtime_budget"}
    final = client.payloads[-1][1]
    product_ids = {b["subject"]["id"] for b in final["selected_evidence"]}
    assert {"SKU-A12", "SKU-B07", "SKU-C03"} <= product_ids
    for node, payload in client.payloads:
        for field in ("tool_history", "runtime_events", "internal_reference", "plan_before", "current_value"):
            assert field not in json.dumps(payload)
    events = [e for e in state["runtime_events"] if e["event_type"] == "CONTEXT_PROJECTION"]
    assert len(events) == 11
    assert all(set(e) >= {"event_type", "node", "included_fact_count", "included_evidence_count", "included_plan_step_count", "serialized_context_chars"} for e in events)


def test_projection_fault_injection_full_e2e_and_usage(seeded_engine):
    state, client, delays = projected_run(seeded_engine, True)
    assert state["status"] == "DRAFT_READY" and state["llm_call_count"] == 12
    assert state["tool_call_count"] == 4 and state["retry_count"] == 1 and delays == [0.5]
    assert state["token_usage"]["total_tokens"] == 1320
    usages = [e for e in state["runtime_events"] if e["event_type"] == "LLM_USAGE"]
    assert len(usages) == 11 and any(e["node"] == "select_tool" and e["attempt"] == 2 for e in usages)
    assert [r["tool_name"] for r in state["tool_history"]] == ["get_sales_overview", "analyze_region_performance", "analyze_product_performance", "query_orders"]
