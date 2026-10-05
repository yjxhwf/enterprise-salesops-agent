from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from backend.app.agent.guards.context import build_llm_context, visible_ids, result_view
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.facts import canonical_facts
from backend.app.agent.schemas import ProgressReview
from backend.app.data.database import session_factory
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import DATES, goal, plan, selection
from tests.test_agent_investigation import offline, run, scripted, review_at


@pytest.fixture
def fifty(seeded_engine):
    args = {**DATES, "limit": 50}
    with build_default_registry(session_factory(seeded_engine)) as registry:
        return [{**registry.invoke("query_orders", args).model_dump(mode="json"), "arguments": args}]


def test_fifty_rows_remain_in_state_but_not_context(fifty):
    state = {"tool_history": fifty, "observations": []}
    before = deepcopy(state)
    view = build_llm_context("review_progress", history=fifty)
    digest = view["result_digest"]
    assert (digest["returned_count"], digest["visible_row_count"], digest["omitted_row_count"]) == (50, 10, 40)
    assert digest["matched_count"] >= 50 and digest["truncated"] is True
    ids = list(extract_evidence(fifty))
    assert visible_ids(view) == set(ids[:10])
    text = json.dumps(view, ensure_ascii=False)
    assert all(eid not in text for eid in ids[10:])
    assert "raw_value" not in text and "orders" not in view
    assert len(text) < len(json.dumps(fifty, ensure_ascii=False)) * .9
    for bundle in view["latest_evidence"]:
        assert len(bundle["facts"]) == len({f["metric"] for f in bundle["facts"]})
    assert view == build_llm_context("review_progress", history=deepcopy(fifty))
    assert state == before == json.loads(json.dumps(state))
    assert len(state["tool_history"][0]["data"]["orders"]) == 50


def test_selected_order_outside_first_ten_kept_once(fifty):
    ids = list(extract_evidence(fifty))
    observations = [{"evidence_ids": [ids[-1]], "facts": canonical_facts(extract_evidence(fifty), [ids[-1]])}]
    view = build_llm_context("review_progress", history=fifty, observations=observations)
    assert ids[-1] in visible_ids(view) and len(visible_ids(view)) == 10
    assert not view["prior_selected_evidence"]
    assert len([b for b in view["latest_evidence"] if b["evidence_id"] == ids[-1]]) == 1


def test_generic_high_cardinality_without_tool_name_branch(fifty):
    rows = fifty[0]["data"]["orders"]
    record = {**fifty[0], "tool_name": "future_customer_reader", "data": {"customers": [
        {"customer_id": "TEST-"+str(i), "current": {"profit": str(i+1)}} for i in range(19)]},
        "evidence_ids": ["aggregate:test"]}
    view, digest = result_view([record], [])
    assert len(view) == 12 and digest["omitted_row_count"] == 7


def test_synthesis_cap_and_selected_coverage(fifty):
    catalog = extract_evidence(fifty)
    chosen = list(catalog)[:10]
    observations = [dict(evidence_ids=chosen)]
    view = build_llm_context("synthesize", history=fifty, observations=observations)
    assert visible_ids(view) == set(chosen)
    coverage = view["synthesis_coverage"]
    assert coverage["visible_synthesis_fact_count"] == 48
    assert coverage["omitted_synthesis_fact_count"] == len(canonical_facts(catalog, chosen))-48


@pytest.mark.parametrize("gap", [None, "", "需要进一步分析", "还需要更多信息", "继续深入", "需要更多证据", "还可以继续深入分析", "need more evidence"])
def test_continue_requires_specific_gap(gap):
    args = dict(decision="CONTINUE", selected_evidence_ids=["id"], remaining_evidence_gap=gap,
                next_objective="Inspect order discounts", updated_plan={"steps": [dict(objective="Inspect order discounts", expected_output="Order price ratios")]})
    with pytest.raises(ValidationError):
        ProgressReview.model_validate(args)


def test_budget_summary_is_small_and_omits_unreliable_tokens():
    from backend.app.agent.guards.context import active_runtime, budget_summary
    from backend.app.agent.guards.runtime import Runtime, runtime_defaults
    from backend.app.agent.guards.limits import RuntimeLimits
    state = {**runtime_defaults(), "llm_call_count": 5, "tool_call_count": 2, "agent_step_count": 8}
    token = active_runtime.set(Runtime(state, "review_progress", RuntimeLimits(), lambda _: None))
    try:
        summary = budget_summary()
        assert summary == dict(tool_calls_used=2, tool_calls_remaining=6, llm_calls_used=5,
                               llm_calls_remaining=15, agent_steps_used=8, agent_steps_remaining=22)
        state["token_usage"].update(usage_available=True, total_tokens=2345)
        assert budget_summary()["soft_token_budget_remaining"] == 117655
    finally:
        active_runtime.reset(token)


def test_complete_skips_unnecessary_plan_and_still_synthesizes(seeded_engine):
    state = run(scripted(reviews=[review_at(0, complete=True)]), seeded_engine)
    assert state["status"] == "DRAFT_READY" and state["tool_call_count"] == 1
    skipped = [s for s in state["plan"]["steps"] if s["status"] == "SKIPPED"]
    assert skipped and all(s["skip_reason"] == "SUFFICIENT_EVIDENCE" for s in skipped)
    assert state["observations"][0]["plan_before"]["steps"]
    assert state["observations"][0]["plan_after"] == state["plan"]


def test_dynamic_gap_replans_and_selects_next_tool(seeded_engine):
    state = run(scripted(), seeded_engine)
    assert state["status"] == "DRAFT_READY" and state["tool_call_count"] == 4
    assert all(o["remaining_evidence_gap"] and o["next_objective"] for o in state["observations"][:-1])
    assert state["observations"][-1]["remaining_evidence_gap"] is None


def test_invisible_existing_citation_is_rejected(seeded_engine):
    def hidden(**context):
        result = review_at(0, complete=True)(**context)
        result["selected_evidence_ids"] = [list(extract_evidence(context["history"]))[-1]]
        return result
    state = run(scripted(reviews=[hidden], selections=[selection("query_orders", limit=50)]), seeded_engine)
    assert state["status"] == "ERROR" and state["analysis_draft"] is None
    assert len(state["tool_history"][0]["data"]["orders"]) == 50


@pytest.mark.parametrize("field,value", [("remaining_evidence_gap", "Missing order evidence"), ("next_objective", "Query orders"), ("updated_plan", {"steps": [{"objective": "Orders", "expected_output": "Order measurements"}]})])
def test_complete_cannot_schedule_more_work(field, value):
    args = dict(decision="COMPLETE", selected_evidence_ids=["id"], next_objective=None, updated_plan=None, remaining_evidence_gap=None)
    args[field] = value
    with pytest.raises(ValidationError):
        ProgressReview.model_validate(args)
