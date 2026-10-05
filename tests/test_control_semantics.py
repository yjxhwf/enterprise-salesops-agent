from copy import deepcopy
import json

import pytest

from backend.app.agent.evidence import comparison_semantics, extract_evidence, visible_evidence
from backend.app.agent.graph import build_investigation_graph
from backend.app.agent.schemas import ProgressReview
from backend.app.agent.semantics import validate_text
from backend.app.agent.state import initial_state
from backend.app.data.database import session_factory
from backend.app.llm.gateway import GatewayError
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import QUERY
from tests.test_agent_investigation import draft_from_evidence, review_at, run, scripted
from tests.test_canonical_evidence import history, offline


def test_review_schema_excludes_control_state():
    schema = json.dumps(ProgressReview.model_json_schema())
    for field in ("completed_step_id", "step_id", "current_step_index", "run_id", "tool_call_count", "llm_call_count", "status"):
        assert f'"{field}"' not in schema


def test_completion_owned_by_program_after_successful_review(seeded_engine):
    with build_default_registry(session_factory(seeded_engine)) as registry:
        states = list(build_investigation_graph(scripted(), registry).stream(
            initial_state(QUERY), config={"recursion_limit": 40}, stream_mode="values"))
    executed = next(s for s in states if s["status"] == "TOOL_EXECUTED")
    assert executed["completed_steps"] == []
    assert executed["plan"]["steps"][0]["status"] == "IN_PROGRESS"
    reviewed = next(s for s in states if s["status"] == "INVESTIGATING")
    assert reviewed["completed_steps"][0]["step_id"] == "s1"
    assert reviewed["observations"][0]["completed_step_id"] == "s1"
    assert reviewed["completed_steps"][0]["status"] == "COMPLETED"


def test_merge_preserves_completed_history_and_program_assigns_ids(seeded_engine):
    state = run(scripted(), seeded_engine)
    assert state["status"] == "DRAFT_READY"
    completed = state["completed_steps"]
    assert [s["step_id"] for s in completed] == ["s1", "replan-1-1", "replan-2-1", "replan-3-1"]
    for index, observation in enumerate(state["observations"][:-1], 1):
        merged = observation["plan_after"]["steps"]
        assert merged[:index] == completed[:index]
        assert all(s["status"] == "PENDING" for s in merged[index:])
        assert not {s["step_id"] for s in merged[index:]} & {s["step_id"] for s in completed[:index]}
    assert state["plan"]["steps"] == completed
    assert "EAST_CHINA" in state["observations"][1]["plan_after"]["steps"][-1]["objective"]


def test_comparison_basis_from_actual_tool_periods(history):
    catalog = extract_evidence(history)
    for item in catalog.values():
        if item["source_tool"] == "query_orders":
            assert item["comparison_basis"] is None
        else:
            assert item["comparison_basis"] == "PREVIOUS_EQUAL_LENGTH_PERIOD"
            assert item["comparison_label"] == "前一等长周期"
            assert item["comparison_period"] == {"start_date": "2026-07-01", "end_date": "2026-07-31"}
    assert visible_evidence(history)[0]["comparison_basis"] == "PREVIOUS_EQUAL_LENGTH_PERIOD"


def test_unknown_comparison_not_invented(history):
    record = deepcopy(history[0])
    record["meta"]["comparison_period"]["start_date"] = "2026-06-01"
    with pytest.raises(ValueError):
        comparison_semantics(record)


@pytest.mark.parametrize("text", ["收入同比增长", "Revenue YoY rose", "year-over-year revenue increased", "YEAR OVER YEAR growth"])
def test_yoy_requires_evidence(history, text):
    catalog = extract_evidence(history)
    with pytest.raises(GatewayError) as error:
        validate_text(text, catalog)
    assert error.value.code == "MODEL_OUTPUT_VALIDATION_ERROR"
    # A future trusted evidence adapter can provide this basis; wording is not globally banned.
    future = deepcopy(catalog)
    next(iter(future.values()))["comparison_basis"] = "YEAR_OVER_YEAR"
    validate_text(text, future)


@pytest.mark.parametrize("text", ["较前一等长周期增长9.56%", "利润下降11.01%", "利润较前一等长周期减少", "利润下降不等于亏损", "profit declined, not negative profit"])
def test_legal_comparison_and_profit_decline(history, text):
    validate_text(text, extract_evidence(history))


@pytest.mark.parametrize("text", ["公司利润为负", "公司出现亏损", "The company has negative profit"])
def test_positive_profit_rejects_negative_claim(history, text):
    with pytest.raises(GatewayError):
        validate_text(text, extract_evidence(history))


def test_profit_rule_uses_referenced_scope(history):
    catalog = extract_evidence(history[:1])
    negative = deepcopy(next(iter(catalog.values())))
    negative.update(evidence_id="negative-row", fact_key="sample-product")
    negative["internal_reference"]["raw_value"] = {"product_id": "sample-product", "current": {"profit": "-2.00"}}
    catalog["negative-row"] = negative
    validate_text("sample-product 利润为负", catalog, ["negative-row"])
    validate_text("sample-product 利润为负", catalog)
    with pytest.raises(GatewayError):
        validate_text("公司利润为负", catalog)


@pytest.mark.parametrize("text", ["收入同比增长", "公司利润为负", "公司出现亏损"])
def test_review_semantics_reject_before_completion(seeded_engine, text):
    def invalid(**context):
        result = review_at(0, complete=False)(**context)
        result["next_objective"] = text
        return result
    state = run(scripted(reviews=[invalid]), seeded_engine)
    assert state["status"] == "ERROR"
    assert state["errors"][-1]["code"] == "MODEL_OUTPUT_VALIDATION_ERROR"
    assert state["completed_steps"] == [] and state["observations"] == []


@pytest.mark.parametrize("field", ["summary", "finding", "recommendation", "limitation"])
def test_synthesis_semantics_reject_every_output_area(seeded_engine, field):
    def invalid(**context):
        result = draft_from_evidence(**context)
        if field == "summary": result["executive_interpretation"] = "公司利润为负"
        elif field == "finding": result["findings"][0]["interpretation"] = "Revenue increased YoY"
        elif field == "recommendation": result["recommendations"][0]["action"] = "公司出现亏损，建议检查"
        else: result["limitations"] = ["同比比较覆盖有限"]
        return result
    state = run(scripted(reviews=[review_at(0, complete=True)], draft=invalid), seeded_engine)
    assert state["status"] == "ERROR" and state["analysis_draft"] is None
    assert state["errors"][-1]["code"] == "MODEL_OUTPUT_VALIDATION_ERROR"
