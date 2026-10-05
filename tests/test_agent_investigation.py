from copy import deepcopy
import json
import socket

import pytest
from sqlalchemy.exc import OperationalError

from backend.app.agent import prompts
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.graph import build_investigation_graph
from backend.app.agent.grounding import resolve_pointer, validate_references, same_scalar
from backend.app.agent.schemas import EvidenceReference, GroundedAnalysisDraft, ProgressReview
from backend.app.agent.state import initial_state
from backend.app.data.database import session_factory
from backend.app.data.validation import dataset_hashes
from backend.app.llm.gateway import GatewayError
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import QUERY, FakeInvestigationGateway, goal, plan, selection


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pytest must remain offline")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def review_at(index, *, complete=False):
    def output(**context):
        result = context["latest_result"]
        pointer = ("/data/current/revenue", "/data/regions/0/region",
                   "/data/products/0/current/profit", "/data/orders/0/order_id")[index]
        eid = next(iter(extract_evidence([context["history"][-1]])))
        objective = ("Locate regional deterioration", "Investigate products in " + str(resolve_pointer(result, pointer)),
                     "Corroborate the discount signal with individual orders", "Done")[index]
        remaining = {"steps": [{"objective": objective,
                                 "expected_output": "Evidence for the updated objective"}]}
        return {"decision": "COMPLETE" if complete else "CONTINUE",
                "selected_evidence_ids": [eid], "next_objective": None if complete else objective,
                "updated_plan": None if complete else remaining,
                "remaining_evidence_gap": None if complete else "Missing business evidence to: " + objective}
    return output


def draft_from_evidence(**context):
    findings = []
    for obs in context["observations"]:
        findings.append({"title": "Observed business measurement", "claim_type": "CONTRIBUTING_FACTOR", "interpretation": "Observed measurements support further business investigation.",
                         "supporting_evidence_ids": obs["evidence_ids"]})
    return {"executive_interpretation": "A test draft grounded in actual read tools.", "findings": findings,
            "recommendations": [{"title": "Review", "action": "Advisory review only", "related_finding_index": 0}],
            "limitations": ["Scripted model test, not live reasoning quality."]}


def scripted(*, reviews=None, draft=draft_from_evidence, selections=None):
    return FakeInvestigationGateway(goal(), plan(), selections or [
        selection(), selection("analyze_region_performance"),
        selection("analyze_product_performance", region="EAST_CHINA"),
        selection("query_orders", region="EAST_CHINA", product_id="SKU-A12", approval_status="MISSING_APPROVAL", limit=2),
    ], reviews if reviews is not None else [review_at(i, complete=i == 3) for i in range(4)], draft)


def run(fake, engine):
    with build_default_registry(session_factory(engine), raise_unexpected_errors=True) as registry:
        return build_investigation_graph(fake, registry, raise_unexpected_errors=True).invoke(
            initial_state(QUERY), config={"recursion_limit": 40})


def test_real_multistep_replanning_and_read_only(seeded_engine):
    factory = session_factory(seeded_engine)
    with factory() as session:
        before = dataset_hashes(session)
    fake = scripted()
    result = run(fake, seeded_engine)
    assert result["status"] == "DRAFT_READY"
    assert result["tool_call_count"] == len(result["tool_history"]) == 4
    assert result["llm_call_count"] == len(fake.calls) == 11
    assert len(result["observations"]) == len(result["completed_steps"]) == 4
    assert result["visited_nodes"].count("review_progress") == 4
    assert sum(o["decision"] == "CONTINUE" for o in result["observations"]) == 3
    assert result["tool_history"][0]["data"]["current"]["revenue"] == "3150655.70"
    assert result["tool_history"][0]["data"]["current"]["profit"] == "870250.70"
    assert result["latest_tool_result"]["meta"]["total_matches"] == 10
    assert result["latest_tool_result"]["meta"]["returned_count"] == 2
    selectors = [inputs for name, inputs in fake.calls if name == "select_tool"]
    reviews = [inputs for name, inputs in fake.calls if name == "review_progress"]
    assert selectors[2]["plan"] == result["observations"][1]["plan_after"]
    assert "EAST_CHINA" in selectors[2]["step"]["objective"]
    assert selectors[2]["observations"] == result["observations"][:2]
    assert reviews[0]["latest_result"]["data"]["current"]["revenue"] == "3150655.70"
    assert "EAST_CHINA" not in json.dumps(plan())
    assert result["completed_steps"][0]["step_id"] == "s1"
    assert all(step["status"] == "COMPLETED" for step in result["completed_steps"])
    assert json.loads(json.dumps(result)) == result
    assert result["analysis_draft"]["recommendations"][0]["related_finding_index"] == 0
    with factory() as session:
        assert dataset_hashes(session) == before


@pytest.mark.parametrize("retryable", [False, True])
def test_tool_errors_stop_after_bounded_read_retry(empty_engine, retryable):
    calls = []
    def failing_factory():
        calls.append(1)
        raise OperationalError("SELECT private", {}, Exception("private test sentinel"))
    fake = scripted()
    if retryable:
        registry = build_default_registry(failing_factory)
    else:
        from backend.app.data.database import create_tables
        create_tables(empty_engine)
        registry = build_default_registry(session_factory(empty_engine))
    with registry:
        result = build_investigation_graph(fake, registry).invoke(initial_state(QUERY))
    assert result["status"] == "TOOL_ERROR"
    assert result["tool_call_count"] == (2 if retryable else 1) and result["llm_call_count"] == 3
    assert result["latest_tool_result"]["error"]["retryable"] == retryable
    assert result["tool_history"][0]["error_code"] == ("DATABASE_ERROR" if retryable else "NO_DATA")
    assert result["plan"]["steps"][0]["status"] == "FAILED"
    assert not result["observations"] and result["analysis_draft"] is None
    assert "review_progress" not in result["visited_nodes"]
    if retryable:
        assert calls == [1, 1]


@pytest.mark.parametrize("bad", ["evidence", "value", "pointer", "source", "step", "completed_reuse", "objective", "empty_plan"])
def test_invalid_review_cannot_continue(seeded_engine, bad):
    def malformed(**context):
        review = review_at(0)(**context)
        if bad == "evidence": review["selected_evidence_ids"] = ["fabricated-evidence"]
        elif bad == "value": review["raw_value"] = "999999999.99"
        elif bad == "pointer": review["json_pointer"] = "/data/nonexistent"
        elif bad == "source": review["source_tool"] = "query_orders"
        elif bad == "step": review["completed_step_id"] = "unknown-step"
        elif bad == "completed_reuse": review["updated_plan"]["steps"][0]["step_id"] = "s1"
        elif bad == "objective": review["next_objective"] = None
        elif bad == "empty_plan": review["updated_plan"]["steps"] = []
        return review
    result = run(scripted(reviews=[malformed]), seeded_engine)
    assert result["status"] == "ERROR"
    assert result["errors"][-1]["code"] == "MODEL_OUTPUT_VALIDATION_ERROR"
    assert result["tool_call_count"] == 1 and result["analysis_draft"] is None


@pytest.mark.parametrize("bad", ["evidence", "value", "recommendation", "missing_reference"])
def test_invalid_draft_rejected(seeded_engine, bad):
    def malformed(**context):
        draft = draft_from_evidence(**context)
        if bad == "evidence": draft["findings"][0]["supporting_evidence_ids"] = ["E-12345"]
        elif bad == "value": draft["findings"][0]["raw_value"] = "made-up"
        elif bad == "recommendation": draft["recommendations"][0]["related_finding_index"] = 999
        else: draft["findings"][0]["supporting_evidence_ids"] = []
        return draft
    result = run(scripted(reviews=[review_at(0, complete=True)], draft=malformed), seeded_engine)
    assert result["status"] == "ERROR" and result["analysis_draft"] is None
    assert result["errors"][-1]["code"] == "MODEL_OUTPUT_VALIDATION_ERROR"


@pytest.mark.parametrize("output,status", [
    (goal(start_date=None, end_date=None, needs_clarification=True, clarification_question="Which dates?"), "NEEDS_CLARIFICATION"),
    (goal(supported=False, task_type=None), "UNSUPPORTED"),
])
def test_early_exit_preserved(seeded_engine, output, status):
    fake = scripted()
    fake.outputs["understand_goal"] = output
    result = run(fake, seeded_engine)
    assert result["status"] == status and result["tool_call_count"] == 0
    assert result["llm_call_count"] == len(fake.calls) == 1


@pytest.mark.parametrize("proposal", [selection("delete_orders"), selection("query_orders", sql="DROP TABLE orders")])
def test_malicious_selection_never_executes(seeded_engine, proposal):
    result = run(scripted(selections=[proposal]), seeded_engine)
    assert result["status"] == "ERROR" and result["tool_call_count"] == 0
    assert "execute_tool" not in result["visited_nodes"]


def test_order_evidence_does_not_support_another_row(seeded_engine):
    result = run(scripted(), seeded_engine)
    record = result["tool_history"][-1]
    eid = record["evidence_ids"][0]
    reference = EvidenceReference(evidence_id=eid, json_pointer="/data/orders/1/order_id",
                                  value=record["data"]["orders"][1]["order_id"])
    with pytest.raises(GatewayError):
        validate_references([eid], [reference], [record])


def test_semantics_and_no_runtime_answer_hardcoding():
    assert "clarification_question MUST be JSON null" in prompts.UNDERSTAND
    for prompt in (prompts.PLAN, prompts.REVIEW, prompts.SYNTHESIZE):
        assert "previous equal-length period" in prompt and "前一等长周期" in prompt
        assert "not Year-over-Year" in prompt and "Do not call it YoY unless" in prompt
        assert "Profit decrease != negative profit" in prompt
        assert "利润下降不等于利润为负" in prompt
        assert "direct absolute profit loss" in prompt
    from pathlib import Path
    for folder in ("backend/app/agent", "backend/app/llm"):
        for path in Path(folder).rglob("*.py"):
            for marker in ("ground_truth", "SKU-A12", "SKU-B07", "SKU-C03", "C102", "C207", "EAST_CHINA"):
                assert marker not in path.read_text(encoding="utf-8")


def test_exact_numeric_reference_normalization():
    assert same_scalar("0", 0, "/data/changes/order_count_growth_pct")
    assert same_scalar("1.00", "1.0", "/data/current/revenue")
    assert not same_scalar("1.0001", 1, "/data/current/margin")
    assert not same_scalar("0", False, "/data/current/revenue")
    assert not same_scalar("001", 1, "/data/orders/0/customer_id")
    ref = EvidenceReference(evidence_id="numeric-test", json_pointer="/data/current/revenue", value=0)
    validate_references(["numeric-test"], [ref], [{"success": True, "tool_name": "get_sales_overview",
        "evidence_ids": ["numeric-test"], "data": {"current": {"revenue": "0"}}}])
    assert ref.value == "0" and isinstance(ref.value, str)


def test_next_objective_summary_does_not_require_verbatim_plan_text():
    review = ProgressReview.model_validate({"decision": "CONTINUE",
        "selected_evidence_ids": ["example"],
        "remaining_evidence_gap": "Missing regional contribution measurements", "next_objective": "Compare regions", "updated_plan": {"steps": [{"objective": "Compare the regional contribution to profit change", "expected_output": "Regional measurements"}]}})
    assert review.updated_plan.steps[0].objective != review.next_objective


def test_cross_tool_finding_keeps_all_verified_sources():
    records = [{"success": True, "tool_name": name, "evidence_ids": [eid], "data": {"amount": value}}
               for name, eid, value in (("get_sales_overview", "company", "100"),
                                        ("analyze_region_performance", "region", "80"))]
    refs = [EvidenceReference(evidence_id=eid, json_pointer="/data/amount", value=value)
            for eid, value in (("company", "100"), ("region", "80"))]
    assert validate_references(["region"], refs, records, source_tool="analyze_region_performance") == ["company", "region"]
    assert validate_references(["region", "company"], refs[1:], records, source_tool="analyze_region_performance") == ["company", "region"]
    with pytest.raises(GatewayError):
        validate_references(["region"], refs, records, source_tool="query_orders")
    refs[0].evidence_id = "invented"
    with pytest.raises(GatewayError):
        validate_references(["region"], refs, records, source_tool="analyze_region_performance")
