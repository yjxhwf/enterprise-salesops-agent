from copy import deepcopy
import json
from decimal import Decimal

import pytest

from backend.app.agent.evidence import extract_evidence, visible_evidence, visible_observations
from backend.app.agent.facts import canonical_facts, render_fact
from backend.app.agent.schemas import ProgressReview, GroundedAnalysisDraft
from tests.test_agent_investigation import run, scripted, draft_from_evidence, review_at
from tests.test_canonical_evidence import history, offline


def overview_facts(history):
    return {f["metric"]: f for f in canonical_facts(extract_evidence(history[:1]))}


def test_facts_deterministic_source_derived_and_json_safe(history):
    catalog = extract_evidence(history)
    original = deepcopy(catalog)
    facts = canonical_facts(catalog)
    assert facts == canonical_facts(deepcopy(catalog))
    assert catalog == original
    assert json.loads(json.dumps(facts)) == facts
    assert len({f["fact_id"] for f in facts}) == len(facts)
    for f in facts:
        raw = catalog[f["evidence_id"]]["internal_reference"]["raw_value"]
        source = raw.get("current", raw)[f["metric"]]
        assert f["current_value"] == (str(1 - Decimal(source)) if f["semantic_metric"] == "discount_depth" else source)
        assert f["rendered_text"] == render_fact(f)


@pytest.mark.parametrize("metric,phrase,value", [
    ("revenue", "Revenue 3,150,655.70；较前一等长周期增长9.56%", "3150655.70"),
    ("profit", "Profit 870,250.70；较前一等长周期下降11.01%", "870250.70"),
    ("margin", "Margin 27.62%；较前一等长周期下降6.39个百分点", "0.2762125674347723872208569156"),
])
def test_baseline_rendering(history, metric, phrase, value):
    fact = overview_facts(history)[metric]
    assert phrase in fact["rendered_text"]
    assert fact["current_value"] == value
    assert "同比" not in fact["rendered_text"]
    if metric == "profit":
        assert "当前利润为正" in fact["rendered_text"]
        assert "利润为负" not in fact["rendered_text"]


def test_count_discount_share_and_order_metrics(history):
    facts = canonical_facts(extract_evidence(history))
    base = overview_facts(history)
    assert base["order_count"]["display_value"] == "667"
    assert base["order_count"]["change_direction"] == "UNCHANGED"
    assert base["avg_discount"]["display_value"] == "4.99%"
    assert base["quantity"]["display_value"] == "7,404"
    assert any(f["metric"] == "revenue_share" and f["unit"] == "PERCENTAGE" for f in facts)
    assert any(f["metric"] == "approval_status" and f["subject_type"] == "ORDER" for f in facts)


def test_no_prior_comparison_or_null_metric_is_not_invented(history):
    catalog = extract_evidence(history[:1])
    raw = next(iter(catalog.values()))["internal_reference"]["raw_value"]
    raw["comparison_available"] = False
    raw["current"]["margin"] = None
    facts = {f["metric"]: f for f in canonical_facts(catalog)}
    assert facts["profit"]["change_direction"] == "NOT_AVAILABLE"
    assert facts["profit"]["comparison_value"] is None
    assert facts["margin"]["display_value"] == "不可用"


def test_model_schemas_do_not_request_facts():
    review = ProgressReview.model_json_schema()["properties"]
    draft = GroundedAnalysisDraft.model_json_schema()
    assert "selected_evidence_ids" in review and "investigation_interpretation" not in review
    assert "notable_findings" not in review and "facts" not in review
    assert "executive_interpretation" in draft["properties"]
    assert "executive_facts" not in draft["properties"]
    assert set(draft["$defs"]["DraftFinding"]["properties"]) == {"claim_type", "title", "interpretation", "supporting_evidence_ids"}


def test_observation_and_draft_facts_assembled_by_program(seeded_engine):
    state = run(scripted(), seeded_engine)
    catalog = extract_evidence(state["tool_history"])
    assert state["status"] == "DRAFT_READY"
    for observation in state["observations"]:
        assert observation["facts"] == canonical_facts(catalog, observation["evidence_ids"])
    for finding in state["analysis_draft"]["findings"]:
        assert finding["supporting_facts"] == canonical_facts(catalog, finding["supporting_evidence_ids"])
    assert {f["metric"] for f in state["analysis_draft"]["executive_facts"]} == {"revenue", "profit", "margin"}
    assert state["analysis_draft"]["recommendations"][0]["action"] == "Advisory review only"
    assert json.loads(json.dumps(state)) == state


def test_observation_projection_keeps_raw_decimals_internal(seeded_engine):
    state = run(scripted(), seeded_engine)
    projected = visible_observations(state["observations"])
    assert "current_value" not in json.dumps(projected)
    assert "0.2762125674347723872208569156" not in json.dumps(projected)
    assert "rendered_text" in json.dumps(projected)
    assert state["observations"][0]["facts"][0].get("current_value") is not None


def test_product_mix_renderer_has_measurements_not_causality(history):
    catalog = extract_evidence(history[2:3])
    eid = next(k for k, v in catalog.items() if v["fact_key"] == "SKU-B07")
    facts = {f["metric"]: f for f in canonical_facts(catalog, [eid])}
    assert facts["revenue_share"]["change_direction"] == "INCREASE"
    assert facts["profit"]["change_direction"] == "INCREASE"
    assert facts["margin"]["display_value"] == "5.23%"
    text = json.dumps(facts, ensure_ascii=False)
    for term in ("主要原因", "导致", "dilution", "利润损失"):
        assert term not in text


def test_scope_is_preserved_in_rendered_facts(history):
    record = deepcopy(history[2])
    record["arguments"] = {**record["arguments"], "region": "SAMPLE_REGION"}
    assert all("region=SAMPLE_REGION" in f["rendered_text"] for f in canonical_facts(extract_evidence([record])))


def test_interpretation_still_rejected_and_not_rewritten(seeded_engine):
    def invalid(**context):
        result = review_at(0, complete=False)(**context)
        result["next_objective"] = "利润同比下降"
        return result
    state = run(scripted(reviews=[invalid]), seeded_engine)
    assert state["status"] == "ERROR" and state["observations"] == []
    assert state["errors"][-1]["code"] == "MODEL_OUTPUT_VALIDATION_ERROR"


@pytest.mark.parametrize("bad", ["invented", "model_fact"])
def test_draft_cannot_invent_evidence_or_supply_program_facts(seeded_engine, bad):
    def invalid(**context):
        result = draft_from_evidence(**context)
        if bad == "invented": result["findings"][0]["supporting_evidence_ids"] = ["made-up"]
        else: result["findings"][0]["supporting_facts"] = ["fabricated"]
        return result
    state = run(scripted(reviews=[review_at(0, complete=True)], draft=invalid), seeded_engine)
    assert state["status"] == "ERROR" and state["analysis_draft"] is None


def test_renderer_requires_no_model_or_external_files(history, monkeypatch):
    catalog = extract_evidence(history)
    def forbidden(*args, **kwargs):
        raise AssertionError("Renderer must be pure and offline")
    monkeypatch.setattr("builtins.open", forbidden)
    assert canonical_facts(catalog)
    assert visible_evidence(history)[0]["rendered_facts"]
