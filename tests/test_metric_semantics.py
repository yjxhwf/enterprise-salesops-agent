from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from backend.app.agent.evidence import extract_evidence, visible_evidence
from backend.app.agent.facts import canonical_facts
from backend.app.agent.schemas import DraftFinding, GroundedAnalysisDraft, ProgressReview
from backend.app.agent.semantics import validate_draft, validate_review, validate_text
from backend.app.llm.gateway import GatewayError
from tests.test_canonical_evidence import history, offline
from tests.test_agent_investigation import run, scripted, review_at


def ratio_catalog(history, current="0.9501", previous="0.9542", identity=None):
    catalog = extract_evidence(deepcopy(history[:1]))
    item = next(iter(catalog.values()))
    raw = item["internal_reference"]["raw_value"]
    raw["current"]["avg_discount"] = current
    raw["previous"]["avg_discount"] = previous
    if identity:
        raw["product_id"] = identity
    return catalog


@pytest.mark.parametrize("identity", [None, "arbitrary-product", "NEW-42"])
def test_ratio_projection_raw_preserved_and_no_entity_branch(history, identity):
    catalog = ratio_catalog(history, identity=identity)
    original = deepcopy(catalog)
    facts = {f["semantic_metric"]: f for f in canonical_facts(catalog) if f["semantic_metric"]}
    ratio, depth = facts["price_realization_ratio"], facts["discount_depth"]
    assert catalog == original
    assert ratio["metric"] == "avg_discount" and ratio["current_value"] == "0.9501"
    assert ratio["display_value"] == "95.01%" and ratio["comparison_display_value"] == "95.42%"
    assert ratio["change_display_value"] == "-0.41个百分点"
    assert ratio["change_direction"] == "DECREASE"
    assert depth["derived_value"] == "0.0499" and depth["display_value"] == "4.99%"
    assert depth["comparison_display_value"] == "4.58%"
    assert depth["change_display_value"] == "0.41个百分点"
    assert depth["change_direction"] == "INCREASE" and "优惠加深" in depth["rendered_text"]
    assert "导致" not in json.dumps(facts, ensure_ascii=False)


def test_llm_projection_hides_technical_ratio_labels(history):
    original = deepcopy(history)
    text = json.dumps(visible_evidence(history), ensure_ascii=False)
    assert "avg_discount" not in text and "discount_rate" not in text
    assert "price_realization_ratio" in text and "实际优惠幅度" in text
    assert history == original


@pytest.mark.parametrize("text", ["优惠收窄", "折扣收窄", "平均折扣略有收窄", "discount narrowed", "shallower discount"])
def test_deeper_discount_rejects_shallower_claim(history, text):
    with pytest.raises(GatewayError) as error:
        validate_text(text, ratio_catalog(history))
    assert error.value.code == "MODEL_OUTPUT_VALIDATION_ERROR"


@pytest.mark.parametrize("text", ["优惠加深", "折扣扩大", "discount increased", "deeper discount"])
def test_shallower_discount_rejects_deeper_claim(history, text):
    with pytest.raises(GatewayError):
        validate_text(text, ratio_catalog(history, "0.97", "0.95"))


def test_discount_direction_accepts_correct_scoped_claims(history):
    catalog = ratio_catalog(history, identity="sample-deeper")
    other = deepcopy(next(iter(catalog.values())))
    other["internal_reference"]["raw_value"]["product_id"] = "sample-shallower"
    other["internal_reference"]["raw_value"]["current"]["avg_discount"] = "0.98"
    catalog["other"] = other
    validate_text("sample-deeper 优惠加深；sample-shallower 优惠收窄", catalog)
    with pytest.raises(GatewayError):
        validate_text("sample-deeper 优惠收窄", catalog)


@pytest.mark.parametrize("claim", ["不是因为折扣", "利润下滑并非折扣扩大所致", "与产品结构无关", "可以排除客户因素", "唯一原因"])
def test_exclusions_cannot_hide_in_review_objectives(history, claim):
    review = ProgressReview(decision="CONTINUE", selected_evidence_ids=list(extract_evidence(history[:1])),
                            remaining_evidence_gap="Missing order discount corroboration", next_objective=claim, updated_plan={"steps": [{"objective": "Continue investigation", "expected_output": "Evidence"}]})
    with pytest.raises(GatewayError):
        validate_review(review, extract_evidence(history))


def test_review_has_no_conclusion_and_can_drop_remaining_direction(seeded_engine):
    state = run(scripted(), seeded_engine)
    assert state["status"] == "DRAFT_READY"
    assert all("interpretation" not in o for o in state["observations"])
    assert json.loads(json.dumps(state)) == state
    props = ProgressReview.model_json_schema()["properties"]
    assert set(props) == {"decision", "selected_evidence_ids", "next_objective", "updated_plan", "remaining_evidence_gap"}
    first = state["observations"][0]
    assert first["plan_before"] != first["plan_after"]
    with pytest.raises(ValidationError):
        ProgressReview(decision="COMPLETE", selected_evidence_ids=["id"], next_objective=None,
                       updated_plan=None, investigation_interpretation="X is not the cause")


@pytest.mark.parametrize("kind", ["CONTRIBUTING_FACTOR", "OBSERVATION"])
def test_allowed_claim_schema(kind):
    assert DraftFinding(claim_type=kind, title="Pressure", interpretation="Associated pressure",
                        supporting_evidence_ids=["id"]).claim_type == kind


def test_exclusion_claim_type_rejected():
    with pytest.raises(ValidationError):
        DraftFinding(claim_type="EXCLUSION", title="Excluded", interpretation="Not a cause", supporting_evidence_ids=["id"])


def test_contribution_requires_two_independent_real_measurements(history):
    catalog = ratio_catalog(history)
    eid = next(iter(catalog))
    raw = catalog[eid]["internal_reference"]["raw_value"]
    raw["current"] = {"avg_discount": "0.9501"}
    draft = GroundedAnalysisDraft(executive_interpretation="Business pressure", findings=[dict(
        claim_type="CONTRIBUTING_FACTOR", title="Pricing pressure", interpretation="优惠加深形成压力",
        supporting_evidence_ids=[eid, eid])], recommendations=[dict(title="Review", action="Review pricing", related_finding_index=0)],
        limitations=["Association does not establish causation"])
    with pytest.raises(GatewayError):
        validate_draft(draft, catalog)
    raw["current"]["profit"] = None
    with pytest.raises(GatewayError):
        validate_draft(draft, catalog)
    raw["current"]["profit"] = "100"
    validate_draft(draft, catalog)
    draft.findings[0].interpretation = "折扣收窄"
    with pytest.raises(GatewayError):
        validate_draft(draft, catalog)
    draft.findings[0].interpretation = "与产品结构无关"
    with pytest.raises(GatewayError):
        validate_draft(draft, catalog)
