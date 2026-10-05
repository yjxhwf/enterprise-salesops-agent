import json

import pytest

from backend.app.agent.evidence import extract_evidence
from backend.app.agent.schemas import GroundedAnalysisDraft
from backend.app.agent.semantics import validate_text, validate_draft
from backend.app.agent.validation_diagnostics import SemanticValidationError, safe_excerpt
from tests.test_canonical_evidence import history, offline
from tests.test_agent_investigation import scripted, run, draft_from_evidence


@pytest.mark.parametrize("text", ["利润下降11.01%", "利润减少", "盈利能力下降", "利润承压", "利润损失约10万元",
    "应采取措施避免未来出现亏损", "当前尚未亏损", "利润仍为正", "没有亏损", "尚未亏损", "并非亏损",
    "避免亏损", "防止亏损", "若继续恶化可能亏损", "prevent future negative profit", "not loss-making"])
def test_legal_profit_language(history, text):
    validate_text(text, extract_evidence(history[:1]))


@pytest.mark.parametrize("text", ["公司已经亏损", "当前利润为负", "本期出现负利润", "negative profit",
    "the business is loss-making", "当前尚未亏损，但本期出现负利润", "应防止亏损，但公司已经亏损"])
def test_current_negative_profit_still_rejected(history, text):
    with pytest.raises(SemanticValidationError) as caught:
        validate_text(text, extract_evidence(history[:1]), field_path="findings[0].interpretation")
    diagnostic = caught.value.diagnostic
    assert diagnostic["rule_id"] == "PROFIT_SIGN_CONSISTENCY"
    assert diagnostic["field_path"] == "findings[0].interpretation"
    assert diagnostic["evidence_profit_sign"] == "POSITIVE"
    assert diagnostic["language_category"] == "CURRENT_NEGATIVE_PROFIT_ASSERTION"
    assert diagnostic["related_evidence_ids"] == list(extract_evidence(history[:1]))
    assert json.loads(json.dumps(diagnostic)) == diagnostic


@pytest.mark.parametrize("field", ["executive_interpretation", "findings[0].title", "findings[0].interpretation",
                                 "recommendations[0].title", "recommendations[0].action", "limitations[0]"])
def test_field_paths_and_no_full_draft_in_state(seeded_engine, field):
    def invalid(**context):
        draft = draft_from_evidence(**context)
        text = "当前利润为负"
        if field == "executive_interpretation": draft[field] = text
        elif field.startswith("findings"): draft["findings"][0][field.split('.')[-1]] = text
        elif field.startswith("recommendations"): draft["recommendations"][0][field.split('.')[-1]] = text
        else: draft["limitations"][0] = text
        return draft
    state = run(scripted(draft=invalid), seeded_engine)
    assert state["status"] == "ERROR" and state["analysis_draft"] is None
    assert state["llm_call_count"] == 11 and state["retry_count"] == 0
    diagnostic = state["errors"][-1]["validation_diagnostic"]
    assert diagnostic["field_path"] == field
    assert diagnostic["claim_excerpt"] == "当前利润为负"
    assert state["runtime_events"][-1]["event_type"] == "SEMANTIC_VALIDATION_REJECTED"
    assert "supporting_facts" not in json.dumps(diagnostic) and "executive_interpretation" not in diagnostic


def test_excerpt_bounded_and_secrets_redacted(history, monkeypatch):
    secret = "diagnostic-secret-value-987654321"
    monkeypatch.setenv("TEST_API_KEY", secret)
    text = "x" * 500 + " 公司已经亏损 " + secret + " Authorization: Bearer private-value"
    with pytest.raises(SemanticValidationError) as caught:
        validate_text(text, extract_evidence(history[:1]))
    diagnostic = caught.value.diagnostic
    serialized = json.dumps(diagnostic)
    assert len(diagnostic["claim_excerpt"]) <= 160
    assert secret not in serialized and "private-value" not in serialized
    assert "公司已经亏损" in diagnostic["claim_excerpt"]
    assert text not in serialized and "Authorization" not in serialized


def test_redaction_before_truncation(monkeypatch):
    secret = "short-private-marker"
    monkeypatch.setenv("TEST_SECRET", secret)
    assert secret not in safe_excerpt("a" * 150 + secret)
    assert "sk-" not in safe_excerpt("利润为负 sk-abcdefghijklmno")


@pytest.mark.parametrize("text,rule", [("公司利润同比下降", "COMPARISON_BASIS"),
    ("优惠收窄", "DISCOUNT_DIRECTION"), ("与产品结构无关", "EXCLUSION_CLAIM"), ("唯一原因", "ABSOLUTE_CAUSALITY")])
def test_other_semantic_results_preserved_with_diagnostics(history, text, rule):
    with pytest.raises(SemanticValidationError) as caught:
        validate_text(text, extract_evidence(history[:1]))
    assert caught.value.diagnostic["rule_id"] == rule


def test_preventive_recommendation_remains_valid_draft(seeded_engine):
    def draft(**context):
        result = draft_from_evidence(**context)
        result["recommendations"][0]["action"] = "应采取措施避免未来出现亏损"
        return result
    state = run(scripted(draft=draft), seeded_engine)
    assert state["status"] == "DRAFT_READY" and state["llm_call_count"] == 11
