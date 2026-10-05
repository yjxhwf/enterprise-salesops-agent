from copy import deepcopy
import json

import pytest

from backend.app.agent.evidence import extract_evidence
from backend.app.agent.semantics import validate_text, profit_claim_category
from backend.app.agent.validation_diagnostics import SemanticValidationError
from tests.test_canonical_evidence import history, offline
from tests.test_agent_investigation import scripted, run, draft_from_evidence


@pytest.mark.parametrize("text", [
    "但不代表该产品本身亏损", "不代表该产品亏损", "这并不代表该产品本身亏损",
    "不能说明公司已经亏损", "不能据此认为产品处于亏损状态", "现有证据不足以证明该产品亏损",
    "利润下降，但不代表当前利润为负", "利润下降不意味着公司已经亏损", "并不意味着公司亏损",
    "不能表明该产品本身亏损", "无法说明当前处于亏损状态", "无法证明该产品本身亏损",
    "不足以说明当前处于亏损状态", "不能据此判断公司亏损", "该产品没有亏损", "该产品并未亏损",
    "This does not mean the product is loss-making.", "This doesn't mean the product is loss-making.",
    "This does not imply negative profit.", "This does not indicate current negative profit.",
    "We cannot conclude that the business is loss-making.", "We cannot infer negative profit.",
    "This does not prove the product is loss-making.", "The product is not loss-making.",
    "The profit declined, but it is not negative.", "This does not mean current profit is negative.",
    "如果趋势继续恶化，未来可能出现亏损", "建议控制风险，避免未来亏损", "若折扣继续扩大，可能进入负利润状态",
    "应避免后续出现亏损", "防止业务进入负利润状态", "利润下降", "利润减少", "利润损失约10万元", "盈利能力承压",
])
def test_negation_and_noncurrent_language_is_accepted(history, text):
    catalog = extract_evidence(history[:1])
    before = deepcopy(catalog)
    validate_text(text, catalog)
    assert catalog == before


@pytest.mark.parametrize("text", [
    "该产品本身亏损", "公司已经亏损", "当前利润为负", "本期处于负利润状态",
    "利润下降，并且公司已经亏损", "不能说明销量下降，但该产品本身亏损",
    "虽然收入增长，但公司实际已经亏损", "不代表销量下降，但该产品本身亏损。",
    "虽然不能说明收入问题，但公司已经亏损。", "并不意味着利润下降，实际公司已经亏损。",
    "不代表销量下降但是该产品亏损", "不能说明收入问题不过公司已经亏损",
    "无法证明订单问题然而公司已经亏损", "不能说明公司亏损。该产品亏损",
    "不能说明公司亏损；当前利润为负", "不能说明公司亏损！该产品亏损",
    "The product is loss-making.", "The business currently has negative profit.", "Current profit is negative.",
    "Revenue improved, but the company is loss-making.",
    "This does not mean lower sales. The product is loss-making.",
    "This does not mean lower sales but the product is loss-making.",
    "We cannot infer lower sales; however the business has negative profit.",
])
def test_assertions_and_scope_boundaries_reject(history, text):
    catalog = extract_evidence(history[:1])
    with pytest.raises(SemanticValidationError) as caught:
        validate_text(text, catalog, field_path="findings[0].interpretation")
    diagnostic = caught.value.diagnostic
    assert diagnostic["rule_id"] == "PROFIT_SIGN_CONSISTENCY"
    assert diagnostic["language_category"] == "CURRENT_NEGATIVE_PROFIT_ASSERTION"
    assert diagnostic["field_path"] == "findings[0].interpretation"
    assert diagnostic["evidence_profit_sign"] == "POSITIVE"
    assert diagnostic["related_evidence_ids"] == list(catalog)
    assert len(diagnostic["claim_excerpt"]) <= 160
    assert json.loads(json.dumps(diagnostic)) == diagnostic


def test_real_live_sentence_survives_full_graph_unchanged(seeded_engine):
    sentence = "但不代表该产品本身亏损"
    def draft(**context):
        result = draft_from_evidence(**context)
        result["findings"][0]["interpretation"] = sentence
        return result
    state = run(scripted(draft=draft), seeded_engine)
    assert state["status"] == "DRAFT_READY"
    assert state["analysis_draft"]["findings"][0]["interpretation"] == sentence
    assert state["llm_call_count"] == 11 and state["retry_count"] == 0
    assert not any(e["event_type"] == "SEMANTIC_VALIDATION_REJECTED" for e in state["runtime_events"])


def test_bounded_classification():
    assert profit_claim_category("不代表该产品本身") == "NEGATED_NEGATIVE_PROFIT_ASSERTION"
    assert profit_claim_category("防止业务进入") == "HYPOTHETICAL_OR_PREVENTIVE_LOSS"
    assert profit_claim_category("不代表" + "长文本" * 30) == "CURRENT_NEGATIVE_PROFIT_ASSERTION"
