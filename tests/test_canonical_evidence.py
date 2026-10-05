from copy import deepcopy
import json
import socket

import pytest

from backend.app.agent.evidence import display_value, extract_evidence, visible_evidence, validate_citations
from backend.app.agent.schemas import GroundedAnalysisDraft, ProgressReview
from backend.app.data.database import session_factory
from backend.app.llm.gateway import GatewayError
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import DATES
from tests.test_agent_investigation import run, scripted


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Canonical evidence tests must remain offline")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def history(seeded_engine):
    records = []
    with build_default_registry(session_factory(seeded_engine)) as registry:
        for tool in ("get_sales_overview", "analyze_region_performance", "analyze_product_performance", "query_orders"):
            result = registry.invoke(tool, DATES).model_dump(mode="json")
            records.append({**result, "arguments": DATES})
    return records


def test_extraction_from_real_results_is_deterministic(history):
    original = deepcopy(history)
    assert extract_evidence(history) == extract_evidence(deepcopy(history))
    assert history == original
    assert len(extract_evidence(history)) == 1 + 4 + 12 + 20


def test_internal_reference_resolves_exact_raw_value(history):
    for item in extract_evidence(history).values():
        ref = item["internal_reference"]
        record = next(r for r in history if ref["source_evidence_id"] in r["evidence_ids"])
        value = record
        for part in ref["json_pointer"].split("/")[1:]:
            value = value[int(part)] if isinstance(value, list) else value[part]
        assert value == ref["raw_value"]
        assert item["display_value"] == display_value(value)


def test_display_precision_and_units(history):
    original = history[0]["data"]["changes"]["profit_growth_pct"]
    assert len(original) > 10
    visible = visible_evidence(history)[0]["display_value"]
    assert visible["current"]["revenue"] == "3150655.70"
    assert visible["current"]["margin"] == "27.62%"
    assert visible["current"]["order_count"] == 667
    assert visible["changes"]["profit_growth_pct"] == "-11.01%"
    assert visible["changes"]["margin_change_pp"] == "-6.39 pp"
    assert original not in json.dumps(visible_evidence(history))
    assert extract_evidence(history)[history[0]["evidence_ids"][0]]["internal_reference"]["raw_value"]["changes"]["profit_growth_pct"] == original


def test_discount_format_and_nonmetrics():
    assert display_value("0.954285714285714", "avg_discount") == "0.9543"
    assert display_value("0.7500", "discount_rate") == "0.7500"
    assert display_value("001", "customer_id") == "001"
    assert display_value(None, "margin") is None
    assert display_value("0", "profit_growth_pct") == "0.00%"


def test_llm_contract_has_no_internal_fields(history):
    for schema in (ProgressReview, GroundedAnalysisDraft):
        serialized = json.dumps(schema.model_json_schema())
        for field in ('"json_pointer"', '"raw_value"', '"references"', '"source_tool"'):
            assert field not in serialized
    for item in visible_evidence(history):
        assert set(item) == {"evidence_id", "source_tool", "summary", "display_value",
                             "comparison_basis", "comparison_label", "current_period", "comparison_period", "rendered_facts"}


def test_id_only_model_output_completes(seeded_engine):
    state = run(scripted(), seeded_engine)
    assert state["status"] == "DRAFT_READY"
    for finding in state["analysis_draft"]["findings"]:
        assert set(finding) == {"claim_type", "title", "interpretation", "supporting_evidence_ids", "supporting_facts"}
        assert validate_citations(finding["supporting_evidence_ids"], state["tool_history"])


@pytest.mark.parametrize("kind", ["invented", "typo", "empty", "other_run"])
def test_invalid_ids_never_resolved_by_guessing(history, kind):
    eid = next(iter(extract_evidence(history)))
    other = deepcopy(history[:1])
    other[0]["evidence_ids"] = ["aggregate:another-scope"]
    ids = {"invented": ["fabricated"], "typo": [eid + "x"], "empty": [],
           "other_run": list(extract_evidence(other))}[kind]
    with pytest.raises(GatewayError) as caught:
        validate_citations(ids, history)
    assert caught.value.code == "MODEL_OUTPUT_VALIDATION_ERROR"


def test_scope_and_entity_ids_do_not_collide(history):
    one = extract_evidence(history[1:2])
    assert len(set(one)) == 4
    other = deepcopy(history[1:2])
    other[0]["evidence_ids"] = ["aggregate:different-result-scope"]
    assert set(one).isdisjoint(extract_evidence(other))


def test_existing_order_ids_preserved(history):
    catalog = extract_evidence(history[-1:])
    assert set(catalog) == set(history[-1]["evidence_ids"])
    for eid, item in catalog.items():
        assert eid == "order:" + item["internal_reference"]["raw_value"]["order_id"]


def test_failed_results_create_no_evidence(history):
    record = deepcopy(history[0])
    record["success"] = False
    assert extract_evidence([record]) == {}


def test_extractor_does_not_load_external_answers(history, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Extraction may only inspect supplied tool results")
    monkeypatch.setattr("builtins.open", forbidden)
    assert extract_evidence(history)
    import inspect
    from backend.app.agent import evidence
    assert "ground_truth" not in inspect.getsource(evidence)
