"""Run-local policy citations, isolated from all numeric business validators."""
import re
from backend.app.agent.grounding import invalid_output
from backend.app.rag.schemas import PolicyEvidence, PolicySearchResult

MAX_POLICY_HITS_IN_LLM_CONTEXT = 5


def policy_catalog(history):
    result = {}
    for record in history:
        if record["tool_name"] != "search_sales_policy" or not record["success"]:
            continue
        data = PolicySearchResult.model_validate(record["data"])
        for hit in data.hits:
            citation = hit.citation
            eid = citation.policy_evidence_id
            if eid != "POLICY::"+hit.chunk_id or eid not in record["evidence_ids"] or citation.chunk_id != hit.chunk_id:
                invalid_output()
            item = PolicyEvidence(evidence_id=eid, excerpt=hit.excerpt, citation=citation).model_dump(mode="json")
            if eid in result and result[eid] != item:
                invalid_output()
            result[eid] = item
    return result


def policy_view(node, history, observations):
    selected = {}
    for observation in observations:
        for item in observation.get("policy_evidence", []):
            selected[item["evidence_id"]] = item
    if node == "review_progress":
        # Latest hits first so policy tools can expose new evidence; prior selections fill remaining capacity.
        latest = policy_catalog(history[-1:])
        selected = {**latest, **{eid: item for eid, item in selected.items() if eid not in latest}}
    return [{**item, "trust_boundary": "UNTRUSTED_RETRIEVED_CONTENT"} for item in list(selected.values())[:MAX_POLICY_HITS_IN_LLM_CONTEXT]]


def validate_policy_recommendations(draft, history, allowed):
    catalog = policy_catalog(history)
    for recommendation in draft.recommendations:
        ids = recommendation.policy_evidence_ids
        if any(eid not in catalog or eid not in allowed for eid in ids):
            invalid_output()
        if bool(ids) != bool(recommendation.policy_interpretation):
            invalid_output()
        # Bounded unsupported-policy assertion guard; semantic entailment remains an evaluation concern.
        text = recommendation.title+recommendation.action
        if not ids and re.search(r"公司规定|制度规定|按照.{0,12}制度|政策要求|须.{0,12}审批|\bcompany policy requires\b", text, re.I):
            invalid_output()
    return catalog
