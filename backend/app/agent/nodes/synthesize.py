from backend.app.agent.grounding import invalid_output
from backend.app.agent.guards.context import build_llm_context, visible_ids
from backend.app.agent.policy_evidence import validate_policy_recommendations
from backend.app.agent.evidence import extract_evidence, validate_citations
from backend.app.agent.semantics import validate_draft
from backend.app.agent.facts import assemble_draft
from backend.app.agent.nodes.common import model_node
from backend.app.agent.schemas import GroundedAnalysisDraft


def make_synthesize_node(gateway, *, strict=False):
    @model_node("synthesize", strict=strict)
    def synthesize(state):
        if state["status"] != "INVESTIGATION_COMPLETE" or not state["observations"]:
            invalid_output()
        draft = GroundedAnalysisDraft.model_validate(gateway.synthesize(
            state["goal"], state["observations"], state["tool_history"]))
        from backend.app.agent.guards.context import require_visible, current_manifest
        allowed = set(current_manifest("synthesize")["included_evidence_ids"])
        for finding in draft.findings:
            finding.supporting_evidence_ids = validate_citations(finding.supporting_evidence_ids, state["tool_history"])
            require_visible("synthesize", finding.supporting_evidence_ids, kind="BUSINESS_DATA")
        for recommendation in draft.recommendations:
            require_visible("synthesize", recommendation.policy_evidence_ids, kind="POLICY")
        evidence = extract_evidence(state["tool_history"])
        policies = validate_policy_recommendations(draft, state["tool_history"], allowed)
        validate_draft(draft, evidence)
        result = assemble_draft(draft, evidence)
        for recommendation in result["recommendations"]:
            recommendation["policy_citations"] = [policies[eid]["citation"] for eid in recommendation["policy_evidence_ids"]]
        if not policies:
            result["policy_coverage"] = "当前知识库没有本次检索取得的政策依据；不能据此声称公司制度要求。"
        return {"analysis_draft": result, "status": "DRAFT_READY"}
    return synthesize
