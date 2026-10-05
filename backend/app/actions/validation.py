"""Revalidate grounded intent; no generated identifiers can authorize a write."""
import hashlib
import json
import re
from pydantic import ValidationError
from backend.app.actions.schemas import ActionProposal, ActionError
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.facts import canonical_facts
from backend.app.agent.policy_evidence import policy_catalog
from backend.app.agent.guards.context import build_llm_context, visible_ids
from backend.app.tools.write_schemas import WRITE_MODELS
from backend.app.actions.diagnostics import reject, diagnostic, pydantic_diagnostic


def fingerprint(tool_name, arguments, scope):
    canonical = json.dumps([tool_name, arguments, scope], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_capabilities(registry):
    return [m.model_dump(mode="json") for m in registry.list_tools(permission="WRITE_APPROVAL_REQUIRED")]


def validate_tool(registry, tool_name, arguments):
    metadata = registry.get_tool(tool_name)
    if metadata is None:
        reject("proposal.tool_name", "UNKNOWN_WRITE_TOOL", "registered_write_tool", tool_name)
    if metadata.permission_level != "WRITE_APPROVAL_REQUIRED":
        reject("proposal.tool_name", "INVALID_TOOL_PERMISSION", "WRITE_APPROVAL_REQUIRED", tool_name)
    if tool_name not in WRITE_MODELS:
        reject("proposal.tool_name", "UNKNOWN_WRITE_TOOL", "supported_write_tool", tool_name)
    model = WRITE_MODELS[tool_name]
    if metadata.input_schema != model.model_json_schema():
        reject("proposal.tool_name", "TOOL_SCHEMA_MISMATCH", "canonical_write_schema", metadata.input_schema)
    try:
        return model.model_validate(arguments).model_dump(mode="json")
    except ValidationError as error:
        wrapped = ActionError("ACTION_PROPOSAL_VALIDATION_ERROR")
        wrapped.action_proposal_diagnostic = pydantic_diagnostic(error, arguments, prefix="proposal.arguments", schema=model)
        raise wrapped from None


def proposal_context(state, registry):
    from backend.app.agent.guards.context import action_context
    return action_context(state, registry)


def validate_proposal(proposal, state, registry):
    raw = proposal
    try:
        proposal = ActionProposal.model_validate(proposal)
    except ValidationError as error:
        error.action_proposal_diagnostic = pydantic_diagnostic(error, raw, prefix="proposal", schema=ActionProposal)
        raise
    args = validate_tool(registry, proposal.tool_name, proposal.arguments)
    if proposal.action_type != {"create_crm_task": "CREATE_CRM_TASK", "create_business_alert": "CREATE_BUSINESS_ALERT"}[proposal.tool_name]:
        reject("proposal.action_type", "ACTION_TOOL_TYPE_MISMATCH", "action_type_matching_tool", proposal.action_type)
    findings = state["analysis_draft"]["findings"]
    indexes = proposal.related_finding_indexes
    if any(type(i) is not int or i < 0 or i >= len(findings) for i in indexes):
        reject("proposal.related_finding_indexes", "INVALID_FINDING_INDEX", "existing_finding_indexes", indexes)
    business = extract_evidence(state["tool_history"])
    policies = policy_catalog(state["tool_history"])
    from backend.app.agent.guards.context import current_manifest
    from backend.app.agent.context.assembly import visible_or_reject
    if not set(proposal.business_evidence_ids) <= business.keys():
        reject("proposal.business_evidence_ids", "INVALID_BUSINESS_EVIDENCE_ID", "registered_business_evidence_ids", proposal.business_evidence_ids)
    if not set(proposal.policy_evidence_ids) <= policies.keys():
        reject("proposal.policy_evidence_ids", "INVALID_POLICY_EVIDENCE_ID", "registered_policy_evidence_ids", proposal.policy_evidence_ids)
    manifest = current_manifest("propose_action") or next((m for m in reversed(state.get("context_manifests", [])) if m["node"] == "propose_action"), None)
    from backend.app.llm.gateway import GatewayError
    for field, kind in (("business_evidence_ids", "BUSINESS_DATA"), ("policy_evidence_ids", "POLICY")):
        ids = getattr(proposal, field)
        try:
            visible_or_reject(ids, manifest, kind=kind)
        except GatewayError as error:
            error.action_proposal_diagnostic = diagnostic("proposal."+field, "EVIDENCE_NOT_VISIBLE", "request_visible_"+kind, ids)
            raise
    allowed = set(manifest["included_evidence_ids"])
    linked = {eid for i in indexes for eid in findings[i]["supporting_evidence_ids"]}
    if not set(proposal.business_evidence_ids) <= business.keys() & linked & allowed:
        reject("proposal.business_evidence_ids", "EVIDENCE_NOT_LINKED_TO_FINDING", "finding_linked_visible_business_ids", proposal.business_evidence_ids)
    if not set(proposal.policy_evidence_ids) <= policies.keys() & allowed:
        reject("proposal.policy_evidence_ids", "INVALID_POLICY_EVIDENCE_SCOPE", "visible_policy_ids", proposal.policy_evidence_ids)
    text = proposal.reason + " " + proposal.expected_outcome + " " + json.dumps(args, ensure_ascii=False)
    policy_claim = proposal.policy_basis or re.search(r"公司规定|制度规定|根据.{0,12}政策|政策要求|company policy requires", text, re.I)
    if policy_claim and not proposal.policy_evidence_ids:
        reject("proposal.policy_evidence_ids", "POLICY_EVIDENCE_REQUIRED", "nonempty_policy_evidence_ids", proposal.policy_evidence_ids)
    # Customer/target must actually be represented by cited business evidence.
    selected = [business[eid] for eid in proposal.business_evidence_ids]
    target_type = "CUSTOMER" if proposal.tool_name == "create_crm_task" else args["target_type"]
    target_id = args.get("customer_id", args.get("target_id"))
    facts = canonical_facts(business, proposal.business_evidence_ids)
    subject_match = any(f["subject_type"] == target_type and f["subject_id"] == target_id for f in facts)
    raw_match = any(e["internal_reference"]["raw_value"].get(
        {"CUSTOMER": "customer_id", "PRODUCT": "product_id", "REGION": "region"}.get(target_type, "__none__")) == target_id for e in selected)
    if not (subject_match or raw_match):
        reject("proposal.arguments."+("customer_id" if proposal.tool_name == "create_crm_task" else "target_id"), "TARGET_NOT_GROUNDED_IN_EVIDENCE", "target_supported_by_cited_business_evidence", target_id)
    return proposal.model_copy(update={"arguments": args})
