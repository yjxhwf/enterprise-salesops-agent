"""Whole-item budgets, node policies and immutable JSON visibility snapshots."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum, StrEnum
import hashlib
import json
from uuid import uuid4
from typing import TypedDict
from backend.app.llm.gateway import ErrorCode, GatewayError


class ContextManifest(TypedDict):
    context_id: str
    node: str
    projected_char_count: int
    included_sections: list[str]
    included_evidence_ids: list[str]
    included_fact_ids: list[str]
    included_business_evidence_ids: list[str]
    included_policy_evidence_ids: list[str]
    omitted_item_counts: dict[str, int]
    included_item_count: int
    context_fingerprint: str
    schema_chars: int


class ContextSection(StrEnum):
    GOAL = "goal"
    CURRENT_STEP = "current_step"
    PLAN = "pending_objectives"
    LATEST_TOOL_RESULT = "result_digest"
    LATEST_FACTS = "latest_evidence"
    PRIOR_SELECTED_EVIDENCE = "selected_evidence"
    POLICY_EVIDENCE = "policy_evidence"
    EXECUTIVE_FACTS = "executive_facts"
    COMPLETED_OBJECTIVES = "completed_objectives"
    RECOMMENDATIONS = "recommendations"
    WRITE_TOOL_SCHEMAS = "available_write_tools"
    RUNTIME_BUDGET_SUMMARY = "runtime_budget"


class Priority(IntEnum):
    MANDATORY = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(frozen=True)
class ContextPolicy:
    char_budget: int
    mandatory: frozenset[str]
    allowed: frozenset[str]


def policy(budget, mandatory, optional=()):
    return ContextPolicy(budget, frozenset(mandatory), frozenset((*mandatory, *optional)))


POLICIES = {
    "understand_goal": policy(8000, ("user_query",)),
    "plan": policy(16000, ("goal", "capabilities")),
    "select_tool": policy(24000, ("goal", "current_step", "pending_objectives"), ("selected_evidence", "policy_evidence")),
    "review_progress": policy(48000, ("goal", "current_step", "pending_objectives", "latest_source_tool", "latest_evidence", "result_digest", "runtime_budget"), ("prior_selected_evidence", "policy_evidence")),
    "synthesize": policy(64000, ("goal", "executive_facts", "completed_objectives", "synthesis_coverage"), ("selected_evidence", "policy_evidence")),
    "propose_action": policy(48000, ("goal", "findings", "recommendations", "available_write_tools"), ("selected_business_evidence", "selected_policy_evidence")),
}


@dataclass(frozen=True)
class ContextItem:
    section: str
    priority: Priority
    stable_id: str
    payload: object
    evidence_ids: tuple[str, ...]
    estimated_chars: int


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def char_count(value):
    # Match the actual human-message JSON encoding; fingerprint uses canonical JSON.
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False))


class ContextBudgetExceeded(GatewayError):
    def __init__(self, node, budget, mandatory, counts):
        super().__init__(ErrorCode.CONTEXT_BUDGET_EXCEEDED)
        self.context_diagnostic = dict(node=node, budget_chars=budget, mandatory_chars=mandatory, section_counts=dict(counts))


BUNDLE_SECTIONS = {"latest_evidence", "prior_selected_evidence", "selected_evidence", "executive_facts", "policy_evidence", "selected_business_evidence", "selected_policy_evidence"}


def assemble(node, candidate, *, schema_chars=0, budget=None):
    config = POLICIES[node]
    if set(candidate)-config.allowed:
        raise ValueError("Unexpected context section")
    missing = config.mandatory-candidate.keys()
    if missing:
        raise ValueError("Missing mandatory context section")
    budget = config.char_budget if budget is None else budget
    payload = deepcopy(candidate)
    # Deduplicate exact evidence IDs globally, retaining all distinct facts once.
    seen, fact_ids = {}, {}
    supporting_ids = {f["fact_id"] for b in payload.get("selected_evidence", []) for f in b.get("facts", [])}
    for section in sorted(BUNDLE_SECTIONS & payload.keys(), key=lambda s: (s not in config.mandatory, s)):
        unique = []
        for bundle in payload[section]:
            eid = bundle["evidence_id"]
            if "facts" in bundle:
                unique_facts = []
                for fact in bundle["facts"]:
                    fid = fact["fact_id"]
                    if fid in fact_ids and fact_ids[fid] != fact:
                        raise ValueError("Conflicting fact identity")
                    if fid not in fact_ids:
                        unique_facts.append(fact)
                        fact_ids[fid] = fact
                bundle["facts"] = unique_facts
            if eid in seen:
                if "facts" not in bundle and seen[eid] != bundle:
                    raise ValueError("Conflicting evidence identity")
                if "facts" in bundle:seen[eid].setdefault("facts", []).extend(bundle["facts"])
                continue
            seen[eid] = bundle
            unique.append(bundle)
        payload[section] = unique
    items = []
    list_sections = set()
    for section, value in payload.items():
        parts = value if isinstance(value, list) else [value]
        if isinstance(value, list):list_sections.add(section)
        for index, part in enumerate(parts):
            eid = part.get("evidence_id") if isinstance(part, dict) else None
            priority = Priority.MANDATORY if section in config.mandatory else Priority.LOW if section == "prior_selected_evidence" else Priority.MEDIUM
            if section in {"policy_evidence", "selected_policy_evidence"}:
                # Keep at least one relevant policy bundle, including its untrusted label.
                priority = Priority.MANDATORY if index == 0 or node == "propose_action" else Priority.HIGH
            if node == "propose_action" and section == "selected_business_evidence":
                priority = Priority.MANDATORY  # findings' cited support cannot silently disappear
            items.append(ContextItem(section, priority, eid or f"{section}:{index:05d}", part,
                                     (eid,) if eid else (), char_count(part)))
    def render(chosen):
        result = {k: [] for k in list_sections}
        for item in chosen:
            if item.section in list_sections:result[item.section].append(item.payload)
            else:result[item.section]=item.payload
        wire = deepcopy(result)
        if "synthesis_coverage" in wire:
            visible_support = {f["fact_id"] for item in chosen if item.section in BUNDLE_SECTIONS for f in item.payload.get("facts", [])} & supporting_ids
            coverage = wire["synthesis_coverage"]
            coverage["visible_synthesis_fact_count"] = len(visible_support)
            coverage["omitted_synthesis_fact_count"] = coverage["total_selected_fact_count"]-len(visible_support)
        for section in BUNDLE_SECTIONS & wire.keys():
            for bundle in wire[section]:
                for fact in bundle.get("facts", []):fact.pop("fact_id", None)
        return wire
    mandatory = [i for i in items if i.priority == Priority.MANDATORY]
    mandatory_chars = char_count(render(mandatory)) + schema_chars
    if mandatory_chars > budget:
        raise ContextBudgetExceeded(node, budget, mandatory_chars, Counter(i.section for i in mandatory))
    included = list(mandatory)
    omitted = Counter()
    # Candidate order already uses evidence usage rank; stable ID resolves ties upstream.
    for item in sorted((i for i in items if i.priority != Priority.MANDATORY), key=lambda i: i.priority):
        if char_count(render(included+[item])) + schema_chars <= budget:included.append(item)
        else:omitted[item.section]+=1
    result = render(included)
    bundles = [v for k, values in result.items() if k in BUNDLE_SECTIONS for v in values]
    business = sorted({v["evidence_id"] for v in bundles if not v["evidence_id"].startswith(("POLICY::", "ACTION::"))})
    policies = sorted({v["evidence_id"] for v in bundles if v["evidence_id"].startswith("POLICY::")})
    facts = sorted({f["fact_id"] for item in included if item.section in BUNDLE_SECTIONS for f in item.payload.get("facts", [])})
    digest = result.get("result_digest", {})
    if digest.get("omitted_row_count"):
        omitted["latest_result_rows"] += digest["omitted_row_count"]
    coverage = result.get("synthesis_coverage", {})
    if coverage.get("omitted_synthesis_fact_count"):
        omitted["supporting_facts"] += coverage["omitted_synthesis_fact_count"]
    manifest: ContextManifest = dict(context_id=str(uuid4()), node=node, projected_char_count=char_count(result)+schema_chars,
        included_sections=sorted(result), included_evidence_ids=sorted(business+policies), included_fact_ids=facts,
        included_business_evidence_ids=business, included_policy_evidence_ids=policies,
        omitted_item_counts=dict(omitted), included_item_count=len(included),
        context_fingerprint=hashlib.sha256(canonical(result).encode()).hexdigest(), schema_chars=schema_chars)
    return result, manifest


def visible_or_reject(ids, manifest, *, kind=None):
    field = {"BUSINESS_DATA":"included_business_evidence_ids", "POLICY":"included_policy_evidence_ids"}.get(kind, "included_evidence_ids")
    if manifest is None or not set(ids) <= set(manifest[field]):
        raise GatewayError(ErrorCode.EVIDENCE_NOT_VISIBLE_IN_CONTEXT)
