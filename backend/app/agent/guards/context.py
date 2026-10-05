from contextvars import ContextVar
from copy import deepcopy
import json


active_runtime = ContextVar("salesops_runtime", default=None)
MAX_LLM_VISIBLE_ROWS_PER_RESULT = 12
MAX_LLM_VISIBLE_ORDER_ROWS = 10
MAX_PRIOR_SELECTED_FACTS = 24
MAX_SYNTHESIS_SUPPORTING_FACTS = 48


def bounded_facts(facts, limit, observations=()):
    from backend.app.agent.context.evidence import selection_rank
    rank = selection_rank(observations)
    """Round-robin across citations before adding more measurements per entity."""
    priority = {name: i for i, name in enumerate(("profit", "revenue", "margin", "quantity", "revenue_share", "price_realization_ratio", "discount_depth"))}
    groups = {}
    for fact in sorted(facts, key=lambda f: (rank(f["evidence_id"]), priority.get(f.get("semantic_metric") or f["metric"], 99), f["fact_id"])):
        groups.setdefault(fact["evidence_id"], []).append(fact)
    result = []
    for depth in range(max((len(g) for g in groups.values()), default=0)):
        for group in groups.values():
            if depth < len(group):
                result.append(group[depth])
    return result[:limit]


def result_view(history, observations):
    from backend.app.agent.evidence import extract_evidence
    record = history[-1] if history else {}
    latest = extract_evidence(history[-1:])
    ids = list(latest)  # Adapter preserves the tool's stable row order.
    is_orders = any(e["internal_reference"]["json_pointer"].startswith("/data/orders/") for e in latest.values())
    cap = MAX_LLM_VISIBLE_ORDER_ROWS if is_orders else MAX_LLM_VISIBLE_ROWS_PER_RESULT
    chosen = selected_ids(observations)
    ordered = [eid for eid in ids if eid in chosen] + [eid for eid in ids if eid not in chosen]
    visible = ordered[:cap]
    meta = record.get("meta", {})
    digest = dict(source_tool=record.get("tool_name"), query_scope=record.get("arguments", {}),
                  matched_count=meta.get("total_matches"), returned_count=len(ids),
                  truncated=meta.get("truncated"), visible_row_count=len(visible),
                  omitted_row_count=len(ids)-len(visible), high_cardinality=len(ids)>cap)
    return {eid: latest[eid] for eid in visible}, digest


def visible_ids(payload):
    return {b["evidence_id"] for value in payload.values() if isinstance(value, list)
            for b in value if isinstance(b, dict) and "evidence_id" in b}


def budget_summary():
    runtime = active_runtime.get()
    if not runtime:
        return {}
    state, limits = runtime.state, runtime.limits
    result = {}
    for name, field, maximum in (("tool_calls", "tool_call_count", limits.max_tool_calls),
                                  ("llm_calls", "llm_call_count", limits.max_llm_calls),
                                  ("agent_steps", "agent_step_count", limits.max_agent_steps)):
        used = state[field]
        result.update({name+"_used": used, name+"_remaining": max(0, maximum-used)})
    usage = state["token_usage"]
    if usage["usage_available"]:
        result.update(soft_token_budget_used=usage["total_tokens"],
                      soft_token_budget_remaining=max(0, limits.soft_token_budget-usage["total_tokens"]))
    return result


class RuntimeStop(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason.value)


def compact_facts(facts):
    """Group shared citation metadata; each measurement has only rendered text."""
    groups, seen = {}, set()
    for fact in sorted(facts, key=lambda f: f["fact_id"]):
        if fact["fact_id"] in seen:
            continue
        seen.add(fact["fact_id"])
        eid = fact["evidence_id"]
        group = groups.setdefault(eid, dict(evidence_id=eid, source_tool=fact["source_tool"],
            subject=dict(type=fact["subject_type"], id=fact["subject_id"]), facts=[]))
        group["facts"].append(dict(fact_id=fact["fact_id"], metric=fact.get("semantic_metric") or fact["metric"], rendered_text=fact["rendered_text"]))
    return list(groups.values())


def objective(step):
    return {k: step[k] for k in ("objective", "expected_output") if k in step}


def selected_ids(observations):
    return {eid for observation in observations for eid in observation.get("evidence_ids", []) if not eid.startswith("POLICY::")}


def project_context(node, *, user_query=None, goal=None, plan=None, step=None,
                      observations=(), history=(), capabilities=(), emit_event=True):
    """Deterministic node views. Never return State, raw history, or plan versions."""
    from backend.app.agent.evidence import extract_evidence
    from backend.app.agent.facts import canonical_facts
    from backend.app.agent.context.evidence import EvidenceLedger, selection_rank
    ledger = EvidenceLedger(dict(tool_history=history, observations=observations)) if node in {"review_progress", "synthesize"} else None
    if node == "understand_goal":
        payload = {"user_query": user_query}
    elif node == "plan":
        payload = {"goal": goal, "capabilities": [{k: item[k] for k in ("name", "description", "permission_level") if k in item} for item in capabilities]}
    elif node in {"select_tool", "review_progress"}:
        pending = [objective(s) for s in (plan or {}).get("steps", [])
                   if s.get("status") in {"PENDING", "IN_PROGRESS"} and s != step]
        payload = dict(goal=goal, current_step=objective(step or {}), pending_objectives=pending)
        if node == "select_tool":
            latest = next((o for o in reversed(observations) if o.get("facts")), None)
            payload["selected_evidence"] = compact_facts(bounded_facts(latest["facts"], MAX_PRIOR_SELECTED_FACTS, observations)) if latest else []
        else:
            catalog = {eid: ledger.resolve(eid, "BUSINESS_DATA") for eid in ledger.ids("BUSINESS_DATA")}
            latest, digest = result_view(history, observations)
            prior = selected_ids(observations) - latest.keys()
            payload.update(latest_source_tool=history[-1]["tool_name"] if history else None,
                           latest_evidence=compact_facts(canonical_facts(latest)),
                           prior_selected_evidence=compact_facts(bounded_facts(canonical_facts(catalog, sorted(prior)), MAX_PRIOR_SELECTED_FACTS, observations)),
                           result_digest=digest, runtime_budget=budget_summary())
    elif node == "synthesize":
        catalog = {eid: ledger.resolve(eid, "BUSINESS_DATA") for eid in ledger.ids("BUSINESS_DATA")}
        all_facts = canonical_facts(catalog)
        executive = [f for f in all_facts if f["subject_type"] == "COMPANY" and f["metric"] in {"revenue", "profit", "margin"}]
        executive_ids = {f["fact_id"] for f in executive}
        selected = selected_ids(observations)
        supporting = [f for f in all_facts if f["evidence_id"] in selected and f["fact_id"] not in executive_ids]
        completed = []
        for o in observations:
            for s in o.get("plan_before", {}).get("steps", []):
                if s.get("step_id") == o.get("completed_step_id"):
                    completed.append(s["objective"])
                    break
        visible_supporting = bounded_facts(supporting, MAX_SYNTHESIS_SUPPORTING_FACTS, observations)
        payload = dict(goal=goal, executive_facts=compact_facts(executive),
                       selected_evidence=compact_facts(visible_supporting), completed_objectives=completed,
                       synthesis_coverage=dict(total_selected_fact_count=len(supporting),
                           visible_synthesis_fact_count=len(visible_supporting),
                           omitted_synthesis_fact_count=len(supporting)-len(visible_supporting)))
    else:
        raise ValueError("Unknown context view")
    if node in {"select_tool", "review_progress", "synthesize"}:
        from backend.app.agent.policy_evidence import policy_catalog, policy_view, MAX_POLICY_HITS_IN_LLM_CONTEXT
        selected_policies = {item["evidence_id"]: item for observation in observations for item in observation.get("policy_evidence", [])}
        latest_policies = policy_catalog(history[-1:]) if node == "review_progress" else {}
        selected_policies.update(latest_policies)
        rank = selection_rank(observations)
        ordered_policy_ids = sorted(selected_policies, key=lambda eid: (eid not in latest_policies, rank(eid)))
        policies = [{**selected_policies[eid], "trust_boundary": "UNTRUSTED_RETRIEVED_CONTENT"} for eid in ordered_policy_ids[:MAX_POLICY_HITS_IN_LLM_CONTEXT]]
        if policies:
            payload["policy_evidence"] = policies
        if node == "review_progress" and history and history[-1]["tool_name"] == "search_sales_policy":
            latest_policies = policy_catalog(history[-1:])
            visible = sum(item["evidence_id"] in latest_policies for item in policies)
            payload["result_digest"].update(evidence_kind="POLICY", returned_count=len(latest_policies),
                visible_row_count=visible, omitted_row_count=len(latest_policies)-visible,
                high_cardinality=len(latest_policies)>MAX_POLICY_HITS_IN_LLM_CONTEXT)
    from backend.app.agent.context.evidence import selection_rank
    rank = selection_rank(observations)
    for section in ("selected_evidence", "prior_selected_evidence"):
        if section in payload:payload[section].sort(key=lambda b: rank(b["evidence_id"]))
    return deepcopy(payload)


def publish_context(node, payload, *, emit_event=True, tools=()):
    from backend.app.agent.context.assembly import assemble, char_count, canonical
    import hashlib
    projected, manifest = assemble(node, payload, schema_chars=char_count(tools) if tools else 0)
    # Native READ schemas also affect request identity and structural cost.
    if tools:
        manifest["context_fingerprint"] = hashlib.sha256(canonical(dict(context=projected, tools=tools)).encode()).hexdigest()
    runtime = active_runtime.get()
    if runtime and emit_event:
        runtime.prepared_manifest = manifest
        runtime.prepared_payload = projected
        detail = {k:v for k,v in projected.get("result_digest", {}).items() if k != "query_scope"}
        detail.update(projected.get("synthesis_coverage", {}))
        runtime.event("CONTEXT_PROJECTION", **detail,
            included_policy_hit_count=len(manifest["included_policy_evidence_ids"]),
            included_fact_count=len(manifest["included_fact_ids"]),
            included_evidence_count=len(manifest["included_evidence_ids"]),
            included_plan_step_count=len(projected.get("pending_objectives", []))+int("current_step" in projected),
            serialized_context_chars=manifest["projected_char_count"])
    return projected


def build_llm_context(node, *, tools=(), emit_event=True, **kwargs):
    payload = project_context(node, emit_event=False, **kwargs)
    return publish_context(node, payload, emit_event=emit_event, tools=tools)


def current_manifest(node):
    runtime = active_runtime.get()
    manifest = getattr(runtime, "current_manifest", None) if runtime else None
    return manifest if manifest and manifest["node"] == node else None


def require_visible(node, ids, *, kind=None):
    from backend.app.agent.context.assembly import visible_or_reject
    visible_or_reject(ids, current_manifest(node), kind=kind)


def action_context(state, registry, *, emit_event=True):
    # One centralized projection for Task07 proposals; only findings' cited support.
    view = project_context("synthesize", goal=state["goal"], observations=state["observations"], history=state["tool_history"], emit_event=False)
    draft = state["analysis_draft"]
    cited = {eid for f in draft["findings"] for eid in f["supporting_evidence_ids"]}
    policies = {eid for r in draft["recommendations"] for eid in r.get("policy_evidence_ids", [])}
    payload = dict(goal=state["goal"], findings=[{k:f[k] for k in ("title", "interpretation", "supporting_evidence_ids")} for f in draft["findings"]],
        recommendations=[{k:v for k,v in r.items() if k != "policy_citations"} for r in draft["recommendations"]],
        selected_business_evidence=[b for b in view["selected_evidence"]+view["executive_facts"] if b["evidence_id"] in cited],
        selected_policy_evidence=[b for b in view.get("policy_evidence", []) if b["evidence_id"] in policies],
        available_write_tools=[m.model_dump(mode="json") for m in registry.list_tools(permission="WRITE_APPROVAL_REQUIRED")])
    return publish_context("propose_action", payload, emit_event=emit_event)


def prepare_gateway_context(name, args):
    """Apply the same preflight before offline gateway calls are counted."""
    if name == "propose_action":
        runtime = active_runtime.get()
        if runtime and runtime.prepared_manifest is not None:return args[0]
        return publish_context(name, args[0])
    if name == "understand_goal":return build_llm_context(name, user_query=args[0])
    if name == "create_plan":return build_llm_context("plan", goal=args[0], capabilities=args[1])
    if name == "select_tool":return build_llm_context(name, goal=args[0], step=args[1], tools=args[2])
    if name == "select_investigation_tool":
        return build_llm_context("select_tool", goal=args[0], plan=args[1], step=args[2], observations=args[3], tools=args[4])
    if name == "review_progress":
        return build_llm_context(name, goal=args[0], plan=args[1], step=args[2], observations=args[4], history=args[5])
    if name == "synthesize":return build_llm_context(name, goal=args[0], observations=args[1], history=args[2])
