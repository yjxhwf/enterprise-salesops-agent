"""Small deterministic claim checks, not a general natural-language judge."""
from decimal import Decimal
import re

from backend.app.agent.validation_diagnostics import reject


YOY = re.compile(r"同比|\byoy\b|\byear[\s\-–—]*over[\s\-–—]*year\b", re.I)
NEGATIVE_PROFIT = re.compile(r"利润\s*(?:为|是)\s*负|负利润|亏损|\bnegative\s+profit\b|\bloss[ -]making\b|\bprofit\s+is\s+negative\b", re.I)
NEGATED = re.compile(r"(?:并非|不是|没有|未出现|不等于|不代表|是否|\bnot(?:\s+a)?|\bno)\s*$", re.I)
EXCLUSION = re.compile(r"不是因为|并非.{0,24}(?:所致|导致|造成|原因)|不是.{0,24}原因|与.{0,24}无关|(?:已经|可以|可|已)排除|\b(?:ruled? out|unrelated to|not (?:a |the )?cause|not caused by)\b", re.I)
ABSOLUTE_CAUSE = re.compile(r"唯一原因|完全由.{0,30}造成|证明.{0,30}导致|主要原因就是|\b(?:sole cause|entirely caused by|proves? .{0,30}caused)\b", re.I)
SHALLOWER = re.compile(r"(?:优惠|折扣)(?:幅度|力度)?(?:略有|有所|明显|进一步)?(?:收窄|变浅|减小|减少)|\bshallower discounts?\b|\bdiscounts? (?:narrowed|decreased)\b", re.I)
DEEPER = re.compile(r"(?:优惠|折扣)(?:幅度|力度)?(?:略有|有所|明显|进一步)?(?:扩大|加深|增加|变深)|\bdeeper discounts?\b|\bdiscounts? (?:widened|increased)\b", re.I)


def targets_for(clause, selected):
    matched = []
    for item in selected:
        raw = item["internal_reference"]["raw_value"]
        identities = [v for k, v in raw.items() if isinstance(v, str)
                      and (k.endswith("_id") or k.endswith("_name") or k == "region")]
        if any(identity in clause for identity in identities):
            matched.append(item)
    if not matched and re.search(r"公司|整体|全局|\bcompany\b|\boverall\b", clause, re.I):
        matched = [item for item in selected if item["fact_key"] == "overview"]
    return matched or selected


PROFIT_SCOPE_BOUNDARY = re.compile(r"[。；;\n，,.!?！？]|但是|但|不过|然而|实际上|实际|并且|而且|\b(?:but|however|and)\b", re.I)
NEGATION_GOVERNOR = re.compile(
    r"(?:并?不代表|并?不意味着|不能(?:说明|表明)|无法(?:说明|证明)|不足以(?:说明|证明)|不能据此(?:认为|判断))[^。；;\n，,.!?！？]{0,48}$|"
    r"\b(?:does\s+not|doesn't)\s+(?:mean|indicate|imply|prove)\b[^.;,!?\n]{0,100}$|"
    r"\bcannot\s+(?:conclude|infer)\b[^.;,!?\n]{0,100}$", re.I)


def profit_claim_category(prefix):
    """Classify only the bounded scope preceding one negative-profit phrase."""
    prefix = PROFIT_SCOPE_BOUNDARY.split(prefix)[-1].rstrip()
    if NEGATION_GOVERNOR.search(prefix) or NEGATED.search(prefix) or re.search(
        r"(?:尚未|还未|并未|未曾)(?:出现|发生|产生)?\s*$|\b(?:not yet|never)(?: been)?\s*$", prefix, re.I):
        return "NEGATED_NEGATIVE_PROFIT_ASSERTION"
    if re.search(
        r"(?:避免|防止|预防)(?:未来|将来|后续)?(?:业务|公司|产品)?(?:出现|发生|产生|进入)?\s*$|"
        r"(?:未来|将来)?(?:可能|或将|将会)(?:出现|发生|进入)?\s*$|"
        r"(?:若|如果|一旦)[^。；;，,]{0,32}会(?:出现|发生)?\s*$|"
        r"\b(?:avoid|prevent)(?: future)?(?: becoming| having)?\s*$|"
        r"\bif\b[^.;,]{0,45}\b(?:may|might|could)(?: become| have| experience)?\s*$", prefix, re.I):
        return "HYPOTHETICAL_OR_PREVENTIVE_LOSS"
    return "CURRENT_NEGATIVE_PROFIT_ASSERTION"


def validate_text(text, catalog, evidence_ids=None, *, field_path="text"):
    selected = [catalog[eid] for eid in evidence_ids] if evidence_ids else list(catalog.values())
    for pattern, rule in ((EXCLUSION, "EXCLUSION_CLAIM"), (ABSOLUTE_CAUSE, "ABSOLUTE_CAUSALITY")):
        match = pattern.search(text)
        if match:
            reject(rule, field_path, "unsupported_exclusion_or_absolute_causality", text, selected, position=match.start())
    if YOY.search(text) and not any(
        e["comparison_basis"] == "YEAR_OVER_YEAR" for e in catalog.values()
    ):
        reject("COMPARISON_BASIS", field_path, "year_over_year_without_supported_basis", text, selected, position=YOY.search(text).start())
    for clause in re.split(r"[。；;\n，,!?！？]", text):
        targets = targets_for(clause, selected)
        for pattern, wrong_sign in ((SHALLOWER, -1), (DEEPER, 1)):
            direction_claims = [m for m in pattern.finditer(clause) if not NEGATED.search(clause[:m.start()].rstrip())]
            if direction_claims:
                for item in targets:
                    raw = item["internal_reference"]["raw_value"]
                    current, prior = raw.get("current", raw), raw.get("previous") or {}
                    if not item["comparison_basis"] or not raw.get("comparison_available", True):
                        continue
                    for metric in ("avg_discount", "discount_rate"):
                        if current.get(metric) is not None and prior.get(metric) is not None:
                            delta = Decimal(str(current[metric])) - Decimal(str(prior[metric]))
                            if delta * wrong_sign > 0:
                                reject("DISCOUNT_DIRECTION", field_path, "discount_direction_conflicts_with_evidence", clause, [item], position=direction_claims[0].start())
        claims = [m for m in NEGATIVE_PROFIT.finditer(clause)
                  if profit_claim_category(clause[:m.start()]) == "CURRENT_NEGATIVE_PROFIT_ASSERTION"]
        if not claims:
            continue
        # Explicit entity identifiers/names take priority over the entire citation set.
        matched = []
        for item in selected:
            raw = item["internal_reference"]["raw_value"]
            identities = [v for k, v in raw.items() if isinstance(v, str)
                          and (k.endswith("_id") or k.endswith("_name") or k == "region")]
            if any(identity in clause for identity in identities):
                matched.append(item)
        if not matched and re.search(r"公司|整体|全局|\bcompany\b|\boverall\b", clause, re.I):
            matched = [item for item in selected if item["fact_key"] == "overview"]
        targets = matched or selected
        positive_evidence = []
        for item in targets:
            raw = item["internal_reference"]["raw_value"]
            value = raw.get("current", raw).get("profit")
            if value is not None and Decimal(str(value)) > 0:
                positive_evidence.append(item)
        # An ambiguous mixed-scope claim must be made explicit, never guessed.
        if positive_evidence:
            reject("PROFIT_SIGN_CONSISTENCY", field_path, "positive_profit_evidence_conflicts_with_negative_profit_claim",
                   clause, positive_evidence, position=claims[0].start(), profit=True)


def validate_review(review, catalog):
    if review.remaining_evidence_gap:
        validate_text(review.remaining_evidence_gap, catalog, field_path="remaining_evidence_gap")
    if review.next_objective:
        validate_text(review.next_objective, catalog, field_path="next_objective")
    if review.updated_plan:
        for index, step in enumerate(review.updated_plan.steps):
            validate_text(step.objective, catalog, field_path=f"updated_plan.steps[{index}].objective")
            validate_text(step.expected_output, catalog, field_path=f"updated_plan.steps[{index}].expected_output")


def validate_draft(draft, catalog):
    from backend.app.agent.facts import canonical_facts
    for index, finding in enumerate(draft.findings):
        facts = canonical_facts(catalog, finding.supporting_evidence_ids)
        # Derived depth and its source ratio are one independent measurement.
        measurements = {(f["evidence_id"], f["metric"]) for f in facts
                        if f["current_value"] is not None and f["unit"] != "TEXT"}
        if finding.claim_type == "CONTRIBUTING_FACTOR" and len(measurements) < 2:
            reject("CONTRIBUTION_EVIDENCE", f"findings[{index}].supporting_evidence_ids", "fewer_than_two_independent_measurements", "", [catalog[eid] for eid in finding.supporting_evidence_ids])
        validate_text(finding.title, catalog, finding.supporting_evidence_ids, field_path=f"findings[{index}].title")
        validate_text(finding.interpretation, catalog, finding.supporting_evidence_ids, field_path=f"findings[{index}].interpretation")
    validate_text(draft.executive_interpretation, catalog, field_path="executive_interpretation")
    for index, recommendation in enumerate(draft.recommendations):
        ids = draft.findings[recommendation.related_finding_index].supporting_evidence_ids
        validate_text(recommendation.title, catalog, ids, field_path=f"recommendations[{index}].title")
        validate_text(recommendation.action, catalog, ids, field_path=f"recommendations[{index}].action")
    for index, limitation in enumerate(draft.limitations):
        validate_text(limitation, catalog, field_path=f"limitations[{index}]")
