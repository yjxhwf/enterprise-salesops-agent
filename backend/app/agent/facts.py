"""Source-derived measurements and fixed rendering; no business causality or model calls."""
from decimal import Decimal, ROUND_HALF_UP

from backend.app.agent.schemas import CanonicalFact


# Metric vocabulary describes the frozen tool schemas, never dataset entities or answers.
METRICS = {
    "revenue": ("Revenue", "MONEY", "revenue_growth_pct"),
    "profit": ("Profit", "MONEY", "profit_growth_pct"),
    "margin": ("Margin", "PERCENTAGE", "margin_change_pp"),
    "order_count": ("Order Count", "COUNT", "order_count_growth_pct"),
    "customer_count": ("Customer Count", "COUNT", None),
    "quantity": ("Quantity", "COUNT", "quantity_growth_pct"),
    "avg_discount": ("成交价/标价比", "PERCENTAGE", None),
    "revenue_share": ("Revenue Share", "PERCENTAGE", "revenue_share_change_pp"),
    "cost": ("Cost", "MONEY", None),
    "list_price": ("List Price", "MONEY", None),
    "sale_price": ("Sale Price", "MONEY", None),
    "discount_rate": ("成交价/标价比", "PERCENTAGE", None),
    "approval_status": ("Approval Status", "TEXT", None),
    "order_date": ("Order Date", "TEXT", None),
    "customer_id": ("Customer", "TEXT", None),
    "product_id": ("Product", "TEXT", None),
}


def format_measure(value, unit, *, fraction=False):
    if value is None:
        return "不可用"
    if unit == "TEXT":
        return str(value)
    number = Decimal(str(value))
    if fraction:
        number *= 100
    places = 0 if unit == "COUNT" else 4 if unit == "DISCOUNT" else 2
    number = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    suffix = "%" if unit == "PERCENTAGE" else "个百分点" if unit == "PERCENTAGE_POINT" else ""
    return f"{number:,.{places}f}" + suffix


def render_fact(fact):
    label = fact.get("business_label") or METRICS[fact["metric"]][0]
    period = fact["period"]
    dates = f" [{period['start_date']} 至 {period['end_date']}]" if period else ""
    filters = ", ".join(f"{key}={value}" for key, value in sorted(fact["scope"].items()) if key not in {"start_date", "end_date", "limit"} and value is not None)
    scope = f"（范围 {filters}）" if filters else ""
    text = f"{fact['subject_type']} {fact['subject_id']}{dates}{scope}：{label} {fact['display_value']}"
    if fact["change_direction"] != "NOT_AVAILABLE":
        basis = {"PREVIOUS_EQUAL_LENGTH_PERIOD": "前一等长周期", "YEAR_OVER_YEAR": "上年同期"}[fact["comparison_basis"]]
        direction = {"INCREASE": "增长", "DECREASE": "下降", "UNCHANGED": "持平"}[fact["change_direction"]]
        amount = "" if direction == "持平" else format_measure(abs(Decimal(str(fact["change_value"]))), fact["change_unit"])
        text += f"；较{basis}{direction}{amount}（前值 {fact['comparison_display_value']}）"
    if fact["metric"] == "profit" and fact["current_sign"] in {"POSITIVE", "NEGATIVE", "ZERO"}:
        text += {"POSITIVE": "；当前利润为正", "NEGATIVE": "；当前利润为负", "ZERO": "；当前利润为零"}[fact["current_sign"]]
    if fact.get("semantic_metric") == "discount_depth" and fact["change_direction"] in {"INCREASE", "DECREASE"}:
        text += "；" + ("优惠加深" if fact["change_direction"] == "INCREASE" else "优惠收窄")
    return text + "。"


def canonical_facts(catalog, evidence_ids=None):
    facts = []
    for eid in sorted(set(evidence_ids if evidence_ids is not None else catalog)):
        evidence = catalog[eid]
        raw = evidence["internal_reference"]["raw_value"]
        subject_type, subject_id = "COMPANY", "company"
        for field, kind in (("order_id", "ORDER"), ("product_id", "PRODUCT"), ("customer_id", "CUSTOMER"), ("region", "REGION")):
            if field in raw:
                subject_type, subject_id = kind, str(raw[field])
                break
        current = raw.get("current", raw)
        previous = raw.get("previous") or {}
        changes = raw.get("changes") or {}
        for metric, (_, unit, change_key) in METRICS.items():
            if metric not in current:
                continue
            value = current[metric]
            prior = previous.get(metric) if evidence["comparison_basis"] and raw.get("comparison_available", True) else None
            change, change_unit, direction = None, None, "NOT_AVAILABLE"
            sign = "NOT_AVAILABLE" if value is None else "NOT_NUMERIC"
            if value is not None and unit != "TEXT":
                number = Decimal(str(value))
                sign = "POSITIVE" if number > 0 else "NEGATIVE" if number < 0 else "ZERO"
                if prior is not None:
                    delta = number - Decimal(str(prior))
                    direction = "INCREASE" if delta > 0 else "DECREASE" if delta < 0 else "UNCHANGED"
                    if change_key and changes.get(change_key) is not None:
                        change = changes[change_key]
                        change_unit = "PERCENTAGE_POINT" if change_key.endswith("_pp") else "PERCENTAGE"
                    else:
                        change = str(delta * 100 if unit == "PERCENTAGE" else delta)
                        change_unit = "PERCENTAGE_POINT" if unit == "PERCENTAGE" else unit
            fact = dict(fact_id=eid + "#" + metric, evidence_id=eid, subject_type=subject_type,
                        subject_id=subject_id, period=evidence["current_period"], scope=evidence["scope"], metric=metric,
                        current_value=value, display_value=format_measure(value, unit, fraction=unit == "PERCENTAGE"),
                        comparison_basis=evidence["comparison_basis"], comparison_value=prior,
                        comparison_display_value=format_measure(prior, unit, fraction=unit == "PERCENTAGE") if prior is not None else None,
                        change_value=change, change_display_value=format_measure(change, change_unit) if change is not None else None,
                        change_direction=direction, current_sign=sign, unit=unit, change_unit=change_unit,
                        source_tool=evidence["source_tool"])
            if metric in {"avg_discount", "discount_rate"}:
                fact.update(semantic_metric="price_realization_ratio", business_label="成交价/标价比",
                            direction_semantics="lower_ratio_means_deeper_discount")
                fact["fact_id"] = eid + "#price_realization_ratio"
            fact["rendered_text"] = render_fact(fact)
            facts.append(CanonicalFact.model_validate(fact).model_dump(mode="json"))
            if metric in {"avg_discount", "discount_rate"}:
                depth = dict(fact)
                derived = str(Decimal(1) - Decimal(str(value))) if value is not None else None
                depth.update(fact_id=eid + "#discount_depth", semantic_metric="discount_depth",
                             business_label="实际优惠幅度", derived_value=derived,
                             current_value=derived, direction_semantics="higher_depth_means_deeper_discount")
                depth["display_value"] = format_measure(derived, "PERCENTAGE", fraction=True)
                depth["comparison_value"] = str(Decimal(1) - Decimal(str(prior))) if prior is not None else None
                depth["comparison_display_value"] = format_measure(depth["comparison_value"], "PERCENTAGE", fraction=True) if prior is not None else None
                depth["change_value"] = str(-Decimal(str(change))) if change is not None else None
                depth["change_display_value"] = format_measure(depth["change_value"], change_unit) if change is not None else None
                depth["change_direction"] = {"INCREASE": "DECREASE", "DECREASE": "INCREASE"}.get(direction, direction)
                depth["current_sign"] = ("POSITIVE" if Decimal(derived) > 0 else "NEGATIVE" if Decimal(derived) < 0 else "ZERO") if derived is not None else "NOT_AVAILABLE"
                depth["rendered_text"] = render_fact(depth)
                facts.append(CanonicalFact.model_validate(depth).model_dump(mode="json"))
    return facts


def assemble_draft(draft, catalog):
    result = draft.model_dump(mode="json")
    result["executive_facts"] = [f for f in canonical_facts(catalog)
                                  if f["subject_type"] == "COMPANY" and f["metric"] in {"revenue", "profit", "margin"}]
    for finding in result["findings"]:
        finding["supporting_facts"] = canonical_facts(catalog, finding["supporting_evidence_ids"])
    return result
