"""Deterministic, in-memory citation adapter over successful read-tool results."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json

from backend.app.agent.grounding import invalid_output
from backend.app.agent.schemas import CanonicalEvidence


def comparison_semantics(record):
    meta = record.get("meta", {})
    current, previous = meta.get("current_period"), meta.get("comparison_period")
    basis, label = None, None
    if current and previous:
        start, end = (date.fromisoformat(current[k]) for k in ("start_date", "end_date"))
        prior_start, prior_end = (date.fromisoformat(previous[k]) for k in ("start_date", "end_date"))
        if prior_end == start - timedelta(days=1) and end - start == prior_end - prior_start:
            basis, label = "PREVIOUS_EQUAL_LENGTH_PERIOD", "前一等长周期"
        else:
            # Do not invent a comparison interpretation for an unsupported adapter.
            raise ValueError("Unsupported comparison periods")
    return dict(comparison_basis=basis, comparison_label=label,
                current_period=current, comparison_period=previous)


def display_value(value, key=""):
    if isinstance(value, dict):
        return {k: display_value(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [display_value(v, key) for v in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if key.endswith("_id") or key in {"id", "region", "start_date", "end_date"}:
        return value
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return value
    if not number.is_finite():
        raise ValueError("Non-finite business metric")
    suffix, places = "", 2
    if key.endswith("_pct"):
        suffix = "%"
    elif key.endswith("_pp"):
        suffix = " pp"
    elif key in {"margin", "revenue_share"}:
        number *= 100
        suffix = "%"
    elif "discount" in key:
        places = 4
    rounded = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{rounded:.{places}f}{suffix}"


def extract_evidence(history):
    """Rebuild only from this run's history; never consult a global citation cache."""
    catalog = {}
    groups = {"regions": "region", "products": "product_id", "customers": "customer_id", "orders": "order_id"}
    for record in history:
        if record["tool_name"] in {"search_sales_policy", "create_crm_task", "create_business_alert"}:
            continue  # Policy and action results are never diagnostic business measurements.
        if not record["success"] or record.get("data") is None:
            continue
        data = record["data"]
        group = next((key for key in groups if isinstance(data.get(key), list)), None)
        rows = enumerate(data[group]) if group else [(None, data)]
        for position, raw in rows:
            identity = str(raw[groups[group]]) if group else "overview"
            if group == "orders":
                parent = "order:" + identity
                if parent not in record["evidence_ids"]:
                    raise ValueError("Order source evidence missing")
                eid = parent
            else:
                parent = record["evidence_ids"][0]
                eid = parent + ":" + identity if group else parent
            pointer = f"/data/{group}/{position}" if group else "/data"
            scope = record.get("arguments", data.get("period", {}))
            item = CanonicalEvidence(
                evidence_id=eid, source_tool=record["tool_name"], fact_key=identity,
                display_value=display_value(raw), scope=scope,
                summary=f"{record['tool_name']}: {identity}; scope={json.dumps(scope, sort_keys=True)}",
                internal_reference={"source_evidence_id": parent, "json_pointer": pointer,
                                    "raw_value": deepcopy(raw)},
                **comparison_semantics(record),
            ).model_dump(mode="json")
            if eid in catalog and catalog[eid]["internal_reference"]["raw_value"] != raw:
                raise ValueError("Conflicting source evidence")
            catalog[eid] = item
    return catalog


def visible_evidence(history):
    from backend.app.agent.facts import canonical_facts
    catalog = extract_evidence(history)
    return [{**{key: record[key] for key in ("evidence_id", "source_tool", "summary",
                                            "comparison_basis", "comparison_label", "current_period", "comparison_period")},
             "display_value": semantic_display(record["internal_reference"]["raw_value"]),
             "rendered_facts": [f["rendered_text"] for f in canonical_facts(catalog, [eid])]}
            for eid, record in catalog.items()]


def semantic_display(raw):
    """Project frozen ratios without changing source data or rounding before derivation."""
    if isinstance(raw, list):
        return [semantic_display(item) for item in raw]
    if not isinstance(raw, dict):
        return raw
    result = {}
    for key, value in raw.items():
        if key in {"avg_discount", "discount_rate"}:
            from backend.app.agent.facts import format_measure
            ratio = Decimal(str(value)) if value is not None else None
            result["price_realization_ratio（成交价/标价比）"] = format_measure(ratio, "PERCENTAGE", fraction=True)
            result["discount_depth（实际优惠幅度）"] = format_measure(1 - ratio if ratio is not None else None, "PERCENTAGE", fraction=True)
        else:
            result[key] = semantic_display(value) if isinstance(value, (dict, list)) else display_value(value, key)
    return result


def visible_observations(observations):
    result = deepcopy(observations)
    for observation in result:
        if "facts" in observation:
            observation["facts"] = [{key: fact[key] for key in ("fact_id", "evidence_id", "rendered_text")}
                                    for fact in observation["facts"]]
    return result


def validate_citations(evidence_ids, history):
    catalog = extract_evidence(history)
    if not evidence_ids or any(eid not in catalog for eid in evidence_ids):
        invalid_output()
    return sorted(set(evidence_ids))
