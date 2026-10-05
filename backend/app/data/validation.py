"""Read actual rows and compare computed Decimal metrics against expectations."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json

from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session

from backend.app.data.database import Base, make_engine, session_factory
from backend.app.data.ground_truth import GroundTruth, load_ground_truth
from backend.app.data.models import Customer, Order, Product, REGIONS, SalesRep

ZERO = Decimal("0")
CENT = Decimal("0.01")


class DatasetValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    metrics: dict
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def require_valid(self) -> None:
        if not self.passed:
            raise DatasetValidationError("Dataset validation failed: " + ", ".join(k for k, v in self.checks.items() if not v))

    def to_json(self) -> str:
        return json.dumps({"metrics": self.metrics, "checks": self.checks, "passed": self.passed}, default=str, indent=2, sort_keys=True)


def summarize(rows: list[Order]) -> dict:
    revenue = sum((o.revenue for o in rows), ZERO)
    profit = sum((o.profit for o in rows), ZERO)
    return {
        "revenue": revenue, "cost": sum((o.cost for o in rows), ZERO), "profit": profit,
        "margin_pct": profit / revenue * 100 if revenue else ZERO,
        "quantity": sum(o.quantity for o in rows), "orders": len(rows),
        "avg_discount": sum((o.discount_rate for o in rows), ZERO) / len(rows) if rows else ZERO,
    }


def growth(before: Decimal | int, after: Decimal | int) -> Decimal:
    if before == 0:
        raise DatasetValidationError("Cannot compare against a zero baseline")
    return (Decimal(after) / Decimal(before) - 1) * 100


def comparison(july: list[Order], august: list[Order]) -> dict:
    before, after = summarize(july), summarize(august)
    return {"july": before, "august": after, "profit_change": after["profit"] - before["profit"]}


def dataset_hashes(session: Session) -> dict[str, str]:
    """Stable sorted content, not row counts or SQLite's physical file layout."""
    result = {}
    for table in Base.metadata.sorted_tables:
        rows = session.execute(select(table).order_by(*table.primary_key.columns)).all()
        payload = [[str(value) if isinstance(value, (Decimal, date)) else value for value in row] for row in rows]
        result[table.name] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return result


def integrity_checks(session: Session, orders: list[Order], products: dict[str, Product], customers: dict[str, Customer], reps: dict[str, SalesRep]) -> dict[str, bool]:
    foreign_keys = all(o.customer_id in customers and o.product_id in products and o.sales_rep_id in reps for o in orders)
    foreign_keys = foreign_keys and all(c.account_owner in reps for c in customers.values())
    foreign_keys = foreign_keys and not session.execute(text("PRAGMA foreign_key_check")).all()
    financial = True
    approvals = True
    for order in orders:
        product = products.get(order.product_id)
        if product is None or order.list_price <= 0:
            financial = False
            continue
        expected = (
            (order.sale_price * order.quantity).quantize(CENT, rounding=ROUND_HALF_UP),
            (product.standard_cost * order.quantity).quantize(CENT, rounding=ROUND_HALF_UP),
            (order.revenue - order.cost).quantize(CENT, rounding=ROUND_HALF_UP),
        )
        financial &= all(abs(a-b) <= CENT for a, b in zip((order.revenue, order.cost, order.profit), expected))
        rate = (order.sale_price / order.list_price).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        financial &= abs(order.discount_rate-rate) <= Decimal("0.0001")
        financial &= order.list_price == product.list_price
        approvals &= (order.approval_status in {"APPROVED", "MISSING_APPROVAL"}) if order.discount_rate < product.minimum_discount_rate else order.approval_status == "NOT_REQUIRED"
    systems_empty = all(session.execute(select(Base.metadata.tables[name])).first() is None for name in ("crm_tasks", "business_alerts", "agent_runs"))
    return {
        "foreign_keys_enabled": session.execute(text("PRAGMA foreign_keys")).scalar_one() == 1,
        "foreign_keys": bool(foreign_keys), "financial_calculations": bool(financial),
        "approval_consistency": bool(approvals), "system_tables_empty": systems_empty,
    }


def validate_dataset(session: Session, truth: GroundTruth | None = None) -> ValidationReport:
    truth = truth or load_ground_truth()
    orders = list(session.scalars(select(Order).order_by(Order.order_id)))
    products = {p.product_id: p for p in session.scalars(select(Product))}
    customers = {c.customer_id: c for c in session.scalars(select(Customer))}
    reps = {r.sales_rep_id: r for r in session.scalars(select(SalesRep))}
    if not orders:
        raise DatasetValidationError("Dataset has no orders")
    months: dict[str, list[Order]] = defaultdict(list)
    for order in orders:
        months[order.order_date.strftime("%Y-%m")].append(order)
    july, august = months[truth.comparison_period], months[truth.analysis_period]
    totals = comparison(july, august)
    j, a = totals["july"], totals["august"]
    revenue_growth = growth(j["revenue"], a["revenue"])
    profit_growth = growth(j["profit"], a["profit"])
    margin_change = a["margin_pct"] - j["margin_pct"]
    regions = {region: comparison([o for o in july if o.region == region], [o for o in august if o.region == region]) for region in REGIONS}
    product_metrics = {pid: comparison([o for o in july if o.product_id == pid], [o for o in august if o.product_id == pid]) for pid in products}
    discount_cause, mix_cause, decline_cause = truth.primary_causes
    region, a12, b07, c03 = discount_cause.region, discount_cause.product_id, mix_cause.product_id, decline_cause.product_id
    if not {a12, b07, c03}.issubset(products) or region not in regions:
        raise DatasetValidationError("Required ground-truth products/region missing")
    east_products = {pid: comparison([o for o in july if o.region == region and o.product_id == pid], [o for o in august if o.region == region and o.product_id == pid]) for pid in products}
    east_a12_july = [o for o in july if o.region == region and o.product_id == a12]
    east_a12_august = [o for o in august if o.region == region and o.product_id == a12]
    threshold = products[a12].minimum_discount_rate
    abnormal = [o for o in east_a12_august if o.discount_rate < threshold]
    missing = [o for o in abnormal if o.approval_status == "MISSING_APPROVAL"]
    customer_metrics = {}
    for cid in sorted({o.customer_id for o in east_a12_july + east_a12_august}):
        scoped = [o for o in abnormal if o.customer_id == cid]
        customer_metrics[cid] = comparison([o for o in east_a12_july if o.customer_id == cid], [o for o in east_a12_august if o.customer_id == cid])
        customer_metrics[cid]["anomaly"] = summarize(scoped)
        customer_metrics[cid]["missing_approvals"] = sum(o.approval_status == "MISSING_APPROVAL" for o in scoped)
    # One diagnostic only: restore July customer/A12 weighted prices on August's
    # discounted units, holding quantities and costs fixed. Not an additive cause.
    recovery = ZERO
    for order in abnormal:
        before = customer_metrics[order.customer_id]["july"]
        if before["quantity"]:
            reference_price = before["revenue"] / before["quantity"]
            recovery += (reference_price - order.sale_price) * order.quantity
    recovery = recovery.quantize(CENT, rounding=ROUND_HALF_UP)
    b = product_metrics[b07]
    b_j_share = b["july"]["revenue"] / j["revenue"] * 100
    b_a_share = b["august"]["revenue"] / a["revenue"] * 100
    c = product_metrics[c03]
    c_growth = growth(c["july"]["revenue"], c["august"]["revenue"])
    t = truth.thresholds
    ranked_customers = sorted(customer_metrics, key=lambda cid: customer_metrics[cid]["profit_change"])
    checks = {
        "revenue_up": a["revenue"] > j["revenue"],
        "profit_down": a["profit"] < j["profit"],
        "revenue_growth_range": t.revenue_growth_pct[0] <= revenue_growth <= t.revenue_growth_pct[1],
        "profit_growth_range": t.profit_growth_pct[0] <= profit_growth <= t.profit_growth_pct[1],
        "margin_down": margin_change <= -t.minimum_margin_drop_pp,
        "east_china": regions[region]["profit_change"] == min(r["profit_change"] for r in regions.values()) and growth(regions[region]["july"]["profit"], regions[region]["august"]["profit"]) <= -t.minimum_east_profit_drop_pct,
        "sku_a12": east_products[a12]["profit_change"] < 0 and east_products[a12]["profit_change"] == min(p["profit_change"] for p in east_products.values()),
        "a12_policy": products[a12].product_tier == "A" and threshold == t.a12_minimum_discount_rate,
        "discount_range": bool(abnormal) and all(t.discount_range[0] <= o.discount_rate <= t.discount_range[1] for o in abnormal),
        "normal_a12_retained": any(o.discount_rate >= threshold for o in east_a12_august),
        "discount_scope": all(o.order_date.strftime("%Y-%m") == truth.analysis_period and o.region == region and o.product_id == a12 and o.customer_id in truth.expected_problem_customers for o in orders if o.discount_rate < products[o.product_id].minimum_discount_rate),
        "missing_approval": len(missing) > 0,
        "core_customer_rank": set(ranked_customers[:2]) == set(truth.expected_problem_customers),
        "discount_materiality": recovery >= (j["profit"]-a["profit"]) * t.minimum_discount_recovery_fraction,
        "b07_mix_shift": b_a_share - b_j_share >= t.minimum_b07_share_gain_pp and b["august"]["quantity"] > b["july"]["quantity"] and b["august"]["margin_pct"] < a["margin_pct"],
        "b07_legal_prices": all(o.discount_rate >= products[b07].minimum_discount_rate and o.approval_status == "NOT_REQUIRED" for o in august if o.product_id == b07),
        "c03_decline": t.c03_revenue_growth_pct[0] <= c_growth <= t.c03_revenue_growth_pct[1] and c["august"]["quantity"] < c["july"]["quantity"],
        "c03_high_margin": ((products[c03].list_price-products[c03].standard_cost)/products[c03].list_price - (products[b07].list_price-products[b07].standard_cost)/products[b07].list_price) * 100 >= t.minimum_standard_margin_gap_pp,
        "c03_secondary": ZERO < -c["profit_change"] < -east_products[a12]["profit_change"],
        "date_range": min(o.order_date for o in orders) == date(2026, 3, 1) and max(o.order_date for o in orders) == date(2026, 8, 31) and len(months) == 6,
        "dataset_size": len(reps) == 8 and len(customers) == 24 and len(products) == 12 and 3000 <= len(orders) <= 5000,
    }
    for cid in truth.expected_problem_customers:
        evidence = customer_metrics.get(cid)
        checks[cid.lower()] = bool(evidence and customers[cid].region == region and evidence["anomaly"]["orders"] > 0 and evidence["missing_approvals"] > 0 and evidence["profit_change"] < 0)
    checks.update(integrity_checks(session, orders, products, customers, reps))
    return ValidationReport(metrics={
        "counts": {"sales_reps": len(reps), "customers": len(customers), "products": len(products), "orders": len(orders)},
        "date_range": [str(min(o.order_date for o in orders)), str(max(o.order_date for o in orders))],
        "monthly": {month: summarize(rows) for month, rows in sorted(months.items())},
        "overall": totals, "revenue_growth_pct": revenue_growth, "profit_growth_pct": profit_growth,
        "margin_change_pp": margin_change, "regions": regions, "products": product_metrics,
        "east_products": east_products, "east_a12_customers": customer_metrics,
        "missing_approvals": len(missing), "abnormal_a12_orders": len(abnormal),
        "discount_price_recovery": recovery,
        "counterfactual_august_profit": a["profit"] + recovery,
        "b07_revenue_share_pct": {"july": b_j_share, "august": b_a_share},
        "c03_revenue_growth_pct": c_growth,
    }, checks=checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only validation of the configured SalesOps dataset")
    parser.parse_args()
    engine: Engine = make_engine()
    try:
        if "orders" not in inspect(engine).get_table_names():
            raise DatasetValidationError("Database is not seeded; run backend.app.data.seed first")
        with session_factory(engine)() as session:
            report = validate_dataset(session)
            print(report.to_json())
            report.require_valid()
        return 0
    except DatasetValidationError as error:
        print(str(error))
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
