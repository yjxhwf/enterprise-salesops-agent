from decimal import Decimal

from backend.app.data.database import session_factory
from backend.app.tools.registry import build_default_registry

DATES = {"start_date": "2026-08-01", "end_date": "2026-08-31"}


def test_missing_approvals_and_stable_order(seeded_engine):
    registry = build_default_registry(session_factory(seeded_engine))
    result = registry.invoke("query_orders", {**DATES, "region": "EAST_CHINA", "product_id": "SKU-A12", "approval_status": "MISSING_APPROVAL"})
    assert result.success
    assert result.meta.total_matches == result.meta.returned_count == 10
    assert not result.meta.truncated
    rows = result.data.orders
    assert rows == sorted(rows, key=lambda r: (-r.order_date.toordinal(), r.order_id))
    assert result.evidence_ids == [f"order:{r.order_id}" for r in rows]
    assert all(r.discount_rate < Decimal("0.85") and r.customer_id in {"C102", "C207"} for r in rows)


def test_truncation_filters_and_default_limit(seeded_engine):
    registry = build_default_registry(session_factory(seeded_engine))
    default = registry.invoke("query_orders", DATES)
    assert default.meta.returned_count == 20 and default.meta.total_matches == 667 and default.meta.truncated
    maximum = registry.invoke("query_orders", {**DATES, "limit": 50})
    assert len(maximum.data.orders) == 50 and maximum.meta.total_matches == 667
    assert maximum.data.orders[:20] == default.data.orders
    filtered = registry.invoke("query_orders", {**DATES, "customer_id": "C102", "min_discount": "0.74", "max_discount": "0.82"})
    assert filtered.meta.total_matches == 7
    assert all(Decimal("0.74") <= r.discount_rate <= Decimal("0.82") for r in filtered.data.orders)


def test_exact_discount_boundary(seeded_engine):
    registry = build_default_registry(session_factory(seeded_engine))
    result = registry.invoke("query_orders", {**DATES, "min_discount": "0.93", "max_discount": "0.93"})
    assert result.success and all(row.discount_rate == Decimal("0.93") for row in result.data.orders)


def test_existing_entities_with_no_matching_orders(seeded_engine):
    registry = build_default_registry(session_factory(seeded_engine))
    result = registry.invoke("query_orders", {**DATES, "product_id": "SKU-B07", "approval_status": "MISSING_APPROVAL"})
    assert result.error.code == "NO_DATA"
