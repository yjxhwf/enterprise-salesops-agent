from datetime import date
from decimal import Decimal

import pytest

from backend.app.data.database import create_tables, session_factory
from backend.app.data.models import Customer, Order, Product, SalesRep
from backend.app.data.validation import validate_dataset
from backend.app.tools.registry import build_default_registry

D = Decimal
DATES = {"start_date": "2026-08-01", "end_date": "2026-08-31"}


@pytest.fixture
def registry(seeded_engine):
    return build_default_registry(session_factory(seeded_engine), raise_unexpected_errors=True)


def test_overview_golden_exact(registry, data_session):
    result = registry.invoke("get_sales_overview", DATES)
    assert result.success
    data = result.data
    truth = validate_dataset(data_session).metrics
    assert data.current.revenue == D("3150655.70")
    assert data.current.profit == D("870250.70")
    assert data.current.order_count == 667 and data.current.customer_count == 24
    assert data.previous.revenue == D("2875670.90")
    assert data.previous.profit == D("977925.90")
    assert abs(data.current.margin*100 - truth["overall"]["august"]["margin_pct"]) < D("1e-20")
    assert abs(data.changes.revenue_growth_pct - truth["revenue_growth_pct"]) < D("1e-20")
    assert abs(data.changes.profit_growth_pct - truth["profit_growth_pct"]) < D("1e-20")
    assert abs(data.changes.margin_change_pp - truth["margin_change_pp"]) < D("1e-20")
    assert data.comparison_available


def test_region_and_product_golden(registry, data_session):
    truth = validate_dataset(data_session).metrics
    regions = registry.invoke("analyze_region_performance", DATES).data.regions
    assert len(regions) == 4 and regions[0].region == "EAST_CHINA"
    assert regions[0].previous.profit == D("370379.40")
    assert regions[0].current.profit == D("284474.50")
    assert regions[0].changes.profit_change_amount == D("-85904.90")
    product = registry.invoke("analyze_product_performance", {**DATES, "region": "EAST_CHINA", "product_id": "SKU-A12"}).data.products[0]
    expected = truth["east_products"]["SKU-A12"]["august"]
    assert product.current.revenue == expected["revenue"]
    assert product.current.profit == expected["profit"]
    assert product.current.avg_discount == expected["avg_discount"] == D("0.8550")
    assert product.current.avg_discount < product.previous.avg_discount
    assert product.current.revenue_share == product.current.revenue / regions[0].current.revenue
    assert product.previous.revenue_share == product.previous.revenue / regions[0].previous.revenue
    all_products = registry.invoke("analyze_product_performance", DATES).data.products
    items = {p.product_id: p for p in all_products}
    assert len(items) == 12
    assert abs(sum(p.current.revenue_share for p in items.values()) - 1) < D("1e-20")
    assert items["SKU-B07"].current.revenue_share > items["SKU-B07"].previous.revenue_share
    assert items["SKU-C03"].current.revenue == D("186752.00")
    assert items["SKU-C03"].previous.revenue == D("281688.00")
    filtered = registry.invoke("analyze_product_performance", {**DATES, "product_id": "SKU-B07"}).data.products[0]
    assert filtered == items["SKU-B07"]


def test_customers_golden_and_consistent_discount(registry, data_session):
    all_rows = registry.invoke("analyze_customer_performance", DATES).data.customers
    assert len(all_rows) == 24
    for cid in ("C102", "C207"):
        result = registry.invoke("analyze_customer_performance", {**DATES, "customer_id": cid})
        row = result.data.customers[0]
        assert row == next(r for r in all_rows if r.customer_id == cid)
        from sqlalchemy import select
        orders = list(data_session.scalars(select(Order).where(Order.customer_id == cid, Order.order_date.between(date(2026, 8, 1), date(2026, 8, 31)))))
        assert row.current.revenue == sum(o.revenue for o in orders)
        assert row.current.avg_discount == sum(o.discount_rate for o in orders) / len(orders)
    mismatch = registry.invoke("analyze_customer_performance", {**DATES, "customer_id": "C102", "region": "WEST_CHINA"})
    assert mismatch.error.code == "NO_DATA"


@pytest.mark.parametrize(("tool", "filter_args"), [
    ("analyze_product_performance", {"product_id": "SKU-NOT-EXIST"}),
    ("analyze_customer_performance", {"customer_id": "C999999"}),
    ("query_orders", {"customer_id": "C999999"}),
    ("query_orders", {"product_id": "SKU-NOT-EXIST"}),
])
def test_entity_not_found(registry, tool, filter_args):
    result = registry.invoke(tool, {**DATES, **filter_args})
    assert result.error.code == "ENTITY_NOT_FOUND" and not result.error.retryable


@pytest.mark.parametrize("tool", ["get_sales_overview", "analyze_region_performance", "analyze_product_performance", "analyze_customer_performance", "query_orders"])
def test_no_data(registry, tool):
    result = registry.invoke(tool, {"start_date": "2030-01-01", "end_date": "2030-01-02"})
    assert result.error.code == "NO_DATA" and not result.error.retryable


@pytest.mark.parametrize("tool", ["get_sales_overview", "analyze_region_performance", "analyze_product_performance", "analyze_customer_performance"])
def test_no_comparison_data(registry, tool):
    result = registry.invoke(tool, {"start_date": "2026-03-01", "end_date": "2026-03-10"})
    assert result.success and result.meta.comparison_available is False
    rows = [result.data] if tool == "get_sales_overview" else next(iter(result.data.model_dump().values()))
    for row in rows:
        change = row["changes"] if isinstance(row, dict) else row.changes.model_dump()
        assert all(value is None for value in change.values())


def test_decimal_cents_inclusive_dates_zero_profit_and_discount_convention(empty_engine):
    create_tables(empty_engine)
    factory = session_factory(empty_engine)
    with factory.begin() as session:
        session.add(SalesRep(sales_rep_id="R", name="Fictional Rep", region="EAST_CHINA", team="Test", manager="Fictional Lead"))
        session.flush()
        session.add(Customer(customer_id="C", customer_name="Fictional Lab", customer_tier="A", industry="Test", region="EAST_CHINA", account_owner="R", status="ACTIVE"))
        session.add_all([Product(product_id=pid, product_name=pid, category="Test", list_price=D("0.30"), standard_cost=cost, minimum_discount_rate=D("0.3"), product_tier="A") for pid, cost in [("P1", D("0.10")), ("P2", D("0.20"))]])
        session.flush()
        for oid, day, pid, qty, price, cost, discount in [
            ("T0", date(2026, 7, 31), "P1", 1, "0.10", "0.10", "0.3333"),
            ("T1", date(2026, 8, 1), "P1", 1, "0.10", "0.10", "0.3333"),
            ("T2", date(2026, 8, 1), "P1", 9, "0.20", "0.90", "0.6667"),
            ("T3", date(2026, 8, 1), "P2", 1, "0.30", "0.20", "1.0000"),
        ]:
            revenue = D(price) * qty
            session.add(Order(order_id=oid, order_date=day, customer_id="C", product_id=pid, sales_rep_id="R", region="EAST_CHINA", quantity=qty, list_price=D("0.30"), sale_price=D(price), revenue=revenue, cost=D(cost), profit=revenue-D(cost), discount_rate=D(discount), approval_status="NOT_REQUIRED"))
    registry = build_default_registry(factory)
    args = {"start_date": "2026-08-01", "end_date": "2026-08-01"}
    result = registry.invoke("get_sales_overview", args)
    assert result.data.current.revenue == D("2.20")
    assert result.data.current.profit == D("1.00")
    assert result.data.current.order_count == 3
    assert result.data.previous.order_count == 1 and result.data.previous.profit == 0
    assert result.data.changes.profit_growth_pct is None and result.data.comparison_available
    product = registry.invoke("analyze_product_performance", {**args, "product_id": "P1"}).data.products[0]
    assert product.current.avg_discount == D("0.5")  # Arithmetic order mean, not quantity weighted.
    assert product.current.revenue_share == D("1.90") / D("2.20")
    region_rows = registry.invoke("analyze_region_performance", args).data.regions
    empty = next(r for r in region_rows if r.region == "NORTH_CHINA")
    assert empty.current.margin is None and empty.previous.margin is None
    missing_baseline = registry.invoke("analyze_product_performance", {**args, "product_id": "P2"}).data.products[0]
    assert not missing_baseline.comparison_available and missing_baseline.changes.revenue_growth_pct is None
