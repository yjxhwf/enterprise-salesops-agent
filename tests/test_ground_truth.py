from decimal import Decimal
import json

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from backend.app.data.ground_truth import GroundTruth, load_ground_truth
from backend.app.data.models import Customer, Order
from backend.app.data.validation import DatasetValidationError, validate_dataset


def test_ground_truth_load_and_schema(tmp_path):
    truth = load_ground_truth()
    assert truth.seed == 20260917
    assert truth.expected_problem_customers == ["C102", "C207"]
    assert len(truth.primary_causes) == 3
    payload = truth.model_dump(mode="json")
    payload["unexpected_answer"] = "not allowed"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_ground_truth(path)
    payload.pop("unexpected_answer")
    payload["thresholds"]["revenue_growth_pct"] = ["12", "4"]
    with pytest.raises(ValidationError, match="lower bound"):
        GroundTruth.model_validate(payload)


def test_actual_ground_truth_and_integrity(data_session):
    report = validate_dataset(data_session)
    report.require_valid()
    assert report.passed
    assert report.metrics["counts"]["orders"] == 4005
    assert report.metrics["missing_approvals"] > 0
    assert report.checks["system_tables_empty"]
    # A price-only counterfactual significantly alleviates the loss without
    # incorrectly summing customer evidence as another independent cause.
    m = report.metrics
    assert m["counterfactual_august_profit"] > m["overall"]["august"]["profit"]
    assert m["discount_price_recovery"] > abs(m["overall"]["profit_change"]) / 2
    baseline = [v["revenue"] for month, v in m["monthly"].items() if month != "2026-08"]
    assert max(baseline) / min(baseline) < Decimal("1.15")
    assert len(set(baseline)) > 1


def test_financial_corruption_fails(data_session):
    order = data_session.scalar(select(Order).limit(1))
    order.profit += Decimal("100.00")
    data_session.flush()
    report = validate_dataset(data_session)
    assert not report.checks["financial_calculations"]
    with pytest.raises(DatasetValidationError, match="financial_calculations"):
        report.require_valid()


def test_historical_region_uses_order_snapshot(data_session):
    before = validate_dataset(data_session)
    customer = data_session.get(Customer, "C102")
    customer.region = "NORTH_CHINA"
    data_session.flush()
    after = validate_dataset(data_session)
    assert after.metrics["regions"] == before.metrics["regions"]
    assert after.metrics["east_a12_customers"] == before.metrics["east_a12_customers"]
    assert not after.checks["c102"]  # Fixture requires the customer's current region too.


@pytest.mark.parametrize("cause", ["discount", "mix", "decline"])
def test_removing_each_cause_fails_validation(data_session, cause):
    july = {o.order_id: o for o in data_session.scalars(select(Order).where(Order.order_date.between("2026-07-01", "2026-07-31")))}
    august = list(data_session.scalars(select(Order).where(Order.order_date.between("2026-08-01", "2026-08-31"))))
    for order in august:
        previous = july[order.order_id.replace("202608", "202607")]
        if cause == "discount" and order.product_id == "SKU-A12":
            order.sale_price = previous.sale_price
            order.discount_rate = previous.discount_rate
            order.approval_status = "NOT_REQUIRED"
        elif cause == "mix" and order.product_id == "SKU-B07":
            order.quantity = previous.quantity
            order.cost = previous.cost
        elif cause == "decline" and order.product_id == "SKU-C03":
            order.quantity = previous.quantity
            order.cost = previous.cost
        order.revenue = order.sale_price * order.quantity
        order.profit = order.revenue - order.cost
    data_session.flush()
    report = validate_dataset(data_session)
    expected_check = {"discount": "discount_range", "mix": "b07_mix_shift", "decline": "c03_decline"}[cause]
    assert not report.checks[expected_check]
    assert not report.passed


def test_b07_discount_is_not_a_mix_substitute(data_session):
    order = data_session.scalar(select(Order).where(Order.product_id == "SKU-B07", Order.order_date >= "2026-08-01").limit(1))
    order.discount_rate = Decimal("0.89")
    order.sale_price = order.list_price * order.discount_rate
    order.revenue = order.sale_price * order.quantity
    order.profit = order.revenue - order.cost
    order.approval_status = "APPROVED"
    data_session.flush()
    report = validate_dataset(data_session)
    assert report.checks["financial_calculations"]
    assert not report.checks["b07_legal_prices"]
