from datetime import date
from decimal import Decimal
import json

import pytest
from pydantic import ValidationError

from backend.app.data.database import session_factory
from backend.app.tools.registry import build_default_registry
from backend.app.tools.schemas import OrdersInput, SalesInput, ToolResult

DATES = {"start_date": "2026-08-01", "end_date": "2026-08-31"}
NAMES = ["get_sales_overview", "analyze_region_performance", "analyze_product_performance", "analyze_customer_performance", "query_orders"]


@pytest.mark.parametrize("name", NAMES)
def test_success_contract_json_and_determinism(seeded_engine, name):
    registry = build_default_registry(session_factory(seeded_engine))
    first = registry.invoke(name, DATES)
    second = registry.invoke(name, dict(reversed(list(DATES.items()))))
    assert first.success and first.error is None
    assert first.meta.latency_ms > 0
    assert first.evidence_ids and first.summary
    encoded = first.model_dump(mode="json")
    assert ToolResult.model_validate_json(json.dumps(encoded)) == first
    assert first.model_dump(exclude={"meta": {"latency_ms"}}) == second.model_dump(exclude={"meta": {"latency_ms"}})
    if name == "get_sales_overview":
        assert isinstance(first.data.current.revenue, Decimal)
        assert encoded["data"]["current"]["revenue"] == "3150655.70"


def test_invalid_envelope_is_rejected():
    with pytest.raises(ValidationError):
        ToolResult(success=True, tool_name="example", data=None, summary=None, evidence_ids=[], error=None, meta={"latency_ms": 0})


def test_equal_length_periods():
    monthly = SalesInput(**DATES).comparison_period()
    assert monthly.start_date == date(2026, 7, 1)
    assert monthly.end_date == date(2026, 7, 31)
    weekly = SalesInput(start_date="2026-08-15", end_date="2026-08-21").comparison_period()
    assert weekly.start_date == date(2026, 8, 8)
    assert weekly.end_date == date(2026, 8, 14)
    single = SalesInput(start_date="2026-03-01", end_date="2026-03-01").comparison_period()
    assert single.start_date == single.end_date == date(2026, 2, 28)


@pytest.mark.parametrize("changes", [
    {"start_date": "2026-09-01"}, {"start_date": "invalid"}, {"start_date": 0},
    {"start_date": "0001-01-01"}, {"start_date": "2026-08-01T00:00:00"},
    {"region": "华东"}, {"region": "east_china"}, {"limit": 0}, {"limit": 51},
    {"limit": 10000}, {"limit": True}, {"limit": 1.5},
    {"min_discount": "0.9", "max_discount": "0.8"}, {"min_discount": 0},
    {"max_discount": "1.01"}, {"max_discount": "NaN"},
    {"approval_status": "UNKNOWN"}, {"sql": "DROP TABLE orders"},
    {"database_url": "sqlite:///private.db"}, {"session_factory": "injected"},
])
def test_invalid_arguments_do_not_access_database(changes):
    def forbidden_factory():
        raise AssertionError("Invalid arguments must be rejected before DB access")
    result = build_default_registry(forbidden_factory).invoke("query_orders", {**DATES, **changes})
    assert result.error.code == "INVALID_ARGUMENT"
    assert not result.success and not result.error.retryable
    assert result.data is None and result.summary is None and result.evidence_ids == []


def test_order_input_schema_limits():
    schema = OrdersInput.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["limit"]["maximum"] == 50
    assert OrdersInput(**DATES).limit == 20
