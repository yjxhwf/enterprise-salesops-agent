from decimal import Decimal

import pytest
from sqlalchemy import inspect, select, text, update
from sqlalchemy.exc import IntegrityError

from backend.app.config import Settings
from backend.app.data.database import Base, create_tables, drop_tables, make_engine, session_factory
from backend.app.data.models import Customer, Order, Product


def test_schema_and_indexes(empty_engine):
    create_tables(empty_engine)
    schema = inspect(empty_engine)
    assert set(schema.get_table_names()) == {
        "sales_reps", "customers", "products", "orders", "crm_tasks", "business_alerts", "agent_runs",
    }
    assert {tuple(i["column_names"]) for i in schema.get_indexes("orders")} == {
        ("order_date", "region"), ("customer_id",), ("product_id",), ("region",),
    }
    assert len(schema.get_foreign_keys("orders")) == 3
    assert len(schema.get_check_constraints("orders")) >= 6
    assert {c.name for c in Base.metadata.tables["orders"].columns if c.primary_key} == {"order_id"}
    drop_tables(empty_engine)
    assert inspect(empty_engine).get_table_names() == []


def test_configuration_override_and_no_implicit_tables(tmp_path):
    path = tmp_path / "isolated.db"
    settings = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{path.as_posix()}")
    engine = make_engine(settings.DATABASE_URL)
    try:
        assert not path.exists()  # Engine construction/import has no DB file side effect.
        assert inspect(engine).get_table_names() == []
        with engine.connect() as first, engine.connect() as second:
            assert first.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
            assert second.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    finally:
        engine.dispose()


@pytest.mark.parametrize("column", ["customer_id", "product_id", "sales_rep_id"])
def test_invalid_order_foreign_key_fails(data_session, column):
    order = data_session.scalar(select(Order).limit(1))
    with pytest.raises(IntegrityError):
        data_session.execute(update(Order).where(Order.order_id == order.order_id).values({column: "DOES-NOT-EXIST"}))


def test_invalid_customer_owner_fails(data_session):
    with pytest.raises(IntegrityError):
        data_session.execute(update(Customer).where(Customer.customer_id == "C102").values(account_owner="INVALID"))


def test_invalid_fk_insert_fails(data_session):
    with pytest.raises(IntegrityError):
        data_session.add(Customer(customer_id="BAD", customer_name="Fictional Invalid Customer",
                                  customer_tier="A", industry="Manufacturing", region="EAST_CHINA",
                                  account_owner="INVALID", status="ACTIVE"))
        data_session.flush()


@pytest.mark.parametrize(("field", "value"), [
    ("quantity", 0), ("quantity", -1), ("quantity", 1.5),
    ("list_price", 0), ("sale_price", 0), ("cost", -1), ("revenue", 0),
    ("discount_rate", 0), ("discount_rate", Decimal("1.1")),
    ("region", "east_china"), ("approval_status", "UNKNOWN"),
])
def test_order_constraints(data_session, field, value):
    order = data_session.scalar(select(Order).limit(1))
    with pytest.raises(IntegrityError):
        data_session.execute(update(Order).where(Order.order_id == order.order_id).values({field: value}))


@pytest.mark.parametrize(("model", "values"), [
    (Customer, {"customer_tier": "D"}), (Customer, {"status": "UNKNOWN"}),
    (Customer, {"region": "华东"}), (Product, {"product_tier": "D"}),
    (Product, {"standard_cost": -1}), (Product, {"list_price": 0}),
    (Product, {"minimum_discount_rate": Decimal("1.1")}),
])
def test_entity_constraints(data_session, model, values):
    with pytest.raises(IntegrityError):
        data_session.execute(update(model).values(**values))


def test_numeric_values_return_decimal(data_session):
    order = data_session.scalar(select(Order).limit(1))
    for field in ("list_price", "sale_price", "revenue", "cost", "profit", "discount_rate"):
        assert isinstance(getattr(order, field), Decimal)
