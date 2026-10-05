"""Task 01 schema. System tables are intentionally empty until later tasks."""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.data.database import Base

REGIONS = ("EAST_CHINA", "NORTH_CHINA", "SOUTH_CHINA", "WEST_CHINA")
REGION_CHECK = "region IN ('EAST_CHINA','NORTH_CHINA','SOUTH_CHINA','WEST_CHINA')"
MONEY = Numeric(12, 2)
RATE = Numeric(6, 4)


class SalesRep(Base):
    __tablename__ = "sales_reps"
    __table_args__ = (CheckConstraint(REGION_CHECK, name="ck_rep_region"),)
    sales_rep_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    region: Mapped[str] = mapped_column(String(20))
    team: Mapped[str] = mapped_column(String(80))
    manager: Mapped[str] = mapped_column(String(80))


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint(REGION_CHECK, name="ck_customer_region"),
        CheckConstraint("customer_tier IN ('A','B','C')", name="ck_customer_tier"),
        CheckConstraint("status IN ('ACTIVE','AT_RISK')", name="ck_customer_status"),
    )
    customer_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(100))
    customer_tier: Mapped[str] = mapped_column(String(1))
    industry: Mapped[str] = mapped_column(String(60))
    region: Mapped[str] = mapped_column(String(20))
    account_owner: Mapped[str] = mapped_column(ForeignKey("sales_reps.sales_rep_id"))
    status: Mapped[str] = mapped_column(String(20))


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("list_price > 0", name="ck_product_price"),
        CheckConstraint("standard_cost >= 0", name="ck_product_cost"),
        CheckConstraint("minimum_discount_rate > 0 AND minimum_discount_rate <= 1", name="ck_product_discount"),
        CheckConstraint("product_tier IN ('A','B','C')", name="ck_product_tier"),
    )
    product_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    product_name: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(60))
    list_price: Mapped[Decimal] = mapped_column(MONEY)
    standard_cost: Mapped[Decimal] = mapped_column(MONEY)
    minimum_discount_rate: Mapped[Decimal] = mapped_column(RATE)
    product_tier: Mapped[str] = mapped_column(String(1))


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint(REGION_CHECK, name="ck_order_region"),
        CheckConstraint("quantity > 0 AND quantity = CAST(quantity AS INTEGER)", name="ck_order_quantity"),
        CheckConstraint("list_price > 0 AND sale_price > 0", name="ck_order_prices"),
        CheckConstraint("revenue > 0 AND cost >= 0", name="ck_order_amounts"),
        CheckConstraint("discount_rate > 0 AND discount_rate <= 1", name="ck_order_discount"),
        CheckConstraint("approval_status IN ('NOT_REQUIRED','APPROVED','MISSING_APPROVAL')", name="ck_order_approval"),
        Index("ix_orders_date_region", "order_date", "region"),
        Index("ix_orders_customer", "customer_id"),
        Index("ix_orders_product", "product_id"),
        Index("ix_orders_region", "region"),
    )
    order_id: Mapped[str] = mapped_column(String(30), primary_key=True)
    order_date: Mapped[date] = mapped_column(Date)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    product_id: Mapped[str] = mapped_column(ForeignKey("products.product_id"))
    sales_rep_id: Mapped[str] = mapped_column(ForeignKey("sales_reps.sales_rep_id"))
    # Historical sales-region snapshot, not a dynamic join to current ownership.
    region: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[int] = mapped_column(Integer)
    list_price: Mapped[Decimal] = mapped_column(MONEY)
    sale_price: Mapped[Decimal] = mapped_column(MONEY)
    revenue: Mapped[Decimal] = mapped_column(MONEY)
    cost: Mapped[Decimal] = mapped_column(MONEY)
    profit: Mapped[Decimal] = mapped_column(MONEY)
    discount_rate: Mapped[Decimal] = mapped_column(RATE)
    approval_status: Mapped[str] = mapped_column(String(24))


class CRMTask(Base):
    __tablename__ = "crm_tasks"
    task_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    reason: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(String(20))
    suggested_action: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class BusinessAlert(Base):
    __tablename__ = "business_alerts"
    alert_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    target_type: Mapped[str] = mapped_column(String(30))
    target_id: Mapped[str] = mapped_column(String(40))
    severity: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (CheckConstraint(
        "tool_calls >= 0 AND llm_calls >= 0 AND input_tokens >= 0 AND output_tokens >= 0",
        name="ck_run_counters",
    ),)
    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_query: Mapped[str] = mapped_column(Text)
    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str | None] = mapped_column(Text)
