"""Parameterized SELECTs and grouped aggregates; no agent/tool policy or fixtures."""

from datetime import date
from decimal import Decimal

from sqlalchemy import BigInteger, cast, func, select
from sqlalchemy.orm import Session

from backend.app.data.models import Customer, Order, Product, REGIONS

D = Decimal


class EntityNotFound(ValueError):
    pass


class NoData(ValueError):
    pass


def zero_metrics() -> dict:
    return dict(revenue=D("0.00"), profit=D("0.00"), margin=None, order_count=0,
                customer_count=0, quantity=0, avg_discount=None, revenue_share=None)


def growth(current: Decimal | int, previous: Decimal | int) -> Decimal | None:
    return (D(current) - D(previous)) / D(previous) * 100 if previous != 0 else None


def compare(current: dict, previous: dict) -> dict:
    available = previous["order_count"] > 0
    def pp(field):
        return (current[field] - previous[field]) * 100 if current[field] is not None and previous[field] is not None else None
    return {
        "current": current, "previous": previous, "comparison_available": available,
        "changes": {
            "revenue_growth_pct": growth(current["revenue"], previous["revenue"]) if available else None,
            "profit_growth_pct": growth(current["profit"], previous["profit"]) if available else None,
            "quantity_growth_pct": growth(current["quantity"], previous["quantity"]) if available else None,
            "order_count_growth_pct": growth(current["order_count"], previous["order_count"]) if available else None,
            "profit_change_amount": current["profit"] - previous["profit"] if available else None,
            "margin_change_pp": pp("margin"), "revenue_share_change_pp": pp("revenue_share"),
        },
    }


def ranked(rows: list[dict], key: str) -> list[dict]:
    """Worst absolute profit change first, unknown comparison last, then stable ID."""
    return sorted(rows, key=lambda row: (row["changes"]["profit_change_amount"] is None,
                                        row["changes"]["profit_change_amount"] or D(0), row[key]))


class AnalyticsRepository:
    def __init__(self, session: Session):
        self.session = session

    def require_entities(self, product_id: str | None = None, customer_id: str | None = None, region: str | None = None) -> None:
        if product_id is not None and self.session.scalar(select(Product.product_id).where(Product.product_id == product_id)) is None:
            raise EntityNotFound("Product does not exist")
        if customer_id is not None:
            customer_region = self.session.scalar(select(Customer.region).where(Customer.customer_id == customer_id))
            if customer_region is None:
                raise EntityNotFound("Customer does not exist")
            if region is not None and customer_region != region:
                raise NoData("Customer has no data in the requested region")

    @staticmethod
    def filters(start: date, end: date, region: str | None = None,
                product_id: str | None = None, customer_id: str | None = None) -> list:
        predicates = [Order.order_date.between(start, end)]
        for column, value in ((Order.region, region), (Order.product_id, product_id), (Order.customer_id, customer_id)):
            if value is not None:
                predicates.append(column == value)
        return predicates

    def aggregate(self, start: date, end: date, *, group: str | None = None,
                  region: str | None = None, product_id: str | None = None,
                  customer_id: str | None = None) -> dict[str, dict]:
        # Recover each persisted two-decimal amount to integer cents BEFORE SUM.
        # SQLite SUM(Numeric) itself uses binary arithmetic; integer SUM avoids
        # accumulated float error without materializing raw order rows in Python.
        sums = [
            func.sum(cast(func.round(Order.revenue * 100), BigInteger)).label("revenue_cents"),
            func.sum(cast(func.round(Order.profit * 100), BigInteger)).label("profit_cents"),
            func.count(Order.order_id).label("order_count"),
            func.count(func.distinct(Order.customer_id)).label("customer_count"),
            func.sum(Order.quantity).label("quantity"),
            func.sum(cast(func.round(Order.discount_rate * 10000), BigInteger)).label("discount_units"),
        ]
        dimensions = {
            None: [], "region": [Order.region],
            "product": [Order.product_id, Product.product_name],
            "customer": [Order.customer_id, Customer.customer_name, Customer.customer_tier, Customer.region],
        }
        columns = dimensions[group]  # Only called internally with a fixed dimension.
        statement = select(*columns, *sums).select_from(Order)
        if group == "product":
            statement = statement.join(Product, Product.product_id == Order.product_id)
        elif group == "customer":
            statement = statement.join(Customer, Customer.customer_id == Order.customer_id)
        statement = statement.where(*self.filters(start, end, region, product_id, customer_id))
        if columns:
            statement = statement.group_by(*columns).order_by(*columns)
        result = {}
        for row in self.session.execute(statement).mappings():
            count = row["order_count"]
            if not count:
                continue
            revenue = (D(row["revenue_cents"]) / 100).quantize(D("0.01"))
            profit = (D(row["profit_cents"]) / 100).quantize(D("0.01"))
            metrics = dict(revenue=revenue, profit=profit, margin=profit/revenue if revenue else None,
                           order_count=count, customer_count=row["customer_count"], quantity=row["quantity"],
                           avg_discount=D(row["discount_units"])/10000/count, revenue_share=None)
            labels = {column.key: row[column.key] for column in columns}
            key = str(row[columns[0].key]) if columns else "ALL"
            result[key] = {"labels": labels, "metrics": metrics}
        return result

    def overview(self, start: date, end: date, previous_start: date, previous_end: date, region: str | None = None) -> dict:
        current = self.aggregate(start, end, region=region)
        if not current:
            raise NoData("No orders match the current period and scope")
        previous = self.aggregate(previous_start, previous_end, region=region)
        return compare(current["ALL"]["metrics"], previous.get("ALL", {}).get("metrics", zero_metrics()))

    def breakdown(self, group: str, start: date, end: date, previous_start: date, previous_end: date,
                  *, region: str | None = None, product_id: str | None = None,
                  customer_id: str | None = None) -> tuple[list[dict], bool]:
        self.require_entities(product_id, customer_id, region)
        current = self.aggregate(start, end, group=group, region=region, product_id=product_id, customer_id=customer_id)
        if not current:
            raise NoData("No orders match the current period and scope")
        previous = self.aggregate(previous_start, previous_end, group=group, region=region, product_id=product_id, customer_id=customer_id)
        if group == "product":
            if product_id is not None:
                # SKU drill-down must retain ALL products in the share denominator.
                whole_current = self.aggregate(start, end, region=region)
                whole_previous = self.aggregate(previous_start, previous_end, region=region)
                current_total = whole_current["ALL"]["metrics"]["revenue"]
                previous_total = whole_previous.get("ALL", {}).get("metrics", zero_metrics())["revenue"]
            else:
                current_total = sum((r["metrics"]["revenue"] for r in current.values()), D(0))
                previous_total = sum((r["metrics"]["revenue"] for r in previous.values()), D(0))
        keys = set(REGIONS) if group == "region" else current.keys() | previous.keys()
        result = []
        for key in keys:
            now = current.get(key, {}).get("metrics", zero_metrics())
            before = previous.get(key, {}).get("metrics", zero_metrics())
            if group == "product":
                now["revenue_share"] = now["revenue"] / current_total if current_total else None
                before["revenue_share"] = before["revenue"] / previous_total if previous_total else None
            labels = current.get(key, previous.get(key, {"labels": {"region": key}}))["labels"]
            result.append({**labels, **compare(now, before)})
        id_field = {"region": "region", "product": "product_id", "customer": "customer_id"}[group]
        return ranked(result, id_field), bool(previous)

    def orders(self, start: date, end: date, *, region: str | None = None,
               product_id: str | None = None, customer_id: str | None = None,
               min_discount: Decimal | None = None, max_discount: Decimal | None = None,
               approval_status: str | None = None, limit: int = 20) -> tuple[list[dict], int]:
        if not 1 <= limit <= 50:
            raise ValueError("Order result limit must be between 1 and 50")
        self.require_entities(product_id, customer_id, region)
        predicates = self.filters(start, end, region, product_id, customer_id)
        if min_discount is not None:
            predicates.append(Order.discount_rate >= min_discount)
        if max_discount is not None:
            predicates.append(Order.discount_rate <= max_discount)
        if approval_status is not None:
            predicates.append(Order.approval_status == approval_status)
        total = self.session.scalar(select(func.count()).select_from(Order).where(*predicates))
        if not total:
            raise NoData("No orders match the requested filters")
        statement = select(*Order.__table__.columns).where(*predicates).order_by(Order.order_date.desc(), Order.order_id.asc()).limit(limit)
        return [dict(row) for row in self.session.execute(statement).mappings()], total
