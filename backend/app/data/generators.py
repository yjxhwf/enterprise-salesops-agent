"""Fictional, seeded scenario construction; no DB access or ground-truth imports."""

import calendar
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from random import Random

from backend.app.data.models import Customer, Order, Product, REGIONS, SalesRep

DEFAULT_SEED = 20260917
DATASET_VERSION = "salesops-synthetic-v1"
CENT = Decimal("0.01")
RATE_UNIT = Decimal("0.0001")


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass
class Dataset:
    sales_reps: list[SalesRep]
    customers: list[Customer]
    products: list[Product]
    orders: list[Order]


def make_entities() -> tuple[list[SalesRep], list[Customer], list[Product]]:
    # Names are invented labels, not records sourced from real people/businesses.
    names = ["Mira Velorin", "Tavin Orrel", "Sela Brindle", "Corin Veylan",
             "Neri Talven", "Ruvan Elric", "Liora Fenwick", "Daren Mossel"]
    reps = [SalesRep(sales_rep_id=f"SR{i+1:03}", name=name, region=REGIONS[i // 2],
                     team=f"{REGIONS[i // 2]} Commercial", manager=f"Fictional Lead {chr(65+i//2)}")
            for i, name in enumerate(names)]
    company_names = [
        "Velora Loomworks", "Orinth Supply", "Fenmora Components", "Lunavel Packaging",
        "Brindlehaven Retail", "Talvion Systems", "Nerifold Assembly", "Veylora Logistics",
        "Mosselcrest Devices", "Arvendale Textiles", "Silvorn Trade", "Quenlora Workshop",
        "Belvire Networks", "Dornelia Materials", "Fenril Grove Goods", "Lioraven Equipment",
        "Merovale Instruments", "Tavindel Distribution", "Orrelune Fabrication", "Zelvian Services",
        "Calverin Interiors", "Esmora Industrial", "Virelden Supplies", "Noralith Storage",
    ]
    ids = ["C102", "C207"] + [f"C{i:03}" for i in range(301, 323)]
    customers = [Customer(
        customer_id=ids[i], customer_name=name, customer_tier=("A", "A", "B", "B", "C", "B")[i % 6],
        industry=("Manufacturing", "Retail", "Technology", "Logistics")[i % 4],
        region=REGIONS[i // 6], account_owner=reps[(i // 6) * 2 + i % 2].sales_rep_id,
        status="AT_RISK" if i < 2 else "ACTIVE",
    ) for i, name in enumerate(company_names)]
    specs = [
        ("SKU-A12", "Veyra Control Hub", "Control", "1000", "600", "0.85", "A"),
        ("SKU-B07", "Talven Utility Pack", "Consumables", "400", "360", "0.90", "C"),
        ("SKU-C03", "Orinth Precision Module", "Precision", "800", "340", "0.88", "A"),
        ("SKU-D01", "Mossel Relay Kit", "Control", "260", "165", "0.88", "B"),
        ("SKU-D02", "Fenlora Sensor", "Sensors", "360", "220", "0.88", "B"),
        ("SKU-E04", "Velorin Mount Set", "Accessories", "180", "110", "0.88", "C"),
        ("SKU-E05", "Neriva Cable Pack", "Accessories", "150", "95", "0.88", "C"),
        ("SKU-F06", "Tavora Interface", "Control", "520", "325", "0.88", "B"),
        ("SKU-G08", "Luneth Valve", "Mechanics", "440", "280", "0.88", "B"),
        ("SKU-H09", "Orvela Filter", "Consumables", "210", "135", "0.88", "C"),
        ("SKU-J10", "Virelin Monitor", "Sensors", "620", "390", "0.88", "A"),
        ("SKU-K11", "Merovin Coupler", "Mechanics", "300", "190", "0.88", "B"),
    ]
    products = [Product(product_id=pid, product_name=name, category=category,
                        list_price=Decimal(price), standard_cost=Decimal(cost),
                        minimum_discount_rate=Decimal(threshold), product_tier=tier)
                for pid, name, category, price, cost, threshold, tier in specs]
    return reps, customers, products


def make_order(rng: Random, month: int, index: int, customer: Customer, product: Product,
               quantity: int, rate: Decimal, approval: str = "NOT_REQUIRED") -> Order:
    price = money(product.list_price * rate)
    revenue = money(price * quantity)
    cost = money(product.standard_cost * quantity)
    return Order(
        order_id=f"ORD-2026{month:02}-{index:05}",
        order_date=date(2026, month, rng.randint(1, calendar.monthrange(2026, month)[1])),
        customer_id=customer.customer_id, product_id=product.product_id,
        sales_rep_id=customer.account_owner, region=customer.region, quantity=quantity,
        list_price=product.list_price, sale_price=price, revenue=revenue, cost=cost,
        profit=money(revenue-cost), discount_rate=(price/product.list_price).quantize(RATE_UNIT, rounding=ROUND_HALF_UP),
        approval_status=approval,
    )


def baseline_month(rng: Random, month: int, customers: list[Customer], products: list[Product]) -> list[Order]:
    orders: list[Order] = []
    for ci, customer in enumerate(customers):
        for product in products:
            core = customer.customer_id in {"C102", "C207"} and product.product_id == "SKU-A12"
            count = rng.randint(9, 11) if core else rng.choice([2, 2, 2, 3])
            for _ in range(count):
                if core:
                    quantity = rng.randint(24, 30)
                elif product.product_id == "SKU-B07":
                    quantity = rng.randint(12, 18)
                elif product.product_id == "SKU-C03":
                    quantity = rng.randint(6, 8)
                else:
                    quantity = rng.randint(5, 10) + (ci % 3)
                rate = Decimal(rng.randint(93, 98)) / 100
                orders.append(make_order(rng, month, len(orders)+1, customer, product, quantity, rate))
    return orders


def august_scenario(rng: Random, july: list[Order], customers: list[Customer], products: list[Product]) -> list[Order]:
    """Use July's demand cohorts to isolate three independently inspectable changes."""
    customer_map = {c.customer_id: c for c in customers}
    product_map = {p.product_id: p for p in products}
    orders = []
    for previous in july:
        customer = customer_map[previous.customer_id]
        product = product_map[previous.product_id]
        quantity = previous.quantity
        rate = previous.discount_rate
        approval = "NOT_REQUIRED"
        if product.product_id == "SKU-A12" and customer.customer_id in {"C102", "C207"}:
            # Retain normal transactions alongside the deliberately deep discounts.
            if len(orders) % 5 != 0:
                rate = Decimal(rng.randint(74, 82)) / 100
                approval = "APPROVED" if len(orders) % 3 == 0 else "MISSING_APPROVAL"
        elif product.product_id == "SKU-B07":
            # Increased unit demand at unchanged, legal prices (not a discount anomaly).
            quantity = int((Decimal(quantity) * Decimal("2.4")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        elif product.product_id == "SKU-C03":
            quantity = max(1, int((Decimal(quantity) * Decimal("0.65")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))
        else:
            quantity = max(1, quantity + rng.choice([-1, 0, 0, 0, 1]))
        orders.append(make_order(rng, 8, len(orders)+1, customer, product, quantity, rate, approval))
    return orders


def generate_dataset(seed: int = DEFAULT_SEED) -> Dataset:
    rng = Random(seed)
    reps, customers, products = make_entities()
    orders: list[Order] = []
    july: list[Order] = []
    for month in range(3, 8):
        month_orders = baseline_month(rng, month, customers, products)
        orders.extend(month_orders)
        if month == 7:
            july = month_orders
    orders.extend(august_scenario(rng, july, customers, products))
    return Dataset(reps, customers, products, orders)
