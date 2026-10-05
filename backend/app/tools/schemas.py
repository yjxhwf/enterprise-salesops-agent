"""Public, JSON-serializable read-tool contracts. Ratios are not percentages."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from backend.app.rag.schemas import PolicySearchResult
from backend.app.tools.write_schemas import WriteResultData

Region = Literal["EAST_CHINA", "NORTH_CHINA", "SOUTH_CHINA", "WEST_CHINA"]
Approval = Literal["NOT_REQUIRED", "APPROVED", "MISSING_APPROVAL"]
ErrorCode = Literal["INVALID_ARGUMENT", "ENTITY_NOT_FOUND", "NO_DATA", "DATABASE_ERROR", "TOOL_NOT_FOUND", "INTERNAL_ERROR",
                    "KNOWLEDGE_INDEX_NOT_READY", "INDEX_MODEL_MISMATCH", "INDEX_REBUILD_REQUIRED",
                    "EMBEDDING_PROVIDER_ERROR", "NO_RETRIEVAL_RESULTS", "RETRIEVAL_ERROR",
                    "APPROVAL_REQUIRED", "WRITE_EXECUTION_FAILED"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Period(Contract):
    start_date: date
    end_date: date


class DateInput(Period):
    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def date_only(cls, value):
        if isinstance(value, datetime) or not isinstance(value, (str, date)):
            raise ValueError("Use a date or an ISO YYYY-MM-DD string")
        if isinstance(value, str):
            if len(value) != 10 or value[4] != "-" or value[7] != "-":
                raise ValueError("Use ISO YYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def valid_range(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must not exceed end_date")
        days = (self.end_date - self.start_date).days + 1
        if self.start_date.toordinal() <= days:
            raise ValueError("Previous equal-length period is outside the supported date range")
        return self

    def current_period(self) -> Period:
        return Period(start_date=self.start_date, end_date=self.end_date)

    def comparison_period(self) -> Period:
        days = (self.end_date - self.start_date).days + 1
        return Period(start_date=self.start_date - timedelta(days=days), end_date=self.start_date - timedelta(days=1))


class SalesInput(DateInput):
    region: Region | None = None


class RegionInput(DateInput):
    pass


class ProductInput(SalesInput):
    product_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class CustomerInput(SalesInput):
    customer_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class OrdersInput(ProductInput):
    customer_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    min_discount: Decimal | None = Field(default=None, gt=0, le=1)
    max_discount: Decimal | None = Field(default=None, gt=0, le=1)
    approval_status: Approval | None = None
    limit: int = Field(default=20, ge=1, le=50, strict=True)

    @model_validator(mode="after")
    def discount_bounds(self):
        if self.min_discount is not None and self.max_discount is not None and self.min_discount > self.max_discount:
            raise ValueError("min_discount must not exceed max_discount")
        return self


class Metrics(Contract):
    revenue: Decimal
    profit: Decimal
    margin: Decimal | None
    order_count: int
    customer_count: int
    quantity: int
    avg_discount: Decimal | None
    revenue_share: Decimal | None = None


class Changes(Contract):
    revenue_growth_pct: Decimal | None
    profit_growth_pct: Decimal | None
    quantity_growth_pct: Decimal | None
    order_count_growth_pct: Decimal | None
    profit_change_amount: Decimal | None
    margin_change_pp: Decimal | None
    revenue_share_change_pp: Decimal | None


class Comparison(Contract):
    current: Metrics
    previous: Metrics
    changes: Changes
    comparison_available: bool


class OverviewData(Comparison):
    pass


class RegionRow(Comparison):
    region: Region


class ProductRow(Comparison):
    product_id: str
    product_name: str


class CustomerRow(Comparison):
    customer_id: str
    customer_name: str
    customer_tier: Literal["A", "B", "C"]
    region: Region


class RegionData(Contract):
    regions: list[RegionRow] = Field(max_length=4)


class ProductData(Contract):
    products: list[ProductRow]


class CustomerData(Contract):
    customers: list[CustomerRow]


class OrderRow(Contract):
    order_id: str
    order_date: date
    customer_id: str
    product_id: str
    sales_rep_id: str
    region: Region
    quantity: int
    list_price: Decimal
    sale_price: Decimal
    revenue: Decimal
    cost: Decimal
    profit: Decimal
    discount_rate: Decimal
    approval_status: Approval


class OrdersData(Contract):
    orders: list[OrderRow] = Field(max_length=50)


class ToolError(Contract):
    code: ErrorCode
    message: str
    retryable: bool


class ToolMeta(Contract):
    latency_ms: float = Field(ge=0)
    current_period: Period | None = None
    comparison_period: Period | None = None
    comparison_available: bool | None = None
    returned_count: int | None = None
    total_matches: int | None = None
    truncated: bool | None = None


class ToolResult(Contract):
    success: bool
    tool_name: str
    data: OverviewData | RegionData | ProductData | CustomerData | OrdersData | PolicySearchResult | WriteResultData | None
    summary: str | None
    evidence_ids: list[str]
    error: ToolError | None
    meta: ToolMeta

    @model_validator(mode="after")
    def envelope_consistency(self):
        if self.success:
            if self.data is None or self.error is not None or self.summary is None:
                raise ValueError("Successful tools require data and summary, without an error")
        elif self.data is not None or self.summary is not None or self.evidence_ids or self.error is None:
            raise ValueError("Failed tools require an error and no data, summary or evidence")
        return self


class ToolMetadata(Contract):
    name: str
    description: str
    permission_level: Literal["READ", "WRITE_APPROVAL_REQUIRED"] = "READ"
    timeout_seconds: int = Field(default=8, gt=0)
    input_schema: dict
