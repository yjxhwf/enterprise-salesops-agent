"""Machine-readable scenario expectations, not an Agent answer or eval harness."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Directions(StrictModel):
    revenue: Literal["increase"]
    profit: Literal["decrease"]
    margin: Literal["decrease"]


class Cause(StrictModel):
    fact_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    anomaly_type: Literal["deep_discount", "low_margin_mix_shift", "high_margin_volume_decline"]
    region: str | None = None
    customers: list[str] = Field(default_factory=list)


class Thresholds(StrictModel):
    revenue_growth_pct: tuple[Decimal, Decimal]
    profit_growth_pct: tuple[Decimal, Decimal]
    minimum_margin_drop_pp: Decimal = Field(gt=0)
    minimum_east_profit_drop_pct: Decimal = Field(gt=0)
    discount_range: tuple[Decimal, Decimal]
    a12_minimum_discount_rate: Decimal = Field(gt=0, le=1)
    minimum_b07_share_gain_pp: Decimal = Field(gt=0)
    c03_revenue_growth_pct: tuple[Decimal, Decimal]
    minimum_standard_margin_gap_pp: Decimal = Field(gt=0)
    minimum_discount_recovery_fraction: Decimal = Field(gt=0, le=1)

    @model_validator(mode="after")
    def ordered_ranges(self):
        for bounds in (self.revenue_growth_pct, self.profit_growth_pct, self.discount_range, self.c03_revenue_growth_pct):
            if bounds[0] > bounds[1]:
                raise ValueError("Threshold lower bound must not exceed upper bound")
        if not 0 < self.discount_range[0] <= self.discount_range[1] < self.a12_minimum_discount_rate:
            raise ValueError("Deep discount bounds must be below the policy threshold")
        return self


class GroundTruth(StrictModel):
    dataset_version: str = Field(min_length=1)
    seed: int
    analysis_period: str = Field(pattern=r"^2026-08$")
    comparison_period: str = Field(pattern=r"^2026-07$")
    expected_direction: Directions
    primary_causes: list[Cause] = Field(min_length=3, max_length=3)
    expected_policy_violation_product: str
    expected_problem_customers: list[str] = Field(min_length=2, max_length=2)
    fact_ids: list[str] = Field(min_length=9)
    thresholds: Thresholds

    @model_validator(mode="after")
    def coherent_causes(self):
        expected = ["deep_discount", "low_margin_mix_shift", "high_margin_volume_decline"]
        if [c.anomaly_type for c in self.primary_causes] != expected:
            raise ValueError("Three ordered, distinct cause types are required")
        first = self.primary_causes[0]
        if not first.region or first.customers != self.expected_problem_customers:
            raise ValueError("Discount cause needs a region and matching customer evidence")
        if first.product_id != self.expected_policy_violation_product:
            raise ValueError("Policy product must match the discount cause")
        if len(set(self.fact_ids)) != len(self.fact_ids) or any(c.fact_id not in self.fact_ids for c in self.primary_causes):
            raise ValueError("Fact identifiers must be unique and include each primary cause")
        return self


def load_ground_truth(path: Path | None = None) -> GroundTruth:
    path = path or Path(__file__).parent / "fixtures" / "ground_truth.json"
    return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
