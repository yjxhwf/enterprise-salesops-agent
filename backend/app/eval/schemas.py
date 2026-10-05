from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Category = Literal['Normal', 'Failure', 'Adversarial', 'RAG', 'Permission']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class FactAssertion(StrictModel):
    metric: str
    subject_id: str
    field: Literal['current_value', 'comparison_value', 'change_value', 'absolute_change'] = 'current_value'
    expected: Decimal
    tolerance: Decimal = Field(default=Decimal('0'), ge=0)
    scale: Decimal = Decimal('1')
    required_evidence_kind: Literal['BUSINESS_DATA'] = 'BUSINESS_DATA'
    semantic_metric: str | None = None


class EvalCase(StrictModel):
    case_id: str
    category: Category
    name: str
    query: str
    mode: Literal['OFFLINE_SCRIPTED'] = 'OFFLINE_SCRIPTED'
    expected_status: str
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_facts: list[FactAssertion] = Field(default_factory=list)
    expected_evidence_kinds: list[str] = Field(default_factory=list)
    expected_policy_docs: list[str] = Field(default_factory=list)
    expected_stop_reason: str | None = None
    expected_write_behavior: Literal['NONE', 'EXACTLY_ONE'] = 'NONE'
    tags: list[str] = Field(default_factory=list)


class ExecutionInput(StrictModel):
    """The sole case input visible to the executor; expectations cannot cross here."""
    case_id: str
    query: str


class AssertionResult(StrictModel):
    name: str
    passed: bool
    expected: str | int | bool | None = None
    actual: str | int | bool | None = None


class EvalResult(StrictModel):
    case_id: str
    category: Category
    passed: bool
    expected_outcome: str
    actual_outcome: str
    assertions: list[AssertionResult]
    metrics: dict
    failure_reason: list[str]
    trace_summary: dict


class EvalReport(StrictModel):
    mode: Literal['OFFLINE_SCRIPTED'] = 'OFFLINE_SCRIPTED'
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    by_category: dict
    metrics: dict
    results: list[EvalResult]
    limitations: list[str]
