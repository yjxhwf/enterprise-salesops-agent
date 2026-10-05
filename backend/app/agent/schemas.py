"""Business outputs only; model responses are untrusted until validated."""

from datetime import date
from enum import StrEnum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from backend.app.rag.schemas import PolicyEvidence


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class AgentStatus(StrEnum):
    RUNTIME_STOPPED = "RUNTIME_STOPPED"
    RECEIVED = "RECEIVED"
    GOAL_UNDERSTOOD = "GOAL_UNDERSTOOD"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    PLANNED = "PLANNED"
    TOOL_SELECTED = "TOOL_SELECTED"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    INVESTIGATING = "INVESTIGATING"
    INVESTIGATION_COMPLETE = "INVESTIGATION_COMPLETE"
    DRAFT_READY = "DRAFT_READY"
    TOOL_ERROR = "TOOL_ERROR"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    ACTION_APPROVED = "ACTION_APPROVED"
    ACTION_REJECTED = "ACTION_REJECTED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    ACTION_FAILED = "ACTION_FAILED"


class TaskType(StrEnum):
    BUSINESS_OVERVIEW = "BUSINESS_OVERVIEW"
    ROOT_CAUSE_ANALYSIS = "ROOT_CAUSE_ANALYSIS"
    REGION_ANALYSIS = "REGION_ANALYSIS"
    PRODUCT_ANALYSIS = "PRODUCT_ANALYSIS"
    CUSTOMER_ANALYSIS = "CUSTOMER_ANALYSIS"
    ORDER_INVESTIGATION = "ORDER_INVESTIGATION"


class GoalUnderstanding(Contract):
    supported: bool = Field(strict=True)
    task_type: TaskType | None
    normalized_goal: str = Field(min_length=1, max_length=500)
    start_date: date | None
    end_date: date | None
    requested_outputs: list[str] = Field(max_length=8)
    needs_clarification: bool = Field(strict=True)
    clarification_question: str | None = Field(max_length=300)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def iso_date(cls, value):
        if value is not None and (not isinstance(value, str) or len(value) != 10):
            raise ValueError("Expected an ISO date string or null")
        return value

    @model_validator(mode="after")
    def coherent_goal(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("Date range is reversed")
        if self.supported:
            if self.task_type is None:
                raise ValueError("Supported goals require a task type")
            if self.needs_clarification:
                if not self.clarification_question or not self.clarification_question.strip():
                    raise ValueError("Clarification requires a question")
            elif not self.start_date or not self.end_date or not self.requested_outputs:
                raise ValueError("Actionable goals require dates and outputs")
        if not self.needs_clarification and self.clarification_question is not None:
            raise ValueError("A question requires clarification status")
        return self


class PlanStep(Contract):
    step_id: str = Field(min_length=1, max_length=40)
    objective: str = Field(min_length=1, max_length=500)
    expected_output: str = Field(min_length=1, max_length=500)
    status: Literal["PENDING"] = "PENDING"


class Plan(Contract):
    steps: list[PlanStep] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def unique_steps(self):
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError("Step IDs must be unique")
        return self


class ToolSelection(Contract):
    tool_name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, JsonValue]
    selection_summary: str | None = Field(default=None, max_length=300)


class ToolCandidate(ToolSelection):
    provider_order: int = Field(ge=0, strict=True)


class ToolProposals(Contract):
    candidates: list[ToolCandidate] = Field(min_length=1)


class InvestigationStep(PlanStep):
    status: Literal["PENDING", "IN_PROGRESS", "COMPLETED", "SKIPPED", "FAILED"] = "PENDING"
    skip_reason: Literal["SUFFICIENT_EVIDENCE"] | None = None


class RemainingObjective(Contract):
    objective: str = Field(min_length=1, max_length=500)
    expected_output: str = Field(min_length=1, max_length=500)


class RemainingPlan(Contract):
    """Bounded remaining work; completed work lives separately in State."""
    steps: list[RemainingObjective] = Field(min_length=1, max_length=6)


class EvidenceReference(Contract):
    evidence_id: str = Field(min_length=1)
    json_pointer: str = Field(pattern=r"^/data/")
    value: str | int | float | bool | None


class ComparisonBasis(StrEnum):
    PREVIOUS_EQUAL_LENGTH_PERIOD = "PREVIOUS_EQUAL_LENGTH_PERIOD"
    YEAR_OVER_YEAR = "YEAR_OVER_YEAR"


class CanonicalEvidence(Contract):
    evidence_kind: Literal["BUSINESS_DATA"] = "BUSINESS_DATA"
    evidence_id: str
    source_tool: str
    fact_key: str
    display_value: dict[str, JsonValue]
    summary: str
    internal_reference: dict[str, JsonValue]
    comparison_basis: ComparisonBasis | None = None
    comparison_label: str | None = None
    current_period: dict[str, str] | None = None
    comparison_period: dict[str, str] | None = None
    scope: dict[str, JsonValue] = Field(default_factory=dict)


class CanonicalFact(Contract):
    fact_id: str
    evidence_id: str
    subject_type: str
    subject_id: str
    period: dict[str, str] | None
    scope: dict[str, JsonValue]
    metric: str
    semantic_metric: str | None = None
    business_label: str | None = None
    derived_value: JsonValue = None
    direction_semantics: str | None = None
    current_value: JsonValue
    display_value: str
    comparison_basis: ComparisonBasis | None
    comparison_value: JsonValue
    comparison_display_value: str | None
    change_value: JsonValue
    change_display_value: str | None
    change_direction: Literal["INCREASE", "DECREASE", "UNCHANGED", "NOT_AVAILABLE"]
    current_sign: Literal["POSITIVE", "NEGATIVE", "ZERO", "NOT_NUMERIC", "NOT_AVAILABLE"]
    unit: str
    change_unit: str | None
    source_tool: str
    rendered_text: str


class ProgressReview(Contract):
    decision: Literal["CONTINUE", "COMPLETE"]
    selected_evidence_ids: list[str] = Field(min_length=1, max_length=10)
    next_objective: str | None = Field(max_length=500)
    updated_plan: RemainingPlan | None = Field(description=
        'Canonical JSON object {"steps": [{"objective": "...", "expected_output": "..."}]} for CONTINUE; JSON null for COMPLETE. Never a bare array or JSON-encoded string.')
    remaining_evidence_gap: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def coherent_decision(self):
        if self.decision == "CONTINUE":
            if not self.updated_plan or not self.next_objective or not self.next_objective.strip():
                raise ValueError("Continuing requires new remaining work")
            gap = (self.remaining_evidence_gap or "").strip()
            vague = r"[\s。.!！,，;；]*(?:需要进一步分析|还需要更多信息|继续深入|需要更多证据|还可以继续深入分析|need more (?:information|evidence)|further analysis|investigate further)[\s。.!！,，;；]*"
            if len(gap) < 6 or re.fullmatch(vague, gap, re.I):
                raise ValueError("Continuing requires a specific business evidence gap")
        elif self.next_objective is not None or self.updated_plan is not None or self.remaining_evidence_gap is not None:
            raise ValueError("Complete reviews have no proposed next work")
        return self


class InvestigationObservation(Contract):
    observation_id: str
    completed_step_id: str
    source_tool: str
    facts: list[CanonicalFact] = Field(default_factory=list)
    policy_evidence: list[PolicyEvidence] = Field(default_factory=list)
    evidence_ids: list[str]
    decision: Literal["CONTINUE", "COMPLETE"]
    remaining_evidence_gap: str | None = None
    next_objective: str | None = None
    plan_before: dict[str, JsonValue]
    plan_after: dict[str, JsonValue] | None

    @model_validator(mode="after")
    def evidence_required(self):
        if not self.facts and not self.policy_evidence:
            raise ValueError("Observation requires business or policy evidence")
        return self


class ToolExecutionRecord(Contract):
    tool_name: str
    arguments: dict[str, JsonValue]
    success: bool
    summary: str | None
    evidence_ids: list[str]
    error_code: str | None
    meta: dict[str, JsonValue]
    data: dict[str, JsonValue] | None
    plan_step_id: str


class DraftFinding(Contract):
    claim_type: Literal["CONTRIBUTING_FACTOR", "OBSERVATION"]
    title: str = Field(min_length=1, max_length=180)
    interpretation: str = Field(min_length=1, max_length=1000)
    supporting_evidence_ids: list[str] = Field(min_length=1, max_length=15)


class Recommendation(Contract):
    title: str = Field(min_length=1, max_length=180)
    action: str = Field(min_length=1, max_length=700)
    related_finding_index: int = Field(ge=0, strict=True)
    policy_evidence_ids: list[str] = Field(default_factory=list, max_length=5)
    policy_interpretation: str | None = Field(default=None, max_length=600)


class GroundedAnalysisDraft(Contract):
    executive_interpretation: str = Field(min_length=1, max_length=900)
    findings: list[DraftFinding] = Field(min_length=1, max_length=6)
    recommendations: list[Recommendation] = Field(min_length=1, max_length=6)
    limitations: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def related_findings_exist(self):
        if any(item.related_finding_index >= len(self.findings) for item in self.recommendations):
            raise ValueError("Recommendations must reference an existing finding")
        return self
