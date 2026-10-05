from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class ActionProposal(Contract):
    action_type: Literal["CREATE_CRM_TASK", "CREATE_BUSINESS_ALERT"]
    tool_name: Literal["create_crm_task", "create_business_alert"]
    arguments: dict[str, JsonValue]
    reason: str = Field(min_length=1, max_length=1000)
    related_finding_indexes: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=1, max_length=5)
    business_evidence_ids: list[str] = Field(min_length=1, max_length=12)
    policy_evidence_ids: list[str] = Field(default_factory=list, max_length=5)
    policy_basis: bool = Field(default=False, strict=True)
    risk_level: Literal["LOW", "MEDIUM"]
    expected_outcome: str = Field(min_length=1, max_length=500)


class ActionDecision(Contract):
    action_required: bool = Field(strict=True)
    proposal: ActionProposal | None

    @model_validator(mode="after")
    def coherent(self):
        if self.action_required != (self.proposal is not None):
            raise ValueError("Action decision and proposal disagree")
        return self


class PendingAction(Contract):
    action_id: str
    run_id: str
    tool_name: str
    arguments: dict[str, JsonValue]
    reason: str
    evidence_ids: list[str]
    business_evidence_ids: list[str]
    policy_evidence_ids: list[str]
    related_finding_indexes: list[int]
    risk_level: Literal["LOW", "MEDIUM"]
    expected_outcome: str
    scope: dict[str, JsonValue]
    action_fingerprint: str
    approval_status: Literal["PENDING", "APPROVED", "REJECTED", "EXECUTED", "FAILED"] = "PENDING"
    created_at: datetime
    expires_at: datetime


class ApprovalRequest(Contract):
    approval_token: SecretStr
    approval_note: str | None = Field(default=None, max_length=500)


class RejectRequest(Contract):
    approval_token: SecretStr
    reject_reason: str | None = Field(default=None, max_length=500)


class ActionError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)
