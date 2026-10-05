"""Inputs expose business fields only; identity/status/time are server-owned."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class WriteContract(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, str_strip_whitespace=True)


class CRMTaskInput(WriteContract):
    customer_id: str = Field(min_length=1, max_length=12)
    reason: str = Field(min_length=1, max_length=1000)
    priority: Literal["LOW", "MEDIUM", "HIGH"]
    suggested_action: str = Field(min_length=1, max_length=1000)


class BusinessAlertInput(WriteContract):
    target_type: Literal["CUSTOMER", "PRODUCT", "REGION", "COMPANY"]
    target_id: str = Field(min_length=1, max_length=40)
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    reason: str = Field(min_length=1, max_length=1000)


class WriteResultData(WriteContract):
    evidence_kind: Literal["ACTION"] = "ACTION"
    action_id: str
    object_id: str
    object_type: Literal["CRM_TASK", "BUSINESS_ALERT"]
    status: Literal["OPEN"] = "OPEN"
    created_at: datetime


WRITE_MODELS = {"create_crm_task": CRMTaskInput, "create_business_alert": BusinessAlertInput}
