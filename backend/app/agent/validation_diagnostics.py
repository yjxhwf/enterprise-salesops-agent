"""Bounded rejection diagnostics; never serialize the rejected draft or provider error."""
import os
import re
from typing import Literal

from pydantic import Field

from backend.app.agent.schemas import Contract
from backend.app.config import get_settings
from backend.app.llm.gateway import ErrorCode, GatewayError


class ValidationDiagnostic(Contract):
    rule_id: Literal["PROFIT_SIGN_CONSISTENCY", "COMPARISON_BASIS", "DISCOUNT_DIRECTION", "EXCLUSION_CLAIM", "ABSOLUTE_CAUSALITY", "CONTRIBUTION_EVIDENCE"]
    field_path: str = Field(max_length=100)
    reason: str = Field(max_length=120)
    claim_excerpt: str = Field(max_length=160)
    related_evidence_ids: list[str]
    evidence_profit_sign: Literal["POSITIVE"] | None = None
    language_category: Literal["CURRENT_NEGATIVE_PROFIT_ASSERTION"] | None = None


def scrub(text):
    key = get_settings().LLM_API_KEY
    secrets = [key.get_secret_value()] if key else []
    secrets += [v for k, v in os.environ.items() if re.search(r"SECRET|TOKEN|PASSWORD|API_KEY", k, re.I) and len(v) >= 6]
    for value in sorted(set(secrets), key=len, reverse=True):
        if value:
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)(?:authorization|api[_ -]?key|secret|password|access[_ -]?token)\s*[\"']?\s*[:=]\s*[^，。；;\n]+", "[REDACTED]", text)
    text = re.sub(r"(?i)\b(?:bearer|basic)\s+\S+|\bsk-[\w-]+|\b[A-Za-z0-9_+/=-]{24,}\b", "[REDACTED]", text)
    return re.sub(r"[\x00-\x1f\x7f]", " ", text)


def safe_excerpt(text, position=0):
    clean = scrub(text)
    offset = len(scrub(text[:position]))
    start = max(0, offset - 50)
    return clean[start:start + 160]


class SemanticValidationError(GatewayError):
    def __init__(self, diagnostic):
        self.diagnostic = ValidationDiagnostic.model_validate(diagnostic).model_dump(mode="json", exclude_none=True)
        super().__init__(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR)


def reject(rule, field_path, reason, text, evidence, *, position=0, profit=False):
    raise SemanticValidationError(dict(rule_id=rule, field_path=field_path, reason=reason,
        claim_excerpt=safe_excerpt(text, position), related_evidence_ids=sorted({e["evidence_id"] for e in evidence}),
        evidence_profit_sign="POSITIVE" if profit else None,
        language_category="CURRENT_NEGATIVE_PROFIT_ASSERTION" if profit else None))
