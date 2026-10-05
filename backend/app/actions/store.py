"""Trusted process-local capability store. Tokens never enter public snapshots."""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets
from threading import RLock
from uuid import uuid4
from pydantic import SecretStr
from backend.app.actions.schemas import PendingAction, ActionError
from backend.app.actions.validation import fingerprint
from backend.app.observability.integration import approval_event


def utc_now():
    return datetime.now(timezone.utc)


def seal(pending):
    data = pending.model_dump(mode="json", exclude={"approval_status"})
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class IssuedAction:
    pending: PendingAction
    token: SecretStr = field(repr=False)


@dataclass
class _Record:
    pending: PendingAction
    token_hash: str = field(repr=False)
    original_seal: str = field(repr=False)
    result: dict | None = None


class ApprovalStore:
    def __init__(self, *, clock=utc_now, ttl_minutes=30):
        if ttl_minutes <= 0:
            raise ValueError("TTL must be positive")
        self.clock, self.ttl = clock, timedelta(minutes=ttl_minutes)
        self._lock = RLock()
        self._records = {}
        self._attempts = {}  # fingerprint -> successful result or None (failed/uncertain)
        self.events = []  # safe process-local audit identifiers only

    def _event(self, kind, pending):
        self.events.append(dict(event_type=kind, action_id=pending.action_id, tool_name=pending.tool_name))
        approval_event(kind, pending)

    def create_pending(self, proposal, run_id, scope):
        now = self.clock()
        scope = {**deepcopy(scope), "run_id": run_id}
        pending = PendingAction(action_id=str(uuid4()), run_id=run_id, scope=scope,
            tool_name=proposal.tool_name, arguments=deepcopy(proposal.arguments), reason=proposal.reason,
            evidence_ids=list(dict.fromkeys(proposal.business_evidence_ids+proposal.policy_evidence_ids)),
            business_evidence_ids=proposal.business_evidence_ids, policy_evidence_ids=proposal.policy_evidence_ids,
            related_finding_indexes=proposal.related_finding_indexes, risk_level=proposal.risk_level,
            expected_outcome=proposal.expected_outcome, created_at=now, expires_at=now+self.ttl,
            action_fingerprint=fingerprint(proposal.tool_name, proposal.arguments, scope))
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._records[pending.action_id] = _Record(pending, hashlib.sha256(token.encode()).hexdigest(), seal(pending))
            self._event("ACTION_PROPOSED", pending)
        return IssuedAction(pending.model_copy(deep=True), SecretStr(token))

    def _checked(self, action_id):
        record = self._records.get(action_id)
        if record is None:
            raise ActionError("ACTION_NOT_FOUND")
        p = record.pending
        if seal(p) != record.original_seal or p.action_fingerprint != fingerprint(p.tool_name, p.arguments, p.scope):
            raise ActionError("ACTION_TAMPERED")
        return record

    def get_pending(self, action_id):
        with self._lock:
            return self._checked(action_id).pending.model_copy(deep=True)

    def snapshot(self, action_id):
        with self._lock:
            record = self._checked(action_id)
            return record.pending.model_copy(deep=True), deepcopy(record.result)

    def _authorize(self, action_id, token):
        record = self._checked(action_id)
        if record.pending.approval_status != "PENDING":
            raise ActionError("INVALID_APPROVAL_STATE")
        if self.clock() >= record.pending.expires_at:
            raise ActionError("APPROVAL_EXPIRED")
        raw = token.get_secret_value() if isinstance(token, SecretStr) else token
        if not isinstance(raw, str) or not raw or not hmac.compare_digest(hashlib.sha256(raw.encode()).hexdigest(), record.token_hash):
            raise ActionError("INVALID_APPROVAL_TOKEN")
        return record

    def approve(self, action_id, token):
        with self._lock:
            record = self._authorize(action_id, token)
            record.pending.approval_status = "APPROVED"
            record.token_hash = ""  # single-use capability consumed
            self._event("ACTION_APPROVED", record.pending)
            return record.pending.model_copy(deep=True)

    def reject(self, action_id, token):
        with self._lock:
            record = self._authorize(action_id, token)
            record.pending.approval_status = "REJECTED"
            record.token_hash = ""
            self._event("ACTION_REJECTED", record.pending)
            return record.pending.model_copy(deep=True)

    def mark_executed(self, action_id, executor):
        """Lock spans revalidation and one transaction; no approved claim can race."""
        with self._lock:
            record = self._checked(action_id)
            p = record.pending
            if p.approval_status == "EXECUTED":
                raise ActionError("ALREADY_EXECUTED")
            if p.approval_status != "APPROVED":
                raise ActionError("INVALID_APPROVAL_STATE")
            if self.clock() >= p.expires_at:
                p.approval_status = "FAILED"
                self._event("ACTION_FAILED", p)
                raise ActionError("APPROVAL_EXPIRED")
            fp = p.action_fingerprint
            if fp in self._attempts:
                cached = self._attempts[fp]
                if cached is None:
                    p.approval_status = "FAILED"
                    self._event("ACTION_FAILED", p)
                    raise ActionError("WRITE_ALREADY_ATTEMPTED")
                record.result = deepcopy(cached)
                p.approval_status = "EXECUTED"
                self._event("ACTION_EXECUTED", p)
                return deepcopy(cached)
            self._attempts[fp] = None  # Fail closed even if commit outcome becomes uncertain.
            try:
                result = executor(p.model_copy(deep=True))
            except Exception:
                p.approval_status = "FAILED"
                self._event("ACTION_FAILED", p)
                raise ActionError("WRITE_EXECUTION_FAILED") from None
            record.result = result.model_dump(mode="json")
            if result.success:
                p.approval_status = "EXECUTED"
                self._attempts[fp] = deepcopy(record.result)
                self._event("ACTION_EXECUTED", p)
            else:
                p.approval_status = "FAILED"
                self._event("ACTION_FAILED", p)
            return deepcopy(record.result)
