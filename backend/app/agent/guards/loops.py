import hashlib
import json
from datetime import date, datetime

from backend.app.agent.evidence import extract_evidence


def action_signature(name, arguments):
    def normalize(value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    # Keep raw arguments out of runtime events/state signatures.
    payload = json.dumps([name, normalize(arguments)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def progress_fingerprint(state):
    from backend.app.agent.policy_evidence import policy_catalog
    evidence = sorted([*extract_evidence(state["tool_history"]), *policy_catalog(state["tool_history"])])
    remaining = [" ".join(step["objective"].split()) for step in (state.get("plan") or {}).get("steps", [])
                 if step["status"] in {"PENDING", "IN_PROGRESS"}]
    payload = json.dumps([evidence, remaining], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
