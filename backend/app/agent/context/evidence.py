"""Resolver over the existing tool history, not a second evidence store."""
from copy import deepcopy
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.policy_evidence import policy_catalog
from backend.app.llm.gateway import GatewayError, ErrorCode


class EvidenceLedger:
    def __init__(self, state):
        self.state = state
        self._entries = {}
        for kind, values in (("BUSINESS_DATA", extract_evidence(state.get("tool_history", []))),
                             ("POLICY", policy_catalog(state.get("tool_history", [])))):
            for eid, value in values.items():
                self.register(eid, kind, value)
        action = state.get("action_result") or {}
        if action.get("success") and (action.get("data") or {}).get("evidence_kind") == "ACTION":
            for eid in action.get("evidence_ids", []):
                self.register(eid, "ACTION", action["data"])

    def register(self, evidence_id, kind, value):
        if kind not in {"BUSINESS_DATA", "POLICY", "ACTION"}:
            raise ValueError("Unknown evidence kind")
        if kind == "POLICY" and not evidence_id.startswith("POLICY::") or kind == "ACTION" and not evidence_id.startswith("ACTION::") or kind == "BUSINESS_DATA" and evidence_id.startswith(("POLICY::", "ACTION::")):
            raise ValueError("Evidence identity has incompatible kind")
        existing = self._entries.get(evidence_id)
        if existing and existing != (kind, value):
            raise ValueError("Conflicting evidence identity")
        self._entries[evidence_id] = (kind, value)

    def ids(self, kind=None):
        return sorted(eid for eid, (actual, _) in self._entries.items() if kind is None or actual == kind)

    def resolve(self, evidence_id, kind=None):
        entry = self._entries.get(evidence_id)
        if entry is None or (kind is not None and kind != entry[0]):
            raise GatewayError(ErrorCode.EVIDENCE_NOT_VISIBLE_IN_CONTEXT)
        return deepcopy(entry[1])

    def resolve_many(self, evidence_ids, kind=None):
        return [self.resolve(eid, kind) for eid in dict.fromkeys(evidence_ids)]

    def usage(self):
        result = {}
        for index, record in enumerate(self.state.get("tool_history", []), 1):
            keys = list(extract_evidence([record])) + list(policy_catalog([record]))
            for eid in keys:
                result.setdefault(eid, dict(first_seen_step=index, selected_count=0, last_selected_step=0,
                    used_in_synthesis=False, used_in_action_proposal=False))
        for index, observation in enumerate(self.state.get("observations", []), 1):
            for eid in set(observation.get("evidence_ids", [])):
                if eid in result:
                    result[eid]["selected_count"] += 1
                    result[eid]["last_selected_step"] = index
        draft = self.state.get("analysis_draft") or {}
        cited = {eid for f in draft.get("findings", []) for eid in f.get("supporting_evidence_ids", [])}
        cited.update(eid for r in draft.get("recommendations", []) for eid in r.get("policy_evidence_ids", []))
        action_ids = (self.state.get("pending_action") or {}).get("evidence_ids", [])
        for eid, usage in result.items():
            usage["used_in_synthesis"] = eid in cited
            usage["used_in_action_proposal"] = eid in action_ids
        return result


def selection_rank(observations):
    usage = {}
    for index, observation in enumerate(observations, 1):
        for eid in set(observation.get("evidence_ids", [])):
            count, _ = usage.get(eid, (0, 0))
            usage[eid] = (count+1, index)
    return lambda eid: (-usage.get(eid, (0, 0))[0], -usage.get(eid, (0, 0))[1], eid)
