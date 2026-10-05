"""In-memory source checks, not an evidence persistence or semantic entailment system."""
import json
from decimal import Decimal, InvalidOperation

from backend.app.llm.gateway import ErrorCode, GatewayError


def invalid_output():
    raise GatewayError(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR)


def source_index(history):
    index = {}
    for record in history:
        if record["success"]:
            for evidence_id in record["evidence_ids"]:
                index.setdefault(evidence_id, []).append(record)
    return index


def resolve_pointer(record, pointer):
    if not pointer.startswith("/data/"):
        invalid_output()
    value = record
    try:
        for part in pointer[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not part.isdecimal():
                    invalid_output()
                value = value[int(part)]
            else:
                value = value[part]
    except (KeyError, IndexError, TypeError, ValueError):
        invalid_output()
    # References target individual facts, not a whole data dump.
    if isinstance(value, (dict, list)):
        invalid_output()
    return value


def validate_references(evidence_ids, references, history, *, source_tool=None):
    index = source_index(history)
    referenced_ids = {ref.evidence_id for ref in references}
    all_ids = set(evidence_ids) | referenced_ids
    if not evidence_ids or any(eid not in index for eid in all_ids):
        invalid_output()
    # Whole-record citations are valid too. Every declared ID must exist, while
    # every supplied scalar reference is additionally checked below.
    # A derived finding can compare the latest region result with a prior company
    # overview. source_tool identifies its primary source, not its only source.
    if source_tool is not None and not any(record["tool_name"] == source_tool
                                           for eid in all_ids for record in index[eid]):
        invalid_output()
    for ref in references:
        candidates = index[ref.evidence_id]
        if not candidates:
            invalid_output()
        matched = False
        for record in candidates:
            try:
                value = resolve_pointer(record, ref.json_pointer)
            except GatewayError:
                continue
            if same_scalar(value, ref.value, ref.json_pointer):
                # Order evidence can only support its own row, not another order.
                if ref.evidence_id.startswith("order:"):
                    parts = ref.json_pointer.split("/")
                    if len(parts) < 5 or parts[2] != "orders":
                        continue
                    row = record["data"]["orders"][int(parts[3])]
                    if ref.evidence_id != "order:" + row["order_id"]:
                        continue
                # Store the exact source representation, never the model's number formatting.
                ref.value = value
                matched = True
                break
        if not matched:
            invalid_output()
    return sorted(all_ids)


def same_scalar(source, proposed, pointer):
    if json.dumps(source, sort_keys=True) == json.dumps(proposed, sort_keys=True):
        return True
    # JSON models may emit numeric 0 for the tool's Decimal string "0".
    # Compare exact decimal values (no rounding/tolerance), then canonicalize to
    # the source. IDs, booleans, null and arbitrary text never get numeric coercion.
    field = pointer.rsplit("/", 1)[-1]
    if field.endswith("_id") or field == "id" or isinstance(source, bool) or isinstance(proposed, bool):
        return False
    if not isinstance(source, (str, int, float)) or not isinstance(proposed, (str, int, float)):
        return False
    try:
        left, right = Decimal(str(source)), Decimal(str(proposed))
        return left.is_finite() and right.is_finite() and left == right
    except InvalidOperation:
        return False
