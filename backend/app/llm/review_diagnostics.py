"""Value-free diagnostics for the Review plan contract; no response or prompt logging."""
from backend.app.llm.gateway import ErrorCode, GatewayError


def json_type(value):
    if value is None:return "null"
    if isinstance(value, bool):return "boolean"
    if isinstance(value, dict):return "object"
    if isinstance(value, list):return "array"
    if isinstance(value, str):return "string"
    if isinstance(value, (int, float)):return "number"
    return "non_json"


def safe_plan_shape(value, depth=0):
    kind = json_type(value)
    if depth >= 4:return {"type": kind}
    if kind == "object":
        allowed = {"steps", "objective", "expected_output"}
        return {"type": kind, "fields": {key: safe_plan_shape(value[key], depth+1) for key in sorted(allowed & value.keys())},
                "unknown_field_count": len(set(value)-allowed)}
    if kind == "array":
        return {"type": kind, "items": [safe_plan_shape(item, depth+1) for item in value[:3]],
                "item_count": len(value), "omitted_item_count": max(0,len(value)-3)}
    return {"type": kind}  # never serialize string/scalar values, including secrets


class ReviewContractError(GatewayError):
    def __init__(self, args, error):
        super().__init__(ErrorCode.MODEL_OUTPUT_VALIDATION_ERROR)
        value = args.get("updated_plan")
        fields = []
        for item in error.errors(include_url=False, include_context=False, include_input=False):
            loc = item["loc"]
            safe_loc = [part if isinstance(part, int) or part in {"updated_plan", "steps", "objective", "expected_output"} else "<field>" for part in loc]
            expected = "object|null" if tuple(loc)==("updated_plan",) else "array" if loc and loc[-1]=="steps" else "string" if loc and loc[-1] in {"objective","expected_output"} else "canonical Review contract"
            fields.append(dict(field_path=".".join(map(str,safe_loc)) or "$", rule_id=item["type"], expected_type=expected))
        self.structured_output_diagnostic = dict(node="review_progress", field_path="updated_plan",
            expected_type="object|null (RemainingPlan with steps array)",
            actual_type=json_type(value) if "updated_plan" in args else "missing",
            safe_shape=safe_plan_shape(value) if "updated_plan" in args else {"type":"missing"},
            validation_code=self.code.value, violations=fields)
