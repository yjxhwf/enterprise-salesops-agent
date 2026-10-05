"""Value-free Action Proposal diagnostics; never relax or normalize a proposal."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.app.actions.schemas import ActionError

FIELDS = frozenset('action_required proposal action_type tool_name arguments reason related_finding_indexes business_evidence_ids policy_evidence_ids policy_basis risk_level expected_outcome customer_id priority suggested_action target_type target_id severity'.split())
TYPES = frozenset(('object', 'array', 'string', 'integer', 'number', 'boolean', 'null', 'missing', 'non_json'))
MISSING = object()


def json_type(value):
    if value is MISSING: return 'missing'
    if value is None: return 'null'
    if type(value) is bool: return 'boolean'
    if isinstance(value, dict): return 'object'
    if isinstance(value, list): return 'array'
    if isinstance(value, str): return 'string'
    if type(value) is int: return 'integer'
    if type(value) is float: return 'number'
    return 'non_json'


def safe_shape(value, depth=0):
    result = {'type': json_type(value)}
    if depth >= 3: return result
    if isinstance(value, dict):
        result.update(fields={k: safe_shape(value[k], depth+1) for k in sorted(FIELDS & value.keys())},
                      unknown_field_count=len(value.keys()-FIELDS))
    elif isinstance(value, list):
        result.update(items=[safe_shape(v, depth+1) for v in value[:3]], item_count=len(value))
    return result


def validate_shape(value, depth=0):
    """Validate a shape, not arbitrary nested metadata supplied to the exporter."""
    if depth > 3 or type(value) is not dict or value.get('type') not in TYPES:
        raise ValueError('Invalid diagnostic shape')
    if value.keys()-{'type', 'fields', 'unknown_field_count', 'items', 'item_count'}:
        raise ValueError('Unexpected diagnostic shape field')
    for name in ('unknown_field_count', 'item_count'):
        if name in value and (type(value[name]) is not int or value[name] < 0):
            raise ValueError('Invalid shape count')
    if 'fields' in value:
        if type(value['fields']) is not dict or value['fields'].keys()-FIELDS:
            raise ValueError('Unknown diagnostic field')
        for nested in value['fields'].values(): validate_shape(nested, depth+1)
    if 'items' in value:
        if type(value['items']) is not list or len(value['items']) > 3:
            raise ValueError('Invalid diagnostic items')
        for nested in value['items']: validate_shape(nested, depth+1)
    return value


class ActionProposalValidationDiagnostic(BaseModel):
    model_config = ConfigDict(extra='forbid')
    error_code: Literal['ACTION_PROPOSAL_VALIDATION_ERROR'] = 'ACTION_PROPOSAL_VALIDATION_ERROR'
    field_path: str
    reason: str
    expected_type: str
    actual_type: Literal['object', 'array', 'string', 'integer', 'number', 'boolean', 'null', 'missing', 'non_json']
    safe_actual_shape: dict
    validation_rule: str
    validation_source: Literal['PYDANTIC_SCHEMA', 'BUSINESS_VALIDATOR']
    evidence_related: bool = False

    @field_validator('safe_actual_shape')
    @classmethod
    def shape_only(cls, value):
        return validate_shape(value)


def diagnostic(path, rule, expected, actual, *, source='BUSINESS_VALIDATOR', reason=None):
    return ActionProposalValidationDiagnostic(field_path=path, reason=reason or rule,
        expected_type=expected, actual_type=json_type(actual), safe_actual_shape=safe_shape(actual),
        validation_rule=rule, validation_source=source,
        evidence_related='evidence' in path or 'EVIDENCE' in rule).model_dump(mode='json')


def reject(path, rule, expected, actual):
    error = ActionError('ACTION_PROPOSAL_VALIDATION_ERROR')
    error.action_proposal_diagnostic = diagnostic(path, rule, expected, actual)
    raise error


def pydantic_diagnostic(error, raw, *, prefix='', schema=None):
    # Do not retain input, ctx, custom ValueError text, or dynamically supplied keys.
    item = error.errors(include_url=False, include_context=False, include_input=False)[0]
    loc, kind, message = item['loc'], item['type'], item['msg']
    path = '.'.join([p for p in (prefix, *[str(v) if type(v) is int else v if v in FIELDS else '<extra_field>' for v in loc]) if p]) or '$'
    actual = raw
    for part in loc:
        if isinstance(actual, dict): actual = actual.get(part, MISSING)
        elif isinstance(actual, list) and type(part) is int and 0 <= part < len(actual): actual = actual[part]
        else: actual = MISSING; break
    expected = {'dict_type':'object', 'model_type':'object', 'list_type':'array', 'string_type':'string',
                'int_type':'integer', 'int_parsing':'integer', 'bool_type':'boolean', 'bool_parsing':'boolean',
                'extra_forbidden':'absent', 'literal_error':'allowed_enum'}.get(kind, 'schema_constraint')
    if schema is not None and kind == 'missing':
        node = schema.model_json_schema(); definitions = node.get('$defs', {})
        for part in loc:
            if '$ref' in node: node = definitions.get(node['$ref'].split('/')[-1], {})
            if 'anyOf' in node: node = next((v for v in node['anyOf'] if v.get('type') != 'null'), {})
            if '$ref' in node: node = definitions.get(node['$ref'].split('/')[-1], {})
            node = node.get('items', {}) if type(part) is int else node.get('properties', {}).get(part, {})
        expected = node.get('type', 'required_field')
    reasons = {'missing':'FIELD_REQUIRED', 'extra_forbidden':'EXTRA_FIELD_FORBIDDEN', 'literal_error':'ENUM_VALUE_INVALID',
               'value_error':'CONTRACT_CONSISTENCY_FAILED'}
    # msg is used only for the exact built-in message, never copied to diagnostics.
    reason = 'FIELD_REQUIRED' if kind == 'missing' and message == 'Field required' else reasons.get(kind, 'SCHEMA_CONSTRAINT_REJECTED')
    return diagnostic(path, kind, expected, actual, source='PYDANTIC_SCHEMA', reason=reason)
