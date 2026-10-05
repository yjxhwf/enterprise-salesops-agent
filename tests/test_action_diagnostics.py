"""FIX01 diagnostic detail without changing contracts or permitting writes."""
from copy import deepcopy
import json
import pytest
from pydantic import ValidationError
from backend.app.actions.diagnostics import ActionProposalValidationDiagnostic, safe_shape, validate_shape
from backend.app.actions.schemas import ActionError
from backend.app.actions.validation import validate_proposal, validate_tool
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.guards.runtime import GuardedGateway, guarded_node
from backend.app.agent.nodes.propose_action import make_propose_action_node
from backend.app.llm.gateway import GatewayError
from backend.app.llm.provider import ProductionGateway
from backend.app.observability.collector import TraceCollector, trace_scope
from backend.app.observability.integration import validation_error
from tests.action_fakes import proposal_for
from tests.test_run_trace import action_trace, offline
from tests.test_rag_retrieval import rag_index


@pytest.mark.parametrize('case,path,rule', [
    ('missing_tool','proposal.tool_name','missing'),
    ('action_enum','proposal.action_type','literal_error'),
    ('read_tool','proposal.tool_name','literal_error'),
    ('arguments_type','proposal.arguments','dict_type'),
    ('extra_argument','proposal.arguments.<extra_field>','extra_forbidden'),
    ('business_id','proposal.business_evidence_ids','INVALID_BUSINESS_EVIDENCE_ID'),
    ('policy_id','proposal.policy_evidence_ids','INVALID_POLICY_EVIDENCE_ID'),
    ('invisible','proposal.business_evidence_ids','EVIDENCE_NOT_VISIBLE'),
    ('finding','proposal.related_finding_indexes','INVALID_FINDING_INDEX'),
    ('mismatch','proposal.action_type','ACTION_TOOL_TYPE_MISMATCH'),
    ('target','proposal.arguments.customer_id','TARGET_NOT_GROUNDED_IN_EVIDENCE'),
    ('policy_required','proposal.policy_evidence_ids','POLICY_EVIDENCE_REQUIRED'),
])
def test_rejected_proposal_fields_and_safe_trace(action_trace, case, path, rule):
    state, _, service, _, client = action_trace
    proposal = proposal_for(client.payloads[-1][1])['proposal']
    sentinel = 'private-secret-sentinel'
    if case == 'missing_tool': proposal.pop('tool_name')
    elif case == 'action_enum': proposal['action_type'] = sentinel
    elif case == 'read_tool': proposal['tool_name'] = 'query_orders'
    elif case == 'arguments_type': proposal['arguments'] = [sentinel]
    elif case == 'extra_argument': proposal['arguments'][sentinel] = sentinel
    elif case == 'business_id': proposal['business_evidence_ids'] = [sentinel]
    elif case == 'policy_id': proposal['policy_evidence_ids'] = [sentinel]
    elif case == 'invisible':
        manifest = state['context_manifests'][-1]
        proposal['business_evidence_ids'] = [next(iter(extract_evidence(state['tool_history']).keys()-set(manifest['included_business_evidence_ids'])))]
    elif case == 'finding': proposal['related_finding_indexes'] = [999]
    elif case == 'mismatch': proposal['action_type'] = 'CREATE_BUSINESS_ALERT'
    elif case == 'target': proposal['arguments']['customer_id'] = 'no-entity'
    elif case == 'policy_required': proposal.update(policy_evidence_ids=[], policy_basis=True)
    original = deepcopy(proposal)
    with pytest.raises((ValidationError, ActionError, GatewayError)) as caught:
        validate_proposal(proposal, state, service.registry)
    assert proposal == original
    detail = caught.value.action_proposal_diagnostic
    assert detail['field_path'] == path and detail['validation_rule'] == rule
    assert detail['reason'] and detail['expected_type'] and detail['actual_type']
    assert ActionProposalValidationDiagnostic.model_validate(detail)
    trace = TraceCollector()
    with trace_scope(trace): validation_error('propose_action', caught.value, 'ACTION_PROPOSAL_VALIDATION_ERROR')
    metadata = trace.to_dict()['events'][-1]['safe_metadata']
    for key in ('field_path','reason','validation_rule','expected_type','actual_type','safe_actual_shape','validation_source','evidence_related'):
        assert metadata[key] == detail[key]
    text = trace.to_json()
    assert sentinel not in text
    assert proposal['reason'] not in text and proposal['expected_outcome'] not in text
    assert json.loads(text)


@pytest.mark.parametrize('tool,rule', [('query_orders','INVALID_TOOL_PERMISSION'),('nonexistent','UNKNOWN_WRITE_TOOL')])
def test_write_registry_metadata_diagnostics(action_trace, tool, rule):
    _, _, service, _, _ = action_trace
    with pytest.raises(ActionError) as caught: validate_tool(service.registry, tool, {})
    assert caught.value.code == 'ACTION_PROPOSAL_VALIDATION_ERROR'
    assert caught.value.action_proposal_diagnostic['validation_rule'] == rule


@pytest.mark.parametrize('native', [False, True, 'raised_validation'])
def test_node_and_provider_preserve_diagnostic_no_retry(action_trace, native):
    state, _, service, _, client = action_trace
    state = deepcopy(state); state.update(status='DRAFT_READY', pending_action=None)
    decision = proposal_for(client.payloads[-1][1]); decision['proposal'].pop('tool_name')
    if native is True:
        from langchain_core.messages import AIMessage
        class Client:
            def bind_tools(self, *a, **kw): return self
            def invoke(self, *a, **kw): return AIMessage(content='',tool_calls=[dict(name='ActionDecision',args=decision,id='test')])
        gateway = ProductionGateway(); gateway._model = Client()
    else:
        class Gateway:
            def propose_action(self, context):
                if native == 'raised_validation':
                    from backend.app.actions.schemas import ActionDecision
                    ActionDecision.model_validate(decision)
                return decision
        gateway = Gateway()
    collector = TraceCollector(state['run_id'])
    node = guarded_node('propose_action', make_propose_action_node(GuardedGateway(gateway),service,lambda *a:pytest.fail('No token issued')))
    with trace_scope(collector): result = node(state)
    assert result['status'] == 'ERROR'
    assert result['errors'][-1]['code'] == 'ACTION_PROPOSAL_VALIDATION_ERROR'
    assert result['llm_call_count'] == state['llm_call_count']+1
    assert result['retry_count'] == state['retry_count']
    failures = [e for e in collector.to_dict()['events'] if e['event_type']=='VALIDATION_FAILED']
    assert failures[-1]['safe_metadata']['field_path'] == 'proposal.tool_name'


def test_shape_never_retains_scalar_or_unknown_key():
    secret = 'sk-test-secret-approval-token-value'
    data = {'tool_name': secret, 'arguments': {secret: secret, 'customer_id': secret}, 'business_evidence_ids':[secret]}
    shape = safe_shape(data)
    assert secret not in json.dumps(shape)
    assert validate_shape(shape)
    with pytest.raises(ValueError): validate_shape({'type':'string','value':secret})


def test_unknown_nested_field_is_hidden():
    from backend.app.actions.diagnostics import pydantic_diagnostic
    from backend.app.actions.schemas import ActionDecision
    raw={'action_required':False,'proposal':None,'private-secret-field':'private-secret-value'}
    with pytest.raises(ValidationError) as caught: ActionDecision.model_validate(raw)
    detail=pydantic_diagnostic(caught.value,raw,schema=ActionDecision)
    assert detail['field_path']=='<extra_field>'
    assert 'private-secret' not in json.dumps(detail)
