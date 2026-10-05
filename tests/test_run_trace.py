"""Task09: trace the real boundaries with offline transports and real local READ/WRITE."""
import asyncio
from copy import deepcopy
import hashlib
import json
import socket
from time import perf_counter
from uuid import uuid4

import pytest
from backend.app.actions.schemas import ActionError
from backend.app.actions.service import ApprovalService
from backend.app.agent.graph import build_agent_graph, build_investigation_graph
from backend.app.agent.guards.limits import RuntimeLimits
from backend.app.agent.state import initial_state
from backend.app.data.database import session_factory
from backend.app.llm.gateway import ErrorCode, GatewayError
from backend.app.llm.provider import ProductionGateway
from backend.app.observability.collector import TraceCollector, active_trace, observe, trace_scope
from backend.app.rag.retriever import PolicyRetriever
from backend.app.tools.registry import build_default_registry
from tests.action_fakes import ActionFake
from tests.agent_fakes import FakeGateway, QUERY, goal, plan, selection
from tests.test_context_management import StressClient
from tests.test_rag_retrieval import rag_index
from tests.test_runtime_guards import connection

_connect = socket.socket.connect


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def connect(sock, address):
        # Windows asyncio creates a loopback socket pair for its wakeup pipe.
        if address[0] in {'127.0.0.1', '::1'}:
            return _connect(sock, address)
        raise AssertionError('Trace tests cannot contact external services')
    monkeypatch.setattr(socket.socket, 'connect', connect)


def events(trace, kind):
    return [e for e in trace.to_dict()['events'] if e['event_type'] == kind]


@pytest.fixture
def action_trace(seeded_engine, rag_index):
    embedding, storage, _ = rag_index
    registry = build_default_registry(session_factory(seeded_engine), policy_retriever=PolicyRetriever(embedding, storage=storage))
    service = ApprovalService(registry, session_factory(seeded_engine), trace_approval_source='TEST_HARNESS')
    tokens = {}
    client = StressClient()
    gateway = ProductionGateway(); gateway._model = client
    graph = build_investigation_graph(gateway, registry, action_service=service,
        approval_token_sink=lambda aid, token: tokens.update({aid: token}), sleeper=lambda _: None)
    state = graph.invoke(initial_state(QUERY))
    assert state['status'] == 'AWAITING_APPROVAL', state['errors']
    yield state, graph.last_trace, service, tokens, client
    registry.close()


def test_offline_full_trace_e2e(action_trace, tmp_path):
    state, trace, service, tokens, client = action_trace
    before = trace.to_dict()
    assert before['final_status'] == 'AWAITING_APPROVAL'
    assert before['summary']['read_tool_calls'] == 2
    assert before['summary']['write_tool_calls'] == 0
    assert before['summary']['business_evidence_count'] > 0
    assert before['summary']['policy_evidence_count'] > 0
    assert events(trace, 'PLAN_UPDATED') and events(trace, 'EVIDENCE_SELECTED')
    assert events(trace, 'SYNTHESIS_COMPLETED') and events(trace, 'ACTION_PROPOSED')
    assert events(trace, 'APPROVAL_PENDING')
    assert len(events(trace, 'NODE_STARTED')) == len(events(trace, 'NODE_FINISHED'))
    assert [e['node'] for e in events(trace, 'NODE_STARTED')] == [e['node'] for e in events(trace, 'NODE_FINISHED')]
    assert all(e['duration_ms'] >= 0 for e in events(trace, 'NODE_FINISHED'))
    contexts = {e['safe_metadata']['context_id'] for e in events(trace, 'CONTEXT_BUILT')}
    requests = events(trace, 'LLM_REQUEST_STARTED')
    assert {e['safe_metadata']['context_id'] for e in requests} == contexts
    assert len(requests) == state['llm_call_count'] == 8
    assert before['llm_totals']['total_tokens'] == 960
    assert all(e['safe_metadata']['latency_ms'] >= 0 for e in events(trace, 'TOOL_SUCCEEDED'))
    aid = state['pending_action']['action_id']
    result = service.approve(aid, tokens[aid])
    assert result['success']
    after = trace.to_dict()
    assert after['summary']['write_tool_calls'] == 1
    assert after['summary']['write_executed'] is True
    assert after['final_status'] == 'ACTION_EXECUTED'
    write = events(trace, 'ACTION_EXECUTED')
    assert len(write) == 1
    for key, value in dict(approval_existed=True, approval_source='TEST_HARNESS',
        permission='WRITE_APPROVAL_REQUIRED', fingerprint_validated=True, write_attempts=1).items():
        assert write[0]['safe_metadata'][key] == value
    assert len(events(trace, 'ACTION_APPROVED')) == 1
    with pytest.raises(ActionError): service.approve(aid, tokens[aid])
    assert len(events(trace, 'ACTION_EXECUTED')) == 1
    assert trace.safe_summary()['write_tool_calls'] == 1
    assert len(events(trace, 'RUN_STARTED')) == len(events(trace, 'RUN_FINISHED')) == 1
    sequence = [e['sequence'] for e in trace.to_dict()['events']]
    assert sequence == sorted(set(sequence))
    assert not after['trace_truncated'] and after['observability_errors'] == 0
    exported = trace.to_json()
    raw_token = tokens[aid].get_secret_value()
    assert raw_token not in exported and hashlib.sha256(raw_token.encode()).hexdigest() not in exported
    assert QUERY not in exported
    assert not any('trace' in key for key in state)
    assert all('RUN_STARTED' not in json.dumps(payload) for _, payload, _ in client.payloads)
    (tmp_path / 'trace.json').write_text(exported, encoding='utf-8')
    assert json.loads(exported) == trace.to_dict()


def test_failure_trace_e2e():
    graph = build_agent_graph(FakeGateway(connection()), sleeper=lambda _: None)
    state = graph.invoke(initial_state(QUERY))
    trace = graph.last_trace
    assert state['stop_reason'] == 'MODEL_RETRY_EXHAUSTED'
    assert len(events(trace, 'LLM_REQUEST_STARTED')) == len(events(trace, 'LLM_REQUEST_FAILED')) == 3
    assert len(events(trace, 'LLM_RETRY')) == 2
    assert [e['safe_metadata']['delay_ms'] for e in events(trace, 'LLM_RETRY')] == [500, 1000]
    assert all(e['safe_metadata']['error_code'] == 'PROVIDER_CONNECTION_ERROR' for e in events(trace, 'LLM_REQUEST_FAILED'))
    assert all(e['safe_metadata']['cause_class'] == 'RemoteProtocolError' for e in events(trace, 'LLM_REQUEST_FAILED'))
    assert trace.to_dict()['stop_reason'] == state['stop_reason']
    assert trace.safe_summary()['write_tool_calls'] == 0
    assert trace.safe_summary()['usage_complete'] is False
    assert events(trace, 'GUARD_STOP')
    assert 'secret sentinel' not in trace.to_json()


def test_validation_trace_does_not_copy_rejected_response():
    graph = build_agent_graph(FakeGateway({'secret': 'rejected-response-sentinel'}))
    state = graph.invoke(initial_state(QUERY))
    assert state['status'] == 'ERROR'
    trace = graph.last_trace
    assert events(trace, 'VALIDATION_FAILED')[0]['safe_metadata']['rule_id']
    assert 'rejected-response-sentinel' not in trace.to_json()


def test_context_rejection_no_provider_or_retry():
    fake = FakeGateway(goal(), plan(), selection())
    graph = build_agent_graph(fake)
    state = graph.invoke(initial_state('x' * 9000))
    trace = graph.last_trace
    assert state['llm_call_count'] == 0 and not fake.calls
    assert events(trace, 'CONTEXT_REJECTED')
    assert not events(trace, 'LLM_REQUEST_STARTED') and not events(trace, 'LLM_RETRY')


def test_guard_before_node_execution_still_paired():
    graph = build_agent_graph(FakeGateway(goal()), runtime_limits=RuntimeLimits(max_agent_steps=0))
    state = graph.invoke(initial_state(QUERY))
    trace = graph.last_trace
    assert state['stop_reason'] == 'MAX_AGENT_STEPS_EXCEEDED'
    assert len(events(trace, 'NODE_STARTED')) == len(events(trace, 'NODE_FINISHED')) == 1
    assert events(trace, 'GUARD_STOP')


def test_reject_does_not_write(action_trace):
    state, trace, service, tokens, _ = action_trace
    aid = state['pending_action']['action_id']
    service.reject(aid, tokens[aid])
    assert events(trace, 'ACTION_REJECTED')
    assert trace.safe_summary()['write_tool_calls'] == 0
    assert not trace.safe_summary()['write_executed']
    assert trace.to_dict()['final_status'] == 'ACTION_REJECTED'


def test_unauthorized_write_trace(action_trace):
    state, trace, service, _, _ = action_trace
    aid = state['pending_action']['action_id']
    with pytest.raises(ActionError): service.execute_approved_action(aid)
    with trace_scope(trace):
        result = service.registry.invoke('create_crm_task', {})
    assert not result.success
    assert len(events(trace, 'WRITE_BLOCKED')) == 2
    assert trace.safe_summary()['write_tool_calls'] == 0
    assert not trace.safe_summary()['write_executed']


@pytest.mark.parametrize('field', ['prompt', 'response', 'Authorization', 'approval_token', 'token_hash', 'LLM_API_KEY', 'context', 'query'])
def test_metadata_allowlist_excludes_bodies_and_secrets(field):
    trace = TraceCollector()
    trace.emit('LLM_REQUEST_STARTED', **{field: 'private-sentinel'}, arguments={field: 'private-sentinel'})
    assert 'private-sentinel' not in trace.to_json()


def test_known_secret_cannot_hide_in_allowed_fields():
    trace = TraceCollector()
    secret = 'private-sentinel-credential'
    trace.protect(secret)
    trace.emit('CONTEXT_BUILT', model=secret, evidence_id=secret, fingerprint=hashlib.sha256(secret.encode()).hexdigest())
    assert secret not in trace.to_json()
    assert hashlib.sha256(secret.encode()).hexdigest() not in trace.to_json()


def test_export_failure_is_closed(monkeypatch):
    trace = TraceCollector()
    monkeypatch.setattr(trace, '_metadata', lambda _: (_ for _ in ()).throw(ValueError('bad serializer')))
    with pytest.raises(ValueError): trace.to_json()


def test_collector_failure_does_not_change_graph(monkeypatch):
    state = initial_state(QUERY)
    normal = build_agent_graph(FakeGateway(goal(), plan(), selection())).invoke(deepcopy(state))
    trace = TraceCollector(state['run_id'])
    monkeypatch.setattr(trace, 'emit', lambda *a, **k: (_ for _ in ()).throw(ValueError('trace failed')))
    traced = build_agent_graph(FakeGateway(goal(), plan(), selection()), trace_collector=trace).invoke(deepcopy(state))
    for key in ('status', 'selected_tool', 'plan', 'goal', 'llm_call_count', 'tool_call_count', 'retry_count', 'stop_reason'):
        assert traced[key] == normal[key]
    assert trace.trace.observability_errors > 0


def test_trace_capacity_preserves_lifecycle_and_exact_totals():
    trace = TraceCollector(max_events=12)
    for _ in range(30):
        trace.emit('LLM_REQUEST_STARTED', node='plan')
        trace.emit('LLM_REQUEST_SUCCEEDED', node='plan', usage_available=True, input_tokens=3, output_tokens=2, total_tokens=5)
        trace.emit('EVIDENCE_REGISTERED', evidence_id='order:test')
    trace.finish('DRAFT_READY', 'COMPLETED')
    data = trace.to_dict()
    assert len(data['events']) <= 12
    assert data['trace_truncated'] and data['dropped_events'] > 0
    assert data['summary']['llm_attempts'] == 30 and data['summary']['total_tokens'] == 150
    assert events(trace, 'RUN_STARTED') and events(trace, 'RUN_FINISHED')


@pytest.mark.parametrize('mode', ['invoke', 'stream', 'ainvoke', 'astream'])
def test_graph_entrypoints_and_no_context_leak(mode):
    graph = build_agent_graph(FakeGateway(goal(), plan(), selection()))
    state = initial_state(QUERY)
    if mode == 'invoke': graph.invoke(state)
    elif mode == 'stream':
        for _ in graph.stream(state): assert active_trace.get() is None
    elif mode == 'ainvoke': asyncio.run(graph.ainvoke(state))
    else:
        async def consume():
            async for _ in graph.astream(state): assert active_trace.get() is None
        asyncio.run(consume())
    assert active_trace.get() is None
    assert graph.last_trace.to_dict()['final_status'] == 'TOOL_SELECTED'
    assert len(events(graph.last_trace, 'NODE_FINISHED')) == 3


def test_run_trace_isolation():
    graph = build_agent_graph(FakeGateway(goal(), plan(), selection()))
    graph.invoke(initial_state(QUERY)); first = graph.last_trace
    graph.invoke(initial_state(QUERY)); second = graph.last_trace
    assert first is not second and first.trace.run_id != second.trace.run_id
    assert first.safe_summary()['llm_attempts'] == second.safe_summary()['llm_attempts'] == 3


def test_tool_timeout_is_correlated(monkeypatch, seeded_engine):
    from backend.app.agent.guards import runtime
    from backend.app.agent.guards.context import RuntimeStop
    from backend.app.agent.guards.limits import StopReason
    monkeypatch.setattr(runtime, 'soft_call', lambda *a: (_ for _ in ()).throw(RuntimeStop(StopReason.TOOL_TIMEOUT)))
    with build_default_registry(session_factory(seeded_engine)) as registry:
        graph = build_investigation_graph(ActionFake(), registry)
        state = graph.invoke(initial_state(QUERY))
    assert state['stop_reason'] == 'TOOL_TIMEOUT'
    trace = graph.last_trace
    assert events(trace, 'TOOL_TIMEOUT')[0]['safe_metadata']['tool_call_id'] == events(trace, 'TOOL_STARTED')[0]['safe_metadata']['tool_call_id']


def test_utc_timestamps_and_bounded_cost():
    trace = TraceCollector()
    start = perf_counter()
    for _ in range(500): observe('emit', 'LLM_REQUEST_STARTED', collector=trace, node='plan')
    trace.finish('DRAFT_READY', 'COMPLETED')
    assert perf_counter() - start < 5
    data = trace.to_dict()
    assert all(e['timestamp'].endswith('+00:00') for e in data['events'])
    assert len(trace.to_json().encode()) < 500000


def test_native_retry_context_usage_correlation():
    class Flaky(StressClient):
        attempts = 0
        def invoke(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts <= 2:
                raise connection()
            return super().invoke(*args, **kwargs)
    gateway = ProductionGateway(); gateway._model = Flaky()
    graph = build_agent_graph(gateway, sleeper=lambda _: None)
    state = graph.invoke(initial_state(QUERY))
    trace = graph.last_trace
    assert state['llm_call_count'] == trace.safe_summary()['llm_attempts'] == 5
    assert trace.safe_summary()['total_tokens'] == 360
    manifests = events(trace, 'CONTEXT_BUILT')
    assert len({e['safe_metadata']['context_id'] for e in manifests}) == 5
    assert len({e['safe_metadata']['fingerprint'] for e in manifests[:3]}) == 1
    for kind in ('LLM_REQUEST_SUCCEEDED', 'LLM_REQUEST_FAILED'):
        assert all(e['safe_metadata']['context_id'] in {m['safe_metadata']['context_id'] for m in manifests} for e in events(trace, kind))


def test_explicit_collector_matches_run_and_not_reused():
    collector = TraceCollector()
    graph = build_agent_graph(FakeGateway(goal(), plan(), selection()), trace_collector=collector)
    first = initial_state(QUERY)
    graph.invoke(first)
    assert graph.last_trace.trace.run_id == first['run_id']
    graph.invoke(initial_state(QUERY))
    assert graph.last_trace is not collector


@pytest.mark.parametrize('mode', ['invoke', 'stream'])
def test_unexpected_error_finishes_trace_without_swallowing(mode):
    graph = build_agent_graph(FakeGateway(ValueError('private-bug-sentinel')), raise_unexpected_errors=True)
    with pytest.raises(ValueError):
        if mode == 'invoke': graph.invoke(initial_state(QUERY))
        else: list(graph.stream(initial_state(QUERY)))
    trace = graph.last_trace
    assert trace.to_dict()['final_status'] == 'ERROR'
    assert events(trace, 'NODE_FINISHED') and events(trace, 'RUN_FINISHED')
    assert 'private-bug-sentinel' not in trace.to_json()


def test_tool_retry_counts_physical_calls():
    from tests.test_runtime_guards import runtime, Registry
    rt, _ = runtime()
    trace = TraceCollector(rt.state['run_id'])
    with trace_scope(trace):
        rt.tool(Registry(errors=1), 'get_sales_overview', {})
    assert len(events(trace, 'TOOL_STARTED')) == 2
    assert len(events(trace, 'TOOL_RETRY')) == 1
    assert len(events(trace, 'TOOL_FAILED')) == len(events(trace, 'TOOL_SUCCEEDED')) == 1


def test_write_failure_is_observed_without_retry(action_trace, monkeypatch):
    from backend.app.actions import service as module
    state, trace, service, tokens, _ = action_trace
    def fail(*args): raise RuntimeError('private-write-failure')
    monkeypatch.setattr(module, 'execute_write', fail)
    aid = state['pending_action']['action_id']
    with pytest.raises(ActionError): service.approve(aid, tokens[aid])
    assert trace.safe_summary()['write_tool_calls'] == 1
    assert trace.safe_summary()['write_executed'] is False
    assert events(trace, 'ACTION_FAILED')
    assert events(trace, 'TOOL_FAILED')
    assert 'private-write-failure' not in trace.to_json()


def test_wrapped_action_validation_preserves_safe_cause_code():
    from backend.app.observability.integration import validation_error
    collector = TraceCollector()
    try:
        try:
            raise ActionError('ACTION_PROPOSAL_VALIDATION_ERROR')
        except ActionError:
            raise GatewayError(ErrorCode.ACTION_PROPOSAL_VALIDATION_ERROR) from None
    except GatewayError as error:
        with trace_scope(collector): validation_error('propose_action', error, error.code.value)
    metadata = events(collector, 'VALIDATION_FAILED')[0]['safe_metadata']
    assert metadata['cause_error_code'] == 'ACTION_PROPOSAL_VALIDATION_ERROR'
    assert metadata['rule_id'] == 'ACTION_PROPOSAL_VALIDATION_ERROR'
    assert metadata['field_path'] == 'NOT_PROVIDED'
    assert metadata['reason'] == 'CONTRACT_REJECTED_DETAILS_NOT_PROVIDED'
