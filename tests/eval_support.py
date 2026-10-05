"""Test-only execution scenarios, independent of cases.json expected values.

Scripted responses test frozen application contracts, NOT model intelligence.
Only this module creates fake transports, fault injections and disposable databases.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import hashlib
import shutil
import socket
import tempfile
from unittest.mock import patch

import httpx
from openai import APIConnectionError
from sqlalchemy import delete, select, func
from sqlalchemy.exc import OperationalError
from backend.app.actions.schemas import ActionError
from backend.app.actions.store import ApprovalStore
from backend.app.actions.service import ApprovalService
from backend.app.agent.evidence import extract_evidence, validate_citations
from backend.app.agent.facts import canonical_facts
from backend.app.agent.graph import build_agent_graph, build_investigation_graph
from backend.app.agent.guards.context import RuntimeStop, active_runtime
from backend.app.agent.guards.limits import RuntimeLimits
from backend.app.agent.guards.runtime import Runtime, guarded_node
from backend.app.agent.semantics import validate_text
from backend.app.agent.state import initial_state
from backend.app.data.database import make_engine, session_factory
from backend.app.data.models import CRMTask, BusinessAlert
from backend.app.data.seed import seed_database
from backend.app.data.validation import dataset_hashes
from backend.app.llm.gateway import GatewayError
from backend.app.observability.collector import trace_scope
from backend.app.rag.bm25 import LexicalIndex
from backend.app.rag.embeddings import FakeEmbeddingGateway
from backend.app.rag.index import build_index, open_index
from backend.app.rag.retriever import PolicyRetriever, MAX_VECTOR_DISTANCE
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import FakeGateway, FakeInvestigationGateway, goal, plan, selection
from tests.action_fakes import ActionFake


@contextmanager
def no_network():
    def denied(*args, **kwargs):
        raise RuntimeError('OFFLINE_NETWORK_FORBIDDEN')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied), \
         patch.object(socket, 'create_connection', denied), patch.object(socket, 'getaddrinfo', denied):
        yield


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def rejected(function):
    try: function()
    except GatewayError: return True
    return False


def accepted(function):
    try: function()
    except GatewayError: return False
    return True


def draft_from_selected(**context):
    manifest = active_runtime.get().current_manifest
    visible = set(manifest['included_business_evidence_ids'])
    ids = list(dict.fromkeys(eid for o in context['observations'] for eid in o['evidence_ids'] if eid in visible))
    return dict(executive_interpretation='Evidence supports further investigation.',
        findings=[dict(title='Measured business observations', claim_type='OBSERVATION',
            interpretation='Measurements describe the specified period and scope.', supporting_evidence_ids=ids)],
        recommendations=[dict(title='Review evidence', action='Review the underlying business measurements.', related_finding_index=0)],
        limitations=['Scripted contract regression, not live reasoning evaluation.'])


class OfflineExecutor:
    def __enter__(self):
        self.block = no_network(); self.block.__enter__()
        self.temp = tempfile.TemporaryDirectory(prefix='salesops-eval-')
        self.root = Path(self.temp.name)
        self.engine = make_engine('sqlite:///' + (self.root / 'eval.db').as_posix())
        seed_database(self.engine)
        self.factory = session_factory(self.engine)
        self.embedding = FakeEmbeddingGateway()
        self.storage = self.root / 'chroma'
        build_index(self.embedding, storage=self.storage)
        self.retriever = PolicyRetriever(self.embedding, storage=self.storage)
        return self

    def __exit__(self, *args):
        try:
            self.engine.dispose()
            from chromadb.api.shared_system_client import SharedSystemClient
            system = SharedSystemClient._identifier_to_system.pop(str(self.storage), None)
            if system: system.stop()
            self.temp.cleanup()
        finally:
            self.block.__exit__(*args)

    def hashes(self):
        with self.factory() as s: return dataset_hashes(s)

    def rows(self):
        with self.factory() as s:
            return sum(s.scalar(select(func.count()).select_from(m)) for m in (CRMTask, BusinessAlert))

    def __call__(self, request):
        # A new approval store per case; reset only the owned temporary DB's writes.
        with self.factory() as s:
            s.execute(delete(CRMTask)); s.execute(delete(BusinessAlert)); s.commit()
        before = self.hashes()
        if request.case_id.startswith('R'): result = self.rag(request)
        elif request.case_id.startswith('P') or request.case_id in ('A1', 'A2'): result = self.permission(request)
        else: result = self.agent(request)
        after = self.hashes()
        changed = [key for key in before if before[key] != after[key]]
        result['writes'] = self.rows()
        if request.case_id == 'A5': result['unauthorized_successes'] = result['writes']
        result.setdefault('checks', {})['business_tables_unchanged'] = not set(changed) - {'crm_tasks', 'business_alerts'}
        if not result['writes']: result['checks']['all_tables_unchanged'] = not changed
        return result

    def normal_gateway(self, case_id):
        name = {'N1':'get_sales_overview', 'N2':'analyze_region_performance'}.get(case_id, 'analyze_product_performance')
        args = {'region':'EAST_CHINA'} if case_id == 'N3' else {}
        subject = {'N3':'SKU-A12', 'N4':'SKU-B07', 'N5':'SKU-C03'}.get(case_id)
        def review(**context):
            catalog = extract_evidence(context['history'])
            ids = [eid for eid, e in catalog.items() if not subject or e['fact_key'] == subject]
            return dict(decision='COMPLETE', selected_evidence_ids=ids, remaining_evidence_gap=None, next_objective=None, updated_plan=None)
        return FakeInvestigationGateway(goal(), plan(), [selection(name, **args)], [review], draft_from_selected)

    def agent(self, request):
        cid = request.case_id
        registry = build_default_registry(self.factory)
        try:
            if cid.startswith('N') or cid == 'A4':
                fake = self.normal_gateway(cid if cid != 'A4' else 'N1')
            elif cid == 'F1':
                fake = FakeGateway(goal(start_date=None, end_date=None, needs_clarification=True, clarification_question='请提供起止日期。'))
            elif cid == 'F2':
                fake = FakeGateway(goal(supported=False, task_type=None, start_date=None, end_date=None))
            elif cid == 'F3':
                fake = FakeGateway(APIConnectionError(request=httpx.Request('POST', 'https://provider.invalid')))
            elif cid == 'F4': fake = FakeGateway(goal())
            elif cid == 'F5':
                fake = self.normal_gateway('N1')
                registry.close()
                def fail(): raise OperationalError('offline injected failure', {}, Exception('fixture'))
                registry = build_default_registry(fail)
            elif cid == 'A3':
                def hidden(**context):
                    visible = active_runtime.get().current_manifest['included_evidence_ids']
                    eid = next(e for e in extract_evidence(context['history']) if e not in visible)
                    return dict(decision='COMPLETE', selected_evidence_ids=[eid], remaining_evidence_gap=None, next_objective=None, updated_plan=None)
                fake = FakeInvestigationGateway(goal(), plan(), [selection('query_orders', limit=50)], [hidden], draft_from_selected)
            elif cid == 'A5': fake = FakeGateway(goal(), plan(), selection('create_crm_task'))
            else: raise ValueError('Unknown scenario')
            planning = cid in ('F1','F2','F3','F4','A5')
            graph = build_agent_graph(fake, sleeper=lambda _: None) if planning else build_investigation_graph(fake, registry, sleeper=lambda _: None)
            state = graph.invoke(initial_state('x' * 9000 if cid == 'F4' else request.query))
            trace = graph.last_trace.to_dict()
            result = dict(state=state, trace=trace)
            if cid == 'F4': result['outcome'] = state['errors'][-1]['code'] if state['errors'] else state['status']
            if cid == 'A3': result['outcome'] = state['errors'][-1]['code'] if state['errors'] else state['status']
            if cid in ('F1','F2','F3','F4','F5'):
                result['runtime_safety'] = dict(bounded=state['llm_call_count'] <= 3 and state['tool_call_count'] <= 2)
            if cid == 'F1': result['checks'] = dict(no_tool_calls=state['tool_call_count'] == 0)
            if cid == 'F3': result['checks'] = dict(three_attempts=state['llm_call_count'] == 3, two_retries=state['retry_count'] == 2)
            if cid == 'F4': result['checks'] = dict(provider_calls_zero=not fake.calls and state['llm_call_count'] == 0)
            if cid == 'F5':
                result['checks'] = dict(two_read_attempts=state['tool_call_count'] == 2)
                result['loops'] = self.loop_probes()
            if cid == 'A5':
                result['checks'] = dict(selector_rejected=state['errors'][-1]['code'] == 'TOOL_SELECTION_VALIDATION_ERROR', no_execution=state['tool_call_count'] == 0)
                result.update(unauthorized_attempts=1, unauthorized_successes=0)
            if cid.startswith('N') or cid == 'A4':
                self.semantic_probes(cid, result)
            return result
        finally: registry.close()

    def semantic_probes(self, cid, result):
        state = result['state']
        catalog = extract_evidence(state['tool_history'])
        subjects = {'N3':'SKU-A12', 'N4':'SKU-B07', 'N5':'SKU-C03'}
        selected = {k:v for k,v in catalog.items() if cid not in subjects or v['fact_key'] == subjects[cid]}
        checks = result.setdefault('checks', {})
        semantic = result.setdefault('semantic', {})
        if cid == 'N1':
            semantic['yoy_misuse'] = rejected(lambda: validate_text('公司收入同比增长', selected))
            semantic['unsupported_evidence'] = rejected(lambda: validate_citations(['fabricated-evidence'], state['tool_history']))
        if cid == 'N2':
            changes = {e['fact_key']:Decimal(e['internal_reference']['raw_value']['current']['profit'])-Decimal(e['internal_reference']['raw_value']['previous']['profit']) for e in catalog.values()}
            checks['largest_regional_profit_decline'] = min(changes, key=changes.get) == 'EAST_CHINA'
        if cid == 'N3':
            semantic['discount_direction_error'] = rejected(lambda: validate_text('SKU-A12 折扣变浅', selected))
            semantic['discount_direction_correct'] = accepted(lambda: validate_text('SKU-A12 优惠加深', selected))
        if cid == 'N4':
            semantic['b07_direct_loss_error'] = rejected(lambda: validate_text('SKU-B07造成23,608利润损失', selected))
            semantic['low_margin_mix_dilution'] = accepted(lambda: validate_text('SKU-B07低毛利销售占比上升形成结构稀释，但不代表该产品本身亏损', selected))
        if cid == 'N5':
            f = next(iter(selected.values()))['internal_reference']['raw_value']
            checks['high_margin_stable'] = abs(Decimal(f['current']['margin'])-Decimal(f['previous']['margin'])) < Decimal('.001')
        if cid == 'A4':
            semantic['negative_profit_false_claim'] = rejected(lambda: validate_text('公司已经亏损', catalog))
            semantic['negation_regression'] = accepted(lambda: validate_text('但不代表该产品本身亏损', catalog))
            result['outcome'] = 'REJECT' if semantic['negative_profit_false_claim'] else 'ACCEPT'

    def loop_probes(self):
        # Auxiliary guard probes belong to F5, not extra cases or model routing samples.
        with build_default_registry(self.factory) as registry:
            rt = Runtime(initial_state('loop probe'), 'execute_tool', RuntimeLimits(), lambda _: None)
            reason = None
            try:
                for _ in range(5): rt.tool(registry, 'get_sales_overview', selection()['arguments'])
            except RuntimeStop as error: reason = error.reason.value
            repeated = reason == 'REPEATED_ACTION_LOOP' and rt.state['tool_call_count'] == 2
        state = initial_state('stagnant probe')
        state.update(status='INVESTIGATING', plan=plan())
        node = guarded_node('review_progress', lambda s: {'status':'INVESTIGATING'})
        for _ in range(5):
            state = node(state)
            if state['status'] == 'RUNTIME_STOPPED': break
        return dict(repeated_identical_tool=repeated, stagnant_cycle=state['stop_reason'] == 'STAGNANT_LOOP' and state['agent_step_count'] <= 5)

    def rag(self, request):
        response = self.retriever.search(request.query, top_k=3)
        manifest, chunks, store = open_index(self.embedding, storage=self.storage)
        catalog = {c.chunk_id:c.doc_id for c in chunks}
        vector = [r for r in store.search(manifest['collection'], self.embedding.embed([request.query])[0]) if r['distance'] <= MAX_VECTOR_DISTANCE]
        bm25 = LexicalIndex(chunks).search(request.query)
        return dict(outcome='RETRIEVED', evidence_kinds=['POLICY'], rankings=dict(
            hybrid=[h.doc_id for h in response.hits[:3]], vector=[catalog[h['chunk_id']] for h in vector[:3]],
            bm25=[catalog[h['chunk_id']] for h in bm25[:3]]))

    def permission(self, request):
        cid = request.case_id
        parent = self.retriever
        class InjectedPolicy:
            def search(self, query, top_k=5):
                response = parent.search(query, top_k)
                response.hits[0].excerpt += '\n' + request.query
                return response
        registry = build_default_registry(self.factory, policy_retriever=InjectedPolicy() if cid == 'A1' else parent)
        now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
        service = ApprovalService(registry, self.factory, store=ApprovalStore(clock=lambda: now[0]), trace_approval_source='TEST_HARNESS')
        tokens = {}
        fake = ActionFake(policy=cid == 'A1')
        graph = build_investigation_graph(fake, registry, action_service=service,
            approval_token_sink=lambda aid, token: tokens.update({aid:token}), sleeper=lambda _: None)
        try:
            state = graph.invoke(initial_state(request.query))
            if state['status'] != 'AWAITING_APPROVAL': return dict(state=state, trace=graph.last_trace.to_dict(), checks={'pending_action_created':False})
            aid = state['pending_action']['action_id']
            trace = graph.last_trace
            before = self.rows()
            code = None
            attempts = 0
            with trace_scope(trace):
                try:
                    if cid == 'P1':
                        attempts = 1
                        response = registry.invoke('create_crm_task', state['pending_action']['arguments'])
                        code = response.error.code if response.error else 'UNAUTHORIZED_EXECUTED'
                    elif cid in ('A1', 'A2'):
                        attempts = 1
                        service.execute_approved_action(aid)
                    elif cid == 'P2':
                        attempts = 1; service.approve(aid, 'wrong-test-token')
                    elif cid == 'P3':
                        attempts = 1; now[0] += timedelta(minutes=30); service.approve(aid, tokens[aid])
                    else:
                        service.approve(aid, tokens[aid])
                        if cid == 'P4':
                            attempts = 1; before = self.rows(); service.approve(aid, tokens[aid])
                except ActionError as error: code = error.code
            delta = self.rows()-before
            if cid in ('P4','P5'): outcome = 'ACTION_EXECUTED' if self.rows() == 1 else 'WRITE_COUNT_ERROR'
            elif cid == 'P3': outcome = code or 'UNAUTHORIZED_EXECUTED'
            else: outcome = 'BLOCKED' if code in ('APPROVAL_REQUIRED','INVALID_APPROVAL_STATE','INVALID_APPROVAL_TOKEN') else code or 'UNAUTHORIZED_EXECUTED'
            checks = dict(pending_before_approval=state['status']=='AWAITING_APPROVAL')
            if cid == 'P4': checks['duplicate_delta_zero'] = delta == 0 and code == 'INVALID_APPROVAL_STATE'
            if cid == 'A1':
                # Demonstrate injection really entered untrusted retrieved content.
                checks['injection_was_retrieved'] = request.query in str(state['tool_history'])
            return dict(state=state, outcome=outcome, trace=trace.to_dict(), checks=checks,
                        unauthorized_attempts=attempts, unauthorized_successes=max(delta, 0) if attempts else 0)
        finally: registry.close()
