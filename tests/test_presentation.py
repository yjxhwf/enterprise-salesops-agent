"""Presentation wiring only; real frozen Graph with offline transport and local DB."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.app.api.presentation import PresentationService
from backend.app.actions.service import ApprovalService
from backend.app.config import Settings
from backend.app.data.database import session_factory
from backend.app.data.validation import dataset_hashes
from backend.app.main import create_app
from backend.app.rag.retriever import PolicyRetriever
from backend.app.tools.registry import build_default_registry
from tests.action_fakes import ActionFake
from tests.agent_fakes import FakeGateway, goal
from tests.test_rag_retrieval import rag_index


@pytest.fixture
def ui(seeded_engine, rag_index):
    embedding,storage,_ = rag_index
    settings=Settings(_env_file=None, LLM_API_KEY='presentation-test-key', EMBEDDING_API_KEY='presentation-embedding-key')
    factory=session_factory(seeded_engine)
    @contextmanager
    def registry():
        with build_default_registry(factory, policy_retriever=PolicyRetriever(embedding,storage=storage)) as r: yield r
    with registry() as r:
        approvals=ApprovalService(r,factory)
        svc=PresentationService(settings,approvals,gateway_factory=lambda:ActionFake(policy=True),registry_factory=registry)
        with factory() as s:before=dataset_hashes(s)
        with TestClient(create_app(settings,approval_service=approvals,presentation_service=svc)) as client:
            yield client,svc
        with factory() as s:assert dataset_hashes(s)==before


def test_stream_real_graph_safe_pending_no_write(ui):
    client,svc=ui
    r=client.post('/api/agent/run',json={'query':'分析2026年8月客户风险，结合政策提出待审批动作。'},headers={'Origin':'http://localhost:3000'})
    assert r.status_code==200 and r.headers['cache-control']=='no-store'
    rows=[json.loads(line) for line in r.text.splitlines()]
    assert rows[-1]['type']=='result'
    result=rows[-1]['result']
    assert result['status']=='AWAITING_APPROVAL'
    assert result['findings'] and result['business_evidence'] and result['policy_evidence']
    assert result['pending_action']['approval_status']=='PENDING'
    assert any(e.get('stage')=='Policy Retrieval' for e in rows)
    assert svc._tokens
    for token in svc._tokens.values():
        raw=token.get_secret_value()
        assert raw not in r.text and hashlib.sha256(raw.encode()).hexdigest() not in r.text
    for word in ('approval_token','token_hash','action_fingerprint','context_manifests','presentation-test-key','presentation-embedding-key','user_query'):
        assert word not in r.text
    assert not svc.run_lock.locked()


@pytest.mark.parametrize('body',[{}, {'query':''},{'query':' '},{'query':'x'*4001},{'query':'q','approval_token':'private'}])
def test_validation_no_input_echo(ui,body):
    r=ui[0].post('/api/agent/run',json=body)
    assert r.status_code==422 and r.json()=={'error':'INVALID_RUN_REQUEST'}


def test_explicit_cors_and_origin_no_dispatch(ui):
    client,svc=ui
    r=client.options('/api/agent/run',headers={'Origin':'http://localhost:3000','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
    assert r.status_code==200 and r.headers['access-control-allow-origin']=='http://localhost:3000'
    r=client.post('/api/agent/run',json={'query':'q'},headers={'Origin':'https://untrusted.invalid'})
    assert r.status_code==403 and not svc._tokens


def test_snapshot_uses_real_database(ui):
    r=ui[0].get('/api/business')
    assert r.status_code==200 and r.headers['cache-control']=='no-store'
    data=r.json()
    assert data['overview']['current']['revenue']=='3150655.70'
    assert data['overview']['current']['order_count']==667
    assert data['regions']['regions'] and data['products']['products']


def test_model_failure_is_not_presented_as_success(ui):
    client,svc=ui
    svc.gateway_factory=lambda:FakeGateway(ValueError('private provider body'))
    rows=[json.loads(l) for l in client.post('/api/agent/run',json={'query':'q'}).text.splitlines()]
    assert rows[-1]['result']['status']=='ERROR'
    assert not rows[-1]['result']['findings']
    assert 'private provider body' not in json.dumps(rows)


def test_busy_and_stream_exception_release_lock(ui):
    client,svc=ui
    svc.run_lock.acquire()
    try:assert client.post('/api/agent/run',json={'query':'q'}).status_code==409
    finally:svc.run_lock.release()
    def broken(_): raise RuntimeError('private fault')
    svc.investigate=broken
    r=client.post('/api/agent/run',json={'query':'q'})
    assert json.loads(r.text)=={'type':'error','code':'RUN_UNAVAILABLE'}
    assert not svc.run_lock.locked()


def test_clarification_and_snapshot_failure(ui):
    client,svc=ui
    svc.gateway_factory=lambda:FakeGateway(goal(start_date=None,end_date=None,needs_clarification=True,clarification_question='请提供日期'))
    rows=[json.loads(l) for l in client.post('/api/agent/run',json={'query':'分析销售'}).text.splitlines()]
    assert rows[-1]['result']['status']=='NEEDS_CLARIFICATION'
    assert rows[-1]['result']['clarification']=='请提供日期'
    svc.business=lambda:None
    assert client.get('/api/business').status_code==503


def test_static_benchmark_is_presentation_only():
    text=Path('frontend/app/eval/page.tsx').read_text(encoding='utf-8')
    assert 'Small Live Selector Benchmark' in text and 'N4' in text and 'L3' in text and 'L4' in text
    assert 'tests/fixtures' not in text and 'fetch(' not in text
