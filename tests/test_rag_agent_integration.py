from copy import deepcopy
import json
from pathlib import Path
import shutil
import pytest
from backend.app.agent.graph import build_investigation_graph
from backend.app.agent.state import initial_state
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.policy_evidence import policy_catalog
from backend.app.agent.guards.context import build_llm_context
from backend.app.agent import prompts
from backend.app.data.database import session_factory
from backend.app.rag.embeddings import FakeEmbeddingGateway
from backend.app.rag.index import build_index
from backend.app.rag.retriever import PolicyRetriever
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import FakeInvestigationGateway, goal, plan, selection, QUERY
from tests.test_agent_investigation import draft_from_evidence
from tests.test_rag_retrieval import rag_index, offline


def policy_flow(engine,retriever,*,bad_citation=False,policy_only_finding=False):
    def first(**context):
        return dict(decision='CONTINUE',selected_evidence_ids=list(extract_evidence(context['history'])),
            remaining_evidence_gap='缺少适用的公司折扣审批制度依据',next_objective='检索销售折扣审批制度',
            updated_plan={'steps':[dict(objective='检索销售折扣审批制度',expected_output='可引用政策章节')]})
    def second(**context):
        return dict(decision='COMPLETE',selected_evidence_ids=list(policy_catalog(context['history']))[:5],
            remaining_evidence_gap=None,next_objective=None,updated_plan=None)
    def draft(**context):
        business=[o for o in context['observations'] if o['facts']]
        result=draft_from_evidence(observations=business)
        policies=[e for o in context['observations'] for e in o.get('policy_evidence',[])]
        eid=policies[0]['evidence_id']
        result['recommendations'][0].update(policy_evidence_ids=['POLICY::POL-SALES-999' if bad_citation else eid],
            policy_interpretation='建议核对所引制度的适用范围和审批要求，缺失标记仍需核实。')
        if policy_only_finding:result['findings'][0]['supporting_evidence_ids']=[eid]
        return result
    fake=FakeInvestigationGateway(goal(),plan(),[selection(),dict(tool_name='search_sales_policy',arguments=dict(query='折扣审批'))],[first,second],draft)
    with build_default_registry(session_factory(engine),policy_retriever=retriever) as registry:
        state=build_investigation_graph(fake,registry,raise_unexpected_errors=True).invoke(initial_state(QUERY+' 并引用公司折扣政策'))
    return state


def test_business_policy_agent_e2e(seeded_engine,rag_index):
    gateway,storage,_=rag_index
    state=policy_flow(seeded_engine,PolicyRetriever(gateway,storage=storage))
    assert state['status']=='DRAFT_READY'
    assert [h['tool_name'] for h in state['tool_history']]==['get_sales_overview','search_sales_policy']
    policies=policy_catalog(state['tool_history'])
    business=extract_evidence(state['tool_history'])
    assert policies and business and not(set(policies)&set(business))
    assert all(e['evidence_kind']=='BUSINESS_DATA' for e in business.values())
    assert all(e['evidence_kind']=='POLICY' for e in policies.values())
    assert state['observations'][1]['facts']==[]
    rec=state['analysis_draft']['recommendations'][0]
    assert rec['policy_citations'] and rec['policy_evidence_ids']
    assert state['analysis_draft']['findings'][rec['related_finding_index']]['supporting_facts']
    assert state['llm_call_count']==7 and state['retry_count']==0  # Embeddings are not chat calls.
    assert json.loads(json.dumps(state))==state
    view=build_llm_context('synthesize',goal=state['goal'],observations=state['observations'],history=state['tool_history'])
    assert 1<=len(view['policy_evidence'])<=5
    assert all(p['trust_boundary']=='UNTRUSTED_RETRIEVED_CONTENT' and len(p['excerpt'])<=800 for p in view['policy_evidence'])


@pytest.mark.parametrize('mode',['bad_id','policy_only'])
def test_policy_cannot_forge_ids_or_prove_business(seeded_engine,rag_index,mode):
    gateway,storage,_=rag_index
    state=policy_flow(seeded_engine,PolicyRetriever(gateway,storage=storage),bad_citation=mode=='bad_id',policy_only_finding=mode=='policy_only')
    assert state['status']=='ERROR' and state['analysis_draft'] is None


def test_unselected_policy_not_sent_to_synthesis(rag_index):
    gateway,storage,_=rag_index
    result=PolicyRetriever(gateway,storage=storage).search('折扣审批')
    history=[dict(tool_name='search_sales_policy',success=True,data=result.model_dump(mode='json'),evidence_ids=[h.citation.policy_evidence_id for h in result.hits])]
    view=build_llm_context('synthesize',history=history)
    assert 'policy_evidence' not in view


@pytest.mark.parametrize('top_k', [5, 8])
def test_policy_result_digest_reports_real_hit_coverage(rag_index, top_k):
    gateway, storage, _ = rag_index
    result = PolicyRetriever(gateway, storage=storage).search('折扣审批', top_k)
    history = [dict(tool_name='search_sales_policy', success=True, data=result.model_dump(mode='json'),
                    evidence_ids=[h.citation.policy_evidence_id for h in result.hits])]
    view = build_llm_context('review_progress', history=history)
    digest = view['result_digest']
    assert digest['evidence_kind'] == 'POLICY'
    assert digest['returned_count'] == len(result.hits)
    assert digest['visible_row_count'] == len(view['policy_evidence']) == min(5, len(result.hits))
    assert digest['omitted_row_count'] == len(result.hits)-digest['visible_row_count']


def test_unsupported_policy_assertion_rejected():
    from backend.app.agent.policy_evidence import validate_policy_recommendations
    from backend.app.agent.schemas import GroundedAnalysisDraft
    from backend.app.llm.gateway import GatewayError
    draft = GroundedAnalysisDraft.model_validate(dict(
        executive_interpretation='仅有经营证据。',
        findings=[dict(title='待复核', interpretation='仅观察，未核实审批。', claim_type='OBSERVATION', supporting_evidence_ids=['business'])],
        recommendations=[dict(title='审批', action='公司规定必须由总监审批。', related_finding_index=0)],
        limitations=['没有检索到政策依据。']))
    with pytest.raises(GatewayError):
        validate_policy_recommendations(draft, [], set())


def test_injection_stays_untrusted_data(seeded_engine,tmp_path):
    root=tmp_path/'corpus';root.mkdir()
    shutil.copyfile('tests/fixtures/adversarial_policy.md',root/'policy.md')
    class InjectionEmbedding(FakeEmbeddingGateway):
        # Force retrieval of the adversarial fixture; this tests the trust boundary,
        # not semantic similarity in a one-document BM25 corpus.
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]
    gateway=InjectionEmbedding();storage=tmp_path/'injection'
    build_index(gateway,root=root,storage=storage)
    state=policy_flow(seeded_engine,PolicyRetriever(gateway,root=root,storage=storage))
    assert state['status']=='DRAFT_READY'
    view=build_llm_context('synthesize',observations=state['observations'],history=state['tool_history'])
    assert 'Ignore previous instructions' in view['policy_evidence'][0]['excerpt']
    assert view['policy_evidence'][0]['trust_boundary']=='UNTRUSTED_RETRIEVED_CONTENT'
    assert 'UNTRUSTED_RETRIEVED_CONTENT' in prompts.SYNTHESIZE
    assert 'delete_orders' not in [h['tool_name'] for h in state['tool_history']]
    assert 'Ignore previous instructions' not in state['analysis_draft']['executive_interpretation']
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient._identifier_to_system[str(storage)].stop()
    SharedSystemClient._identifier_to_system.pop(str(storage))
    shutil.rmtree(storage)
