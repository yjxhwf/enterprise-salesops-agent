"""Task08 deterministic stress fixture and request-boundary integration tests."""
from copy import deepcopy
from dataclasses import replace
import json
import socket
import pytest
from langchain_core.messages import AIMessage
from backend.app.actions.service import ApprovalService
from backend.app.actions.validation import validate_proposal
from backend.app.agent.context.assembly import POLICIES, assemble, canonical, char_count, ContextBudgetExceeded, visible_or_reject
from backend.app.agent.context.evidence import EvidenceLedger, selection_rank
from backend.app.agent.guards.context import active_runtime, build_llm_context, project_context, action_context, bounded_facts, compact_facts
from backend.app.agent.graph import build_investigation_graph
from backend.app.agent.state import initial_state
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.facts import canonical_facts
from backend.app.agent.policy_evidence import policy_catalog
from backend.app.agent.guards.limits import RuntimeLimits
from backend.app.data.database import session_factory
from backend.app.llm.provider import ProductionGateway
from backend.app.llm.gateway import GatewayError
from backend.app.rag.retriever import PolicyRetriever
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import DATES, goal, plan, selection
from tests.action_fakes import action_run, proposal_for
from tests.test_rag_retrieval import rag_index
from tests.test_context_projection import ProjectedClient, projected_run
from tests.test_agent_investigation import run, scripted, offline


@pytest.fixture
def stress(seeded_engine, rag_index):
    embedding, storage, _ = rag_index
    retriever = PolicyRetriever(embedding, storage=storage)
    records=[]
    with build_default_registry(session_factory(seeded_engine), policy_retriever=retriever) as registry:
        for name in ('get_sales_overview','analyze_region_performance','analyze_product_performance','analyze_customer_performance','query_orders'):
            args={**DATES, **({'limit':50} if name=='query_orders' else {})}
            result=registry.invoke(name,args).model_dump(mode='json')
            records.append({**result,'arguments':args,'plan_step_id':'fixture'})
        for query in ('折扣审批','低毛利治理','特殊报价','区域促销','订单复核','CRM跟进'):
            args={'query':query,'top_k':5}
            result=registry.invoke('search_sales_policy',args).model_dump(mode='json')
            assert result['success']
            records.append({**result,'arguments':args,'plan_step_id':'fixture'})
    business=extract_evidence(records);policies=policy_catalog(records)
    assert len(business)>70 and len(policies)>10
    state=initial_state('分析销售风险，结合政策提出待审批动作，不直接执行')
    state.update(goal=goal(),plan=plan(),tool_history=records)
    observations=[]
    ids=list(business);pids=list(policies)
    for i in range(6):
        selected=ids[i*15:(i+1)*15]+[ids[0]]
        ps=pids[i*2:(i+1)*2]
        observations.append(dict(observation_id=f'fixture-{i}',completed_step_id=f'fixture-{i}',
            source_tool='query_orders',evidence_ids=list(dict.fromkeys(selected+ps)),
            facts=canonical_facts(business,selected),policy_evidence=[policies[eid] for eid in ps],
            plan_before={'steps':[dict(step_id=f'fixture-{i}',objective=f'Completed objective {i}')]},
            decision='CONTINUE',remaining_evidence_gap='Check policy authorization applicability',next_objective='Check policy',plan_after=None))
    state['observations']=observations
    return state,retriever


class StressClient:
    """Fake native provider sees exactly the production wire payload; no network."""
    def __init__(self):self.payloads=[];self.toolsets=[];self.choice=None
    def bind_tools(self,tools,*,tool_choice,**kwargs):
        self.choice=tool_choice;self.toolsets.append(deepcopy(tools));return self
    def invoke(self,messages,config):
        runtime=active_runtime.get();s=runtime.state
        payload=json.loads(messages[1][1]);self.payloads.append((runtime.node,payload,deepcopy(runtime.current_manifest)))
        if self.choice=='GoalUnderstanding':args=goal()
        elif self.choice=='Plan':args=plan()
        elif self.choice=='required':
            selected=selection('analyze_customer_performance') if s['tool_call_count']==0 else dict(tool_name='search_sales_policy',arguments={'query':'折扣审批'})
            return AIMessage(content='',tool_calls=[dict(name=selected['tool_name'],args=selected['arguments'],id='fake')],usage_metadata=dict(input_tokens=100,output_tokens=20,total_tokens=120))
        elif self.choice=='ProgressReview':
            complete=s['tool_call_count']==2
            entries=payload['policy_evidence'] if complete else payload['latest_evidence']
            args=dict(decision='COMPLETE' if complete else 'CONTINUE',selected_evidence_ids=[entries[0]['evidence_id']],
                remaining_evidence_gap=None if complete else 'Missing applicable policy approval evidence',
                next_objective=None if complete else 'Check discount policy',updated_plan=None if complete else {'steps':[dict(objective='Check discount policy',expected_output='Policy citation')]})
        elif self.choice=='GroundedAnalysisDraft':
            eid=next(e['evidence_id'] for e in payload['selected_evidence'] if e['subject']['type']=='CUSTOMER')
            pid=payload['policy_evidence'][0]['evidence_id']
            args=dict(executive_interpretation='Observed measurements warrant further investigation.',findings=[dict(title='Customer measurement',claim_type='CONTRIBUTING_FACTOR',interpretation='Observed measurements warrant review.',supporting_evidence_ids=[eid])],recommendations=[dict(title='Review',action='Check applicable policy and pricing records.',related_finding_index=0,policy_evidence_ids=[pid],policy_interpretation='Check applicable policy scope.')],limitations=['Offline deterministic fixture.'])
        else:args=proposal_for(payload)
        return AIMessage(content='',tool_calls=[dict(name=self.choice,args=args,id='fake')],usage_metadata=dict(input_tokens=100,output_tokens=20,total_tokens=120))


def stress_run(engine,stress):
    state,retriever=stress
    client=StressClient();gateway=ProductionGateway();gateway._model=client
    registry=build_default_registry(session_factory(engine),policy_retriever=retriever)
    service=ApprovalService(registry,session_factory(engine));tokens={}
    result=build_investigation_graph(gateway,registry,action_service=service,approval_token_sink=lambda aid,t:tokens.update({aid:t})).invoke(deepcopy(state))
    return result,client,tokens


def test_stress_full_native_flow_and_snapshots(seeded_engine,stress):
    original=deepcopy(stress[0]);state,client,tokens=stress_run(seeded_engine,stress)
    assert state['status']=='AWAITING_APPROVAL',state['errors']
    assert stress[0]==original
    assert len(extract_evidence(state['tool_history']))>70
    assert len(policy_catalog(state['tool_history']))>10
    assert len(next(r for r in state['tool_history'] if r['tool_name']=='query_orders')['data']['orders'])==50
    assert len(state['context_manifests'])==state['llm_call_count']==8
    for node,payload,manifest in client.payloads:
        assert manifest['projected_char_count']<=POLICIES[node].char_budget
        assert POLICIES[node].mandatory<=payload.keys()
        bundles=[b for k,v in payload.items() if isinstance(v,list) for b in v if isinstance(b,dict) and 'evidence_id' in b]
        ids=[b['evidence_id'] for b in bundles]
        assert len(ids)==len(set(ids)) and set(ids)==set(manifest['included_evidence_ids'])
        assert len(manifest['included_fact_ids'])==len(set(manifest['included_fact_ids']))
        assert not any(key in canonical(payload) for key in ('tool_history','runtime_events','internal_reference','token_hash','approval_token'))
        assert all(t.get_secret_value() not in canonical(payload) for t in tokens.values())
        if node in ('synthesize','propose_action'):
            assert manifest['included_policy_evidence_ids']
            assert 'UNTRUSTED_RETRIEVED_CONTENT' in canonical(payload)
        if node=='synthesize':
            assert sum(len(b['facts']) for b in payload['selected_evidence'])<=48
            assert payload['executive_facts'] and len(payload['completed_objectives'])>=6
        if node=='propose_action':
            assert len(payload['available_write_tools'])==2
            assert all(t['permission_level']=='WRITE_APPROVAL_REQUIRED' for t in payload['available_write_tools'])
    assert all(len(tools)==6 for tools in client.toolsets if tools[0]['function']['name']=='get_sales_overview')
    assert any(u['selected_count']>=6 for u in state['evidence_usage'].values())
    assert any(u['used_in_synthesis'] for u in state['evidence_usage'].values())
    assert any(u['used_in_action_proposal'] for u in state['evidence_usage'].values())
    events=[e for e in state['runtime_events'] if e['event_type']=='LLM_USAGE']
    assert [e['context_id'] for e in events]==[m['context_id'] for m in state['context_manifests']]
    import os
    from pathlib import Path
    destination=os.environ.get('TASK08_STRESS_REPORT')
    if destination:
        Path(destination).write_text(json.dumps(dict(status=state['status'],business_evidence_count=len(extract_evidence(state['tool_history'])),policy_evidence_count=len(policy_catalog(state['tool_history'])),order_rows=50,full_state_chars=char_count(state),manifests=state['context_manifests'],max_projected_chars=max(m['projected_char_count'] for m in state['context_manifests']),budget_violations=0,duplicate_evidence=0,raw_order_leakage=False),ensure_ascii=False,indent=2),encoding='utf-8')


def test_six_review_growth_dedup_and_fifty_orders(stress):
    state,_=stress;orders=next(r for r in state['tool_history'] if r['tool_name']=='query_orders')
    history=[r for r in state['tool_history'] if r is not orders]+[orders]
    sizes=[]
    for i in range(1,7):
        view=build_llm_context('review_progress',goal=state['goal'],plan=state['plan'],history=history,observations=state['observations'][:i])
        sizes.append(char_count(view))
        assert view['result_digest']['visible_row_count']==10
        assert view['result_digest']['returned_count']==50
        assert len(view['latest_evidence'])==10
        ids=[b['evidence_id'] for v in view.values() if isinstance(v,list) for b in v if isinstance(b,dict) and 'evidence_id' in b]
        assert len(ids)==len(set(ids))
    assert max(sizes)<=48000 and sizes[-1]<sizes[0]*2


def test_manifest_fingerprint_key_order_and_changed_selection(stress):
    state,_=stress
    candidate=project_context('synthesize',goal=state['goal'],history=state['tool_history'],observations=state['observations'])
    a,ma=assemble('synthesize',candidate)
    b,mb=assemble('synthesize',dict(reversed(list(candidate.items()))))
    assert ma['context_fingerprint']==mb['context_fingerprint'] and ma['context_id']!=mb['context_id']
    modified=deepcopy(candidate);modified['selected_evidence']=modified['selected_evidence'][1:]
    assert assemble('synthesize',modified)[1]['context_fingerprint']!=ma['context_fingerprint']


@pytest.mark.parametrize('node',list(POLICIES))
def test_mandatory_overflow_is_safe(node):
    candidate={key:[] if key in ('latest_evidence','executive_facts','available_write_tools') else 'x'*100 for key in POLICIES[node].mandatory}
    if node=="synthesize":candidate["synthesis_coverage"]={"total_selected_fact_count":0}
    with pytest.raises(ContextBudgetExceeded) as exc:assemble(node,candidate,budget=20)
    assert exc.value.code=='CONTEXT_BUDGET_EXCEEDED'
    assert exc.value.context_diagnostic['mandatory_chars']>20
    assert 'x'*100 not in str(exc.value)


@pytest.mark.parametrize('native',[False,True])
def test_preflight_no_provider_no_count_no_retry(seeded_engine,native):
    gateway=ProductionGateway() if native else scripted()
    class Forbidden:
        def bind_tools(self,*a,**k):raise AssertionError('Provider must not be reached')
    if native:gateway._model=Forbidden()
    with build_default_registry(session_factory(seeded_engine)) as registry:
        state=build_investigation_graph(gateway,registry).invoke(initial_state('x'*9000))
    assert state['status']=='ERROR' and state['errors'][-1]['code']=='CONTEXT_BUDGET_EXCEEDED'
    assert state['llm_call_count']==state['retry_count']==0 and not state['context_manifests']


def test_exact_dedup_and_kind_validation(stress):
    state,_=stress;ledger=EvidenceLedger(state)
    eid=next(iter(extract_evidence(state['tool_history'])))
    assert len(ledger.resolve_many([eid,eid],'BUSINESS_DATA'))==1
    with pytest.raises(GatewayError):ledger.resolve(eid,'POLICY')
    with pytest.raises(GatewayError):ledger.resolve('missing')
    with pytest.raises(ValueError):ledger.register('ACTION::test','BUSINESS_DATA',{})
    action_state={**state,'action_result':dict(success=True,data=dict(evidence_kind='ACTION',object_id='test'),evidence_ids=['ACTION::test'])}
    assert EvidenceLedger(action_state).resolve('ACTION::test','ACTION')['object_id']=='test'
    view,manifest=assemble('synthesize',project_context('synthesize',history=state['tool_history'],observations=state['observations']))
    with pytest.raises(GatewayError):visible_or_reject(['ACTION::test'],manifest,kind='BUSINESS_DATA')


def test_visibility_existing_hidden_missing_and_kind(stress):
    state,_=stress
    view,manifest=assemble('synthesize',project_context('synthesize',history=state['tool_history'],observations=state['observations']))
    visible=manifest['included_business_evidence_ids'][0]
    visible_or_reject([visible],manifest,kind='BUSINESS_DATA')
    hidden=next(eid for eid in extract_evidence(state['tool_history']) if eid not in manifest['included_evidence_ids'])
    for eid in (hidden,'missing'):
        with pytest.raises(GatewayError,match='not visible'):visible_or_reject([eid],manifest)
    with pytest.raises(GatewayError):visible_or_reject([manifest['included_policy_evidence_ids'][0]],manifest,kind='BUSINESS_DATA')


def test_usage_priority_not_input_order(stress):
    state,_=stress;facts=canonical_facts(extract_evidence(state['tool_history']))
    eid=sorted({f['evidence_id'] for f in facts})[-1]
    selected=bounded_facts(facts,1,[{'evidence_ids':[eid]},{'evidence_ids':[eid]}])
    assert selected[0]['evidence_id']==eid
    assert selected==bounded_facts(list(reversed(facts)),1,[{'evidence_ids':[eid]},{'evidence_ids':[eid]}])


def test_retry_context_usage_links(seeded_engine):
    state,client,delays=projected_run(seeded_engine,True)
    ms=state['context_manifests'];assert len(ms)==12
    retry=next(m for m in ms if m['attempt']==2)
    previous=ms[ms.index(retry)-1]
    assert retry['context_id']!=previous['context_id'] and retry['context_fingerprint']==previous['context_fingerprint']
    assert state['retry_count']==1 and delays==[.5]


def test_no_action_in_read_only_flow(seeded_engine):
    state=run(scripted(),seeded_engine)
    assert state['status']=='DRAFT_READY'
    assert all(m['node']!='propose_action' for m in state['context_manifests'])
    assert state['pending_action'] is None

@pytest.mark.parametrize('node',['review_progress','synthesize','propose_action'])
def test_actual_output_rejects_real_but_invisible_citation(seeded_engine,stress,monkeypatch,node):
    original=StressClient.invoke
    def hidden(self,messages,config):
        response=original(self,messages,config)
        runtime=active_runtime.get()
        if runtime.node==node:
            catalog=extract_evidence(runtime.state['tool_history'])
            eid=next(e for e in catalog if e not in runtime.current_manifest['included_evidence_ids'])
            args=response.tool_calls[0]['args']
            if node=='review_progress':args['selected_evidence_ids']=[eid]
            elif node=='synthesize':args['findings'][0]['supporting_evidence_ids']=[eid]
            else:args['proposal']['business_evidence_ids']=[eid]
        return response
    monkeypatch.setattr(StressClient,'invoke',hidden)
    state,client,tokens=stress_run(seeded_engine,stress)
    assert state['status']=='ERROR'
    assert state['errors'][-1]['code']=='EVIDENCE_NOT_VISIBLE_IN_CONTEXT'
    assert not tokens and state['pending_action'] is None


def test_optional_trim_preserves_policy_and_complete_items(stress):
    state,_=stress
    candidate=project_context('synthesize',history=state['tool_history'],observations=state['observations'])
    full,manifest=assemble('synthesize',candidate)
    budget=char_count(full)-2000
    trimmed,limited=assemble('synthesize',candidate,budget=budget)
    assert char_count(trimmed)<=budget
    assert trimmed['policy_evidence'] and trimmed['executive_facts']
    assert all(b['trust_boundary']=='UNTRUSTED_RETRIEVED_CONTENT' for b in trimmed['policy_evidence'])
    assert limited['omitted_item_counts']
    assert json.loads(json.dumps(trimmed))==trimmed
    assert set(limited['included_fact_ids'])<set(manifest['included_fact_ids'])


def test_manifest_cannot_be_recomputed_after_model_output(seeded_engine,monkeypatch):
    original=ProjectedClient.invoke
    def tamper(self,messages,config):
        response=original(self,messages,config)
        runtime=active_runtime.get()
        if runtime.node=='synthesize':
            # Removing observations after the request would change a rebuilt view.
            # The already-issued snapshot must remain the validation authority.
            runtime.state['observations']=[]
        return response
    monkeypatch.setattr(ProjectedClient,'invoke',tamper)
    state,_,_=projected_run(seeded_engine)
    assert state['status']=='DRAFT_READY'


def test_usage_missing_not_estimated_from_chars(seeded_engine,monkeypatch):
    original=ProjectedClient.invoke
    def missing(self,messages,config):
        response=original(self,messages,config);response.usage_metadata=None;return response
    monkeypatch.setattr(ProjectedClient,'invoke',missing)
    state,_,_=projected_run(seeded_engine)
    assert state['status']=='DRAFT_READY'
    assert state['token_usage']['usage_available'] is False
    assert state['token_usage']['total_tokens']==0
    usages=[e for e in state['runtime_events'] if e['event_type']=='LLM_USAGE']
    assert all(not e['usage_available'] and e['context_id'] for e in usages)

@pytest.mark.parametrize('node',list(POLICIES))
def test_each_node_budget_stops_before_gateway(seeded_engine,monkeypatch,node):
    monkeypatch.setitem(POLICIES,node,replace(POLICIES[node],char_budget=20))
    state,svc,tokens,fake=action_run(seeded_engine)
    assert state['status']=='ERROR' and state['errors'][-1]['code']=='CONTEXT_BUDGET_EXCEEDED'
    assert state['errors'][-1]['context_diagnostic']['node']==node
    assert state['llm_call_count']==len(state['context_manifests'])==len(fake.calls)
    assert all(m['node']!=node for m in state['context_manifests'])
    assert state['retry_count']==0 and not tokens


def test_action_context_does_not_serialize_state_secrets(seeded_engine):
    state,service,tokens,fake=action_run(seeded_engine)
    poisoned=deepcopy(state)
    for key in ('LLM_API_KEY','EMBEDDING_API_KEY','approval_token','token_hash','Authorization'):
        poisoned[key]='never-send-'+key
    poisoned['runtime_events'].append({'secret':'never-send-runtime'})
    context=action_context(poisoned,service.registry,emit_event=False)
    text=canonical(context)
    assert 'never-send' not in text
    assert all(token.get_secret_value() not in text for token in tokens.values())
    assert len(context['available_write_tools'])==2
