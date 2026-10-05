"""Test-only proposals derive target IDs from actual supplied business evidence."""
from copy import deepcopy
from backend.app.actions.service import ApprovalService
from backend.app.actions.store import ApprovalStore
from backend.app.agent.graph import build_investigation_graph
from backend.app.agent.state import initial_state
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.policy_evidence import policy_catalog
from backend.app.data.database import session_factory
from backend.app.tools.registry import build_default_registry
from tests.agent_fakes import FakeInvestigationGateway, goal, plan, selection
from tests.test_agent_investigation import draft_from_evidence


def proposal_for(context, *, alert=False):
    evidence = next(e for e in context['selected_business_evidence'] if e['subject']['type']=='CUSTOMER')
    eid, customer = evidence['evidence_id'], evidence['subject']['id']
    index = next(i for i,f in enumerate(context['findings']) if eid in f['supporting_evidence_ids'])
    policies = context['selected_policy_evidence']
    return dict(action_required=True, proposal=dict(
        action_type='CREATE_BUSINESS_ALERT' if alert else 'CREATE_CRM_TASK',
        tool_name='create_business_alert' if alert else 'create_crm_task',
        arguments=dict(target_type='CUSTOMER',target_id=customer,severity='MEDIUM',reason='需要进一步核实经营风险。') if alert else
                  dict(customer_id=customer,reason='经营测量提示应核实客户变化。',priority='MEDIUM',suggested_action='核实客户报价和审批记录，不预设违规结论。'),
        reason='根据经营证据提出人工复核建议。',related_finding_indexes=[index],business_evidence_ids=[eid],
        policy_evidence_ids=[policies[0]['evidence_id']] if policies else [],policy_basis=bool(policies),
        risk_level='LOW',expected_outcome='取得客户跟进记录，核实风险。'))


class ActionFake(FakeInvestigationGateway):
    def __init__(self, *, policy=False, decision=None):
        def first(**context):
            return dict(decision='CONTINUE' if policy else 'COMPLETE', selected_evidence_ids=list(extract_evidence(context['history']))[:1],
                        remaining_evidence_gap='缺少公司审批制度出处' if policy else None,
                        next_objective='检索公司审批制度' if policy else None,
                        updated_plan={'steps':[dict(objective='检索公司审批制度',expected_output='政策引用')]} if policy else None)
        def second(**context):
            return dict(decision='COMPLETE',selected_evidence_ids=list(policy_catalog(context['history']))[:1],
                        remaining_evidence_gap=None,next_objective=None,updated_plan=None)
        def draft(**context):
            result=draft_from_evidence(observations=[o for o in context['observations'] if o['facts']])
            policies=[p for o in context['observations'] for p in o.get('policy_evidence',[])]
            if policies:
                result['recommendations'][0].update(policy_evidence_ids=[policies[0]['evidence_id']],policy_interpretation='请核对政策适用范围与审批留痕。')
            return result
        super().__init__(goal(),plan(),[selection('analyze_customer_performance'),dict(tool_name='search_sales_policy',arguments={'query':'折扣审批'})], [first,second],draft)
        self.decision=decision or proposal_for
        self.proposal_calls=0

    def propose_action(self, context):
        self.calls.append(('propose_action',deepcopy(context)))
        self.proposal_calls+=1
        return self.decision(context)


def action_run(engine, *, retriever=None, policy=False, decision=None, store=None, limits=None, query=None):
    registry=build_default_registry(session_factory(engine),policy_retriever=retriever)
    service=ApprovalService(registry,session_factory(engine),store=store)
    delivered={}
    fake=ActionFake(policy=policy,decision=decision)
    graph=build_investigation_graph(fake,registry,action_service=service,
          approval_token_sink=lambda aid,token:delivered.update({aid:token}),
          runtime_limits=limits,sleeper=lambda _:None,raise_unexpected_errors=True)
    state=graph.invoke(initial_state(query or '分析经营风险，结合政策给出建议，并提出一个待审批客户跟进动作。'))
    return state,service,delivered,fake
