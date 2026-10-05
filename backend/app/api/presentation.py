"""Local demo HTTP adapter. No planning, scoring, approval or write decisions."""
from collections import OrderedDict
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from backend.app.agent.graph import build_investigation_graph
from backend.app.agent.state import initial_state
from backend.app.agent.evidence import extract_evidence
from backend.app.agent.facts import canonical_facts
from backend.app.agent.policy_evidence import policy_catalog
from backend.app.data.database import PROJECT_ROOT
from backend.app.llm.provider import ProductionGateway
from backend.app.rag.embeddings import ProductionEmbeddingGateway
from backend.app.rag.retriever import PolicyRetriever
from backend.app.tools.base import readonly_factory
from backend.app.tools.registry import ToolRegistry


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    query: str = Field(min_length=1, max_length=4000)


STAGES = {'understand_goal':'Understanding', 'plan':'Planning', 'select_tool':'Business Investigation',
          'execute_tool':'Business Investigation', 'review_progress':'Business Investigation',
          'synthesize':'Synthesis', 'propose_action':'Action Proposal'}


def scrub(value, secrets):
    if isinstance(value, dict): return {k:scrub(v, secrets) for k,v in value.items()}
    if isinstance(value, list): return [scrub(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret: value = value.replace(secret, '[REDACTED]')
        return re.sub(r'(?i)sk-[\w-]+|Bearer\s+\S+', '[REDACTED]', value)
    return value


def public_state(state):
    """Explicit UI projection; never dump AgentState, Context, Trace or token records."""
    history = state.get('tool_history', [])
    business = extract_evidence(history)
    policies = policy_catalog(history)
    draft = state.get('analysis_draft') or {}
    pending = state.get('pending_action')
    findings = [dict(title=f['title'], interpretation=f['interpretation'],
                     evidence_ids=f['supporting_evidence_ids'],
                     facts=[x['rendered_text'] for x in f.get('supporting_facts', [])]) for f in draft.get('findings', [])]
    recommendations = [dict(title=r['title'], action=r['action'], policy_evidence_ids=r.get('policy_evidence_ids', []),
                            policy_interpretation=r.get('policy_interpretation')) for r in draft.get('recommendations', [])]
    facts = canonical_facts(business)
    evidence = [dict(evidence_id=eid, source_tool=e['source_tool'], kind='BUSINESS_DATA',
                     measurements=[f['rendered_text'] for f in facts if f['evidence_id']==eid]) for eid,e in business.items()]
    policy = [dict(evidence_id=eid, kind='POLICY', citation=e['citation'], excerpt=e.get('excerpt', '')) for eid,e in policies.items()]
    return dict(run_id=state['run_id'], status=state['status'], stop_reason=state.get('stop_reason'),
        clarification=(state.get('goal') or {}).get('clarification_question'),
        executive_interpretation=draft.get('executive_interpretation'), findings=findings, recommendations=recommendations,
        business_evidence=evidence, policy_evidence=policy, limitations=draft.get('limitations', []),
        pending_action={k:pending[k] for k in ('action_id','tool_name','reason','risk_level','evidence_ids','expected_outcome','approval_status','expires_at')} if pending else None,
        errors=[dict(node=e.get('node'), code=e['code']) for e in state.get('errors', [])],
        counts=dict(llm_calls=state.get('llm_call_count',0), read_tools=state.get('tool_call_count',0), retries=state.get('retry_count',0)))


class PresentationService:
    def __init__(self, settings, approval_service, *, gateway_factory=None, registry_factory=None):
        self.settings, self.approvals = settings, approval_service
        self.gateway_factory = gateway_factory or (lambda: ProductionGateway(settings))
        self.registry_factory = registry_factory or self.configured_registry
        self.run_lock = Lock()
        self._tokens = OrderedDict()  # trusted application memory only; no HTTP accessor
        self.secrets = [v.get_secret_value() for v in (settings.LLM_API_KEY, settings.EMBEDDING_API_KEY) if v]

    @contextmanager
    def configured_registry(self):
        url = make_url(self.settings.DATABASE_URL)
        if url.drivername not in {'sqlite','sqlite+pysqlite'} or url.query or not url.database:
            raise ValueError('Local SQLite file required')
        path = Path(url.database)
        path = (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
        uri = path.as_uri() + '?mode=ro'
        engine = create_engine('sqlite+pysqlite://', creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False))
        try:
            with ToolRegistry(readonly_factory(engine), policy_retriever=PolicyRetriever(ProductionEmbeddingGateway(self.settings))) as registry:
                yield registry
        finally: engine.dispose()

    def deliver_token(self, action_id, token):
        self._tokens[action_id] = token
        while len(self._tokens) > 128: self._tokens.popitem(last=False)

    def investigate(self, query):
        with self.registry_factory() as registry:
            graph = build_investigation_graph(self.gateway_factory(), registry, action_service=self.approvals,
                approval_token_sink=self.deliver_token)
            last = initial_state(query)
            for state in graph.stream(last, stream_mode='values'):
                last = state
                nodes = state.get('visited_nodes', [])
                node = nodes[-1] if nodes else None
                stage = STAGES.get(node, 'Understanding')
                if node in {'select_tool','execute_tool','review_progress'} and (
                    (state.get('selected_tool') or {}).get('tool_name') == 'search_sales_policy' or
                    (state.get('latest_tool_result') or {}).get('tool_name') == 'search_sales_policy'):
                    stage = 'Policy Retrieval'
                yield dict(type='progress', stage=stage, status=state['status'],
                           read_tools=state.get('tool_call_count',0), retries=state.get('retry_count',0))
            # The Graph, not this adapter, determines terminal status and draft validity.
            yield dict(type='result', result=scrub(public_state(last), self.secrets))

    def business(self):
        dates = dict(start_date='2026-08-01', end_date='2026-08-31')
        with self.registry_factory() as registry:
            results = {key:registry.invoke(name, dates) for key,name in (
                ('overview','get_sales_overview'), ('regions','analyze_region_performance'), ('products','analyze_product_performance'))}
        if not all(r.success for r in results.values()):
            return None
        return dict(dataset='Deterministic synthetic enterprise dataset', period=dates,
                    **{key:r.model_dump(mode='json')['data'] for key,r in results.items()})


def presentation_router(service):
    router = APIRouter(prefix='/api')
    headers = {'Cache-Control':'no-store'}

    @router.get('/business')
    def business():
        try: result = service.business()
        except Exception: result = None
        return JSONResponse(result or {'error':'BUSINESS_DATA_UNAVAILABLE'}, status_code=200 if result else 503, headers=headers)

    @router.post('/agent/run')
    async def run(request: Request):
        origin = request.headers.get('origin')
        if origin and origin not in service.settings.CORS_ORIGINS:
            return JSONResponse({'error':'ORIGIN_NOT_ALLOWED'}, status_code=403, headers=headers)
        try:
            body = RunRequest.model_validate(await request.json())
            if not body.query.strip(): raise ValueError()
        except (ValidationError, ValueError, UnicodeError):
            return JSONResponse({'error':'INVALID_RUN_REQUEST'}, status_code=422, headers=headers)
        if not service.run_lock.acquire(blocking=False):
            return JSONResponse({'error':'RUN_ALREADY_IN_PROGRESS'}, status_code=409, headers=headers)

        def events():
            try:
                for event in service.investigate(body.query):
                    yield json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n'
            except Exception:
                yield json.dumps({'type':'error','code':'RUN_UNAVAILABLE'}) + '\n'
            finally: service.run_lock.release()
        return StreamingResponse(events(), media_type='application/x-ndjson', headers={**headers,'X-Accel-Buffering':'no'})

    return router
