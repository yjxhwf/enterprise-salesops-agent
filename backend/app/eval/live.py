"""Opt-in, one-shot, at most five physical Selector requests; no full Agent run."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from backend.app.agent.nodes.select_tool import make_select_node
from backend.app.agent.state import initial_state
from backend.app.agent.tooling import ToolCatalog
from backend.app.config import get_settings
from backend.app.llm.provider import ProductionGateway
from backend.app.eval.metrics import selection_scores
from backend.app.eval.report import export


class MeteredClient:
    """Observe usage and latency only. Never store response bodies or model arguments."""
    def __init__(self, client):
        self.client = client
        self.calls = 0
        self.usage = None
    def __getattr__(self, name): return getattr(self.client, name)
    def bind_tools(self, *args, **kwargs):
        bound = self.client.bind_tools(*args, **kwargs)
        owner = self
        class Request:
            def invoke(self, *args, **kwargs):
                if owner.calls >= 1: raise RuntimeError('SELECTOR_CALL_LIMIT')
                owner.calls += 1
                response = bound.invoke(*args, **kwargs)
                usage = getattr(response, 'usage_metadata', None)
                if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens','output_tokens','total_tokens')):
                    if usage['total_tokens'] == usage['input_tokens'] + usage['output_tokens']:
                        owner.usage = {k:usage[k] for k in ('input_tokens','output_tokens','total_tokens')}
                return response
        return Request()


def selector_once(inputs, settings):
    # Expected tool is deliberately not accepted by this function.
    gateway = ProductionGateway(settings)
    meter = MeteredClient(gateway._client())
    gateway._model = meter
    state = initial_state(inputs['query'])
    state.update(goal=inputs['goal'], plan={'steps':[inputs['step']]}, current_step_index=0, status='PLANNED')
    started = perf_counter()
    try:
        result = make_select_node(gateway, ToolCatalog())(state)
        selection = result.get('selected_tool')
        code = result['errors'][-1]['code'] if result.get('errors') else None
    except Exception as error:
        selection, code = None, type(error).__name__
    safe_args = {}
    if selection:
        for key, value in selection['arguments'].items():
            if key in ('start_date','end_date','region','product_id','approval_status','limit','top_k'):
                safe_args[key] = value
    return dict(actual=selection['tool_name'] if selection else None, safe_args=safe_args,
                latency_ms=round((perf_counter()-started)*1000, 3), usage=meter.usage,
                physical_calls=meter.calls, error_code=code)


def benchmark(cases, execute):
    if len(cases) != 5: raise ValueError('Exactly five selector cases required')
    rows = []
    for case in cases:
        try: row = execute(case['input'])
        except Exception as error:
            row = dict(actual=None, safe_args={}, latency_ms=None, usage=None, physical_calls=0, error_code=type(error).__name__)
        rows.append(dict(case_id=case['case_id'], expected=case['expected_tool'], **row,
                         passed=row['actual'] == case['expected_tool']))
    scores = selection_scores([r['expected'] for r in rows], [r['actual'] for r in rows], live=True)
    usages = [r['usage'] for r in rows if r.get('usage') is not None]
    latencies = [r['latency_ms'] for r in rows if r.get('latency_ms') is not None]
    return dict(scope='Small Live Selector Benchmark; not Full Agent Cost', sample_size=5,
        provider='SiliconFlow', model='deepseek-ai/DeepSeek-V4-Flash', scores=scores, cases=rows,
        physical_calls=sum(r['physical_calls'] for r in rows), usage_samples=len(usages),
        average_usage={k:sum(u[k] for u in usages)/len(usages) if usages else None for k in ('input_tokens','output_tokens','total_tokens')},
        average_latency_ms=sum(latencies)/len(latencies) if latencies else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='tests/fixtures/eval/live_selector.json')
    parser.add_argument('--output', default='.verification/task10-live-selector')
    args = parser.parse_args()
    settings = get_settings()
    if settings.LLM_PROVIDER.lower() != 'siliconflow' or settings.LLM_MODEL != 'deepseek-ai/DeepSeek-V4-Flash':
        raise ValueError('Benchmark requires the explicitly authorized provider and model')
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    if len(cases) != 5: raise ValueError('Expected five cases')
    lock = Path('.verification/task10-live-selector.started')
    lock.parent.mkdir(exist_ok=True)
    with lock.open('x', encoding='utf-8') as stream: stream.write('One-shot benchmark reserved. Do not rerun.')
    report = benchmark(cases, lambda inputs: selector_once(inputs, settings))
    export(report, args.output, secrets=[settings.LLM_API_KEY.get_secret_value()] if settings.LLM_API_KEY else [])
    print(json.dumps(report['scores']))


if __name__ == '__main__': main()
