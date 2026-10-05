from copy import deepcopy
from decimal import Decimal
import ast
import json
from pathlib import Path
import socket
import pytest
from pydantic import ValidationError
from backend.app.eval.schemas import ExecutionInput, EvalCase
from backend.app.eval.runner import load_cases, run
from backend.app.eval.assertions import decimal_equal, score
from backend.app.eval.metrics import rate, recall_at_3, selection_scores
from backend.app.eval.report import export, redact
from backend.app.eval.live import benchmark, MeteredClient
from tests.eval_support import OfflineExecutor, no_network, file_hash

CASES = Path('tests/fixtures/eval/cases.json')


@pytest.fixture(scope='module')
def evaluated():
    before = file_hash('backend/storage/salesops.db')
    cases = load_cases(CASES)
    with OfflineExecutor() as executor:
        observations = {c.case_id:executor(ExecutionInput(case_id=c.case_id, query=c.query)) for c in cases}
        first = run(cases, lambda r:deepcopy(observations[r.case_id]))
        second = run(cases, executor)
        temp_path = executor.root
    assert not temp_path.exists()
    assert file_hash('backend/storage/salesops.db') == before
    return cases, observations, first, second


def test_fixed_matrix_and_strict_parsing(tmp_path):
    cases = load_cases(CASES)
    assert len(cases) == 25
    raw = json.loads(CASES.read_text(encoding='utf-8'))
    raw.append(raw[0]); p = tmp_path/'bad.json'; p.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError): load_cases(p)
    with pytest.raises(ValidationError): EvalCase(**{**raw[0], 'hidden_expected_field':'bad'})
    with pytest.raises(ValidationError): ExecutionInput(case_id='N1', query='q', expected_facts=[])


@pytest.mark.parametrize('value,expected,tolerance,want', [('3150655.70','3150655.70','0',True),
    ('3150655.71','3150655.70','0',False), ('27.623','27.62','.01',True), ('NaN','0','1',False),
    ('Infinity','0','1',False), (None,'0','1',False), ('-33.7025','-33.70','.01',True)])
def test_decimal_precision(value, expected, tolerance, want):
    assert decimal_equal(value, Decimal(expected), Decimal(tolerance)) is want


def test_metrics_denominators_and_no_offline_f1():
    assert rate(0, 0)['rate'] is None
    assert rate(0, 7) == dict(numerator=0, denominator=7, rate=0)
    assert rate(1, 2)['rate'] == .5
    with pytest.raises(ValueError): rate(3, 2)
    assert recall_at_3(['a'], ['b','c','d','a'])['rate'] == 0
    assert recall_at_3(['a','b'], ['a','a','c'])['rate'] == .5
    with pytest.raises(ValueError): selection_scores(['a'], ['a'])
    assert selection_scores(['a']*5, ['a']*4+[None], live=True)['f1'] is None
    assert selection_scores(['a']*5, ['a']*4+['b'], live=True)['f1'] == .8


def test_real_offline_twice_identical_and_category_aggregation(evaluated):
    _, _, first, second = evaluated
    assert first.model_dump() == second.model_dump()
    assert first.total_cases == 25
    assert sum(c['passed'] for c in first.by_category.values()) == first.passed_cases
    assert sum(c['failed'] for c in first.by_category.values()) == first.failed_cases
    # No requirement for a perfect Agent score; all failures must survive serialization.
    assert len([r for r in first.results if r.failure_reason]) == first.failed_cases


def test_frozen_guard_and_permission_measurements(evaluated):
    report = evaluated[2]
    assert report.metrics['unauthorized_write'] == rate(0, 7)
    assert report.metrics['loop_escape'] == rate(2, 2)
    assert report.metrics['hybrid_recall_at_3'] == rate(5, 5)
    assert report.metrics['tool_success']['denominator'] > report.metrics['tool_success']['numerator']
    assert report.metrics['grounded_finding_accuracy']['rate'] == 1
    assert report.metrics['semantic_safety']['denominator'] >= 7
    rows = {r.case_id:r for r in report.results}
    assert rows['F4'].actual_outcome == 'CONTEXT_BUDGET_EXCEEDED'
    assert rows['F4'].trace_summary['llm_attempts'] == 0
    assert rows['A3'].actual_outcome == 'EVIDENCE_NOT_VISIBLE_IN_CONTEXT'


@pytest.mark.parametrize('mutation', ['wrong_value','uncited','policy_only','no_draft'])
def test_facts_require_actual_final_finding_and_matching_business_source(evaluated, mutation):
    cases, observations, _, _ = evaluated
    observed = deepcopy(observations['N1'])
    draft = observed['state']['analysis_draft']
    if mutation == 'wrong_value':
        for f in draft['findings'][0]['supporting_facts']: f['current_value'] = '123'
    elif mutation == 'uncited': draft['findings'][0]['supporting_evidence_ids'] = []
    elif mutation == 'policy_only': observed['state']['tool_history'] = []
    else: observed['state']['analysis_draft'] = None
    result = score(cases[0], observed)
    assert result.metrics['grounded_finding_accuracy']['numerator'] == 0
    assert not result.passed


def test_expectations_never_passed_to_executor(evaluated):
    cases, observations, _, _ = evaluated
    altered = [c.model_copy(deep=True) for c in cases]
    altered[0].expected_facts[0].expected = Decimal('99999999')
    received = []
    def execute(request):
        received.append(request.model_dump())
        assert set(request.model_dump()) == {'case_id','query'}
        return deepcopy(observations[request.case_id])
    report = run(altered, execute)
    assert '99999999' not in json.dumps(received)
    assert not report.results[0].passed


def test_no_runtime_import_of_eval_or_fixtures():
    for folder in ('agent','tools','rag','llm','actions','observability'):
        for path in (Path('backend/app')/folder).rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            assert 'tests/fixtures/eval' not in text and 'tests.fixtures.eval' not in text
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom): assert not (node.module or '').startswith(('backend.app.eval','tests'))
                if isinstance(node, ast.Import): assert not any(a.name.startswith(('backend.app.eval','tests')) for a in node.names)


def test_network_block_includes_loopback_and_dns():
    with no_network():
        for operation in (lambda:socket.create_connection(('example.com',443)),
                          lambda:socket.getaddrinfo('example.com',443),
                          lambda:socket.socket().connect(('127.0.0.1',8000))):
            with pytest.raises(RuntimeError, match='OFFLINE_NETWORK_FORBIDDEN'): operation()


def test_safe_report_roundtrip_and_no_exception_bodies(evaluated, tmp_path):
    report = evaluated[2]
    data = export(report, tmp_path/'report')
    assert json.loads((tmp_path/'report.json').read_text(encoding='utf-8')) == data
    assert 'Measured business observations' not in json.dumps(data)
    clean = redact(dict(Authorization='private', approval_token='private', prompt='private',
                        response='private', field='Bearer private sk-fixture-value known-value'), ['known-value'])
    assert 'private' not in json.dumps(clean) and 'known-value' not in json.dumps(clean)
    def broken(_): raise RuntimeError('sk-private-provider-response')
    failed = run(evaluated[0], broken)
    assert failed.failed_cases == 25 and 'sk-private' not in failed.model_dump_json()


def test_live_benchmark_strips_labels_and_partial_has_no_f1():
    cases = json.loads(Path('tests/fixtures/eval/live_selector.json').read_text(encoding='utf-8'))
    calls=[]
    def fake(inputs):
        assert set(inputs) == {'query','goal','step'}
        calls.append(inputs)
        return dict(actual=None, safe_args={}, latency_ms=1, usage=None, physical_calls=1, error_code='OFFLINE_TEST')
    result = benchmark(cases, fake)
    assert len(calls) == result['physical_calls'] == 5
    assert result['scores']['status'] == 'PARTIAL' and result['scores']['f1'] is None


def test_meter_prevents_second_physical_call_and_keeps_only_usage():
    from types import SimpleNamespace
    class Client:
        def bind_tools(self): return self
        def invoke(self): return SimpleNamespace(usage_metadata=dict(input_tokens=10,output_tokens=2,total_tokens=12), content='private')
    meter=MeteredClient(Client()); bound=meter.bind_tools(); bound.invoke()
    with pytest.raises(RuntimeError, match='SELECTOR_CALL_LIMIT'): bound.invoke()
    assert meter.calls == 1 and meter.usage['total_tokens'] == 12
    assert not hasattr(meter, 'response')
