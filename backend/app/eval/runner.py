import json
from collections import Counter
from pathlib import Path
from backend.app.eval.schemas import EvalCase, ExecutionInput, EvalReport
from backend.app.eval.assertions import score
from backend.app.eval.metrics import aggregate


def load_cases(path):
    cases = [EvalCase.model_validate(row) for row in json.loads(Path(path).read_text(encoding='utf-8'))]
    if len({c.case_id for c in cases}) != 25 or Counter(c.category for c in cases) != dict.fromkeys(('Normal', 'Failure', 'Adversarial', 'RAG', 'Permission'), 5):
        raise ValueError('Exactly 25 unique cases, five per category, are required')
    return cases


def run(cases, executor):
    results = []
    for case in cases:
        try:
            observed = executor(ExecutionInput(case_id=case.case_id, query=case.query))
        except Exception:
            # Never serialize exception messages (may contain rejected inputs or keys).
            observed = dict(outcome='HARNESS_ERROR', checks={'execution_succeeded': False})
        results.append(score(case, observed))
    categories, metrics = aggregate(results)
    passed = sum(r.passed for r in results)
    return EvalReport(total_cases=len(results), passed_cases=passed, failed_cases=len(results)-passed,
        pass_rate=passed/len(results), by_category=categories, metrics=metrics, results=results,
        limitations=['Offline scripted gateway: routing regression, not model selection quality.',
                     'Fake embeddings: retrieval pipeline regression, not production embedding quality.',
                     'Deterministic assertions measure declared facts and bounded semantic rules, not general language quality.',
                     'Offline latency and token costs are unavailable; real selector metrics are separate.'])
