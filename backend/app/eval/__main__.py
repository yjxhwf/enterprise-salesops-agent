"""Explicit developer CLI; test adapters are imported only for offline execution."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from backend.app.eval.runner import load_cases, run
from backend.app.eval.report import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='tests/fixtures/eval/cases.json')
    parser.add_argument('--output', default='.verification/task10-eval-report')
    args = parser.parse_args()
    from tests.eval_support import OfflineExecutor, file_hash
    main_db = 'backend/storage/salesops.db'
    before = file_hash(main_db)
    cases = load_cases(args.cases)
    timings = []
    with OfflineExecutor() as executor:
        def measured(request):
            start = perf_counter()
            try: return executor(request)
            finally: timings.append((perf_counter()-start)*1000)
        report = run(cases, measured)
    if file_hash(main_db) != before: raise RuntimeError('MAIN_DB_CHANGED')
    export(report, args.output)
    Path(args.output + '-timing.json').write_text(json.dumps(dict(average_latency_ms=sum(timings)/len(timings),
        scope='Offline local harness only; NOT representative of provider latency. Excluded from deterministic scoring.',
        average_token_usage=None)), encoding='utf-8')
    print(f'{report.passed_cases}/{report.total_cases}; main DB unchanged; artifacts: {args.output}')
    # Eval failures are measurements, not harness process failures.


if __name__ == '__main__': main()
