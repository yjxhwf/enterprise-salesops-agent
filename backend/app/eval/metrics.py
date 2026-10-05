"""Rates retain denominators; unavailable is never silently counted as zero."""
from collections import Counter


def rate(numerator, denominator):
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError('Invalid rate counts')
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def recall_at_3(expected, ranked_docs):
    wanted = set(expected)
    return rate(len(wanted & set(ranked_docs[:3])), len(wanted))


def selection_scores(expected, actual, *, live=False, planned=5):
    if not live:
        raise ValueError('Offline scripted routing must not be reported as model F1')
    completed = sum(v is not None for v in actual)
    if len(expected) != planned or len(actual) != planned or completed != planned:
        return dict(status='PARTIAL', completed=completed, planned=planned,
                    accuracy=None, precision=None, recall=None, f1=None)
    tp = sum(a == b for a, b in zip(expected, actual))
    # Single-label micro averaging: FP = FN = number of wrong predictions.
    score = tp / planned
    return dict(status='COMPLETE', completed=completed, planned=planned,
                averaging='micro', accuracy=score, precision=score, recall=score, f1=score)


def aggregate(results):
    def counts(key):
        values = [r.metrics.get(key, {}) for r in results]
        return rate(sum(v.get('numerator', 0) for v in values), sum(v.get('denominator', 0) for v in values))
    names = ['task_completion', 'grounded_finding_accuracy', 'semantic_safety',
             'hybrid_recall_at_3', 'vector_recall_at_3', 'bm25_recall_at_3',
             'unauthorized_write', 'loop_escape', 'tool_success', 'runtime_safety', 'tool_routing_regression']
    metrics = {name: counts(name) for name in names}
    metrics.update(offline_average_latency_ms=None, offline_average_tokens=None,
                   offline_cost_note='Scripted offline metrics are NOT representative of provider latency or token cost.')
    categories = {}
    for category in ('Normal', 'Failure', 'Adversarial', 'RAG', 'Permission'):
        rows = [r for r in results if r.category == category]
        categories[category] = dict(total=len(rows), passed=sum(r.passed for r in rows),
                                    failed=sum(not r.passed for r in rows), **rate(sum(r.passed for r in rows), len(rows)))
    return categories, metrics
