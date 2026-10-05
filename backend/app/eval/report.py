import json
import re
from pathlib import Path


def redact(value, secrets=()):
    """Defense in depth after the runner's typed, body-free projection."""
    if isinstance(value, dict):
        return {k: ('[REDACTED]' if re.search(r'(?i)authorization|api.?key|approval.?token|token.?hash|prompt|response|secret|password', k)
                    else redact(v, secrets)) for k, v in value.items()}
    if isinstance(value, list): return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret: value = value.replace(secret, '[REDACTED]')
        return re.sub(r'(?i)sk-[\w-]+|Bearer\s+\S+', '[REDACTED]', value)
    return value


def export(report, prefix, *, secrets=()):
    data = redact(report.model_dump(mode='json') if hasattr(report, 'model_dump') else report, secrets)
    path = Path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Task 10 Evaluation Report', '']
    if 'results' in data:
        lines += [f"Offline scripted: {data['passed_cases']}/{data['total_cases']} passed ({data['pass_rate']:.2%}).", '',
                  '| Case | Category | Actual outcome | Pass | Failed assertions |', '|---|---|---|---|---|']
        for row in data['results']:
            lines.append(f"| {row['case_id']} | {row['category']} | {row['actual_outcome']} | {row['passed']} | {', '.join(row['failure_reason'])} |")
        lines += ['', 'Offline scripted routing is NOT live model accuracy. Fake embeddings do not measure production embedding quality.', '']
    lines += ['## Full structured results', '', '```json', json.dumps(data, ensure_ascii=False, indent=2), '```']
    path.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    return data
