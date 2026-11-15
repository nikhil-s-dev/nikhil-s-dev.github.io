"""Small reproducible smoke evaluation; not a public retrieval benchmark."""
import json
import platform
import statistics
import time
from pathlib import Path
from rag import Index

CASES = [
    ('energy isolation permit', ['public'], 'permit-01'),
    ('approved maintenance permit', ['public'], 'permit-01'),
    ('pump abnormal vibration', ['public'], 'pump-01'),
    ('inspection vibration reading', ['public'], 'pump-01'),
    ('unresolved alarms shift handover', ['public'], 'handover-01'),
    ('open requests handover', ['public'], 'handover-01'),
    ('R17 vibration limit', ['maintenance'], 'restricted-01'),
    ('fictional pump limit millimeters', ['maintenance'], 'restricted-01'),
]
docs = json.loads(Path(__file__).with_name('documents.json').read_text())
index = Index(docs)
rows = []
for query, groups, expected in CASES:
    start = time.perf_counter()
    results = index.search(query, groups)
    elapsed_ms = (time.perf_counter() - start) * 1000
    ids = [result['id'] for result in results]
    rows.append({'query': query, 'groups': groups, 'expected': expected, 'ranked_ids': ids, 'elapsed_ms': elapsed_ms})
report = {'dataset': '8 hand-written queries, 4 fictional documents', 'python': platform.python_version(),
          'hit_at_1': sum(r['ranked_ids'][:1] == [r['expected']] for r in rows) / len(rows),
          'hit_at_3': sum(r['expected'] in r['ranked_ids'][:3] for r in rows) / len(rows),
          'median_ms': statistics.median(r['elapsed_ms'] for r in rows), 'queries': rows}
print(json.dumps(report, indent=2))
