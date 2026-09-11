"""Recompute RQ2 association after skipping withdrawn OSV advisories.

Usage::

    python -m scripts.audit_osv_withdrawn

Writes ``outputs/analysis/rq2/withdrawn_impact.json`` and the official
severity / disclosure CSVs. "Before" counts are asserted against
``outputs/analysis/rq2/unfiltered/``.

Scope: the only matching change is that OSV records with a nonempty
``withdrawn`` field are skipped when the index is rebuilt. The existing
version-join logic (FIXME 2, semantic version-string normalization) is
deliberately left untouched, and the clean task list, resolution decisions,
install results, and stored run outputs are not modified.

The audit recomputes, under the withdrawal-only filter:
  * task-level RQ2 association funnels (population and conditional rates);
  * dependency-occurrence counts and the VA exact severity distribution;
  * the per-model disclosure-timing panel inputs used by
    plots/rq2_cve_characteristics.py.

Every "before" quantity is asserted against the unfiltered RQ2 tables so the
original population is verified before any "after" number is trusted.
"""
import csv
import inspect
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import ANALYSIS_RQ2, EVALUATION, OSV_INDEX
from stages import vuln_checker as vc
from stages.osv_utils import build_osv_alias_canonical_map

DATA = EVALUATION
EXTENDED = ANALYSIS_RQ2 / "unfiltered"
OSV = OSV_INDEX
OUT_DIR = ANALYSIS_RQ2


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _dt(value):
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


clean = {r['task_id'] for r in rows(ROOT / 'resources/revision/pintrace_clean_task_ids.jsonl')}
assert len(clean) == 849
advisories = {}
for path in OSV.glob('*.json'):
    record = json.loads(path.read_text())
    advisories[record['id']] = record
withdrawn = {key: value for key, value in advisories.items() if value.get('withdrawn')}

old_index = vc.load_osv_index(str(OSV))
source = inspect.getsource(vc.load_osv_index)
needle = '            data = json.load(f)\n'
assert source.count(needle) == 1
source = source.replace(needle, needle + '        if data.get("withdrawn"):\n            continue\n')
namespace = dict(vc.__dict__)
exec(compile(source, '<withdrawal-only-index>', 'exec'), namespace)
active_index = namespace['load_osv_index'](str(OSV))
alias_map = build_osv_alias_canonical_map(str(OSV))

osv_published = {}
for path in OSV.glob('*.json'):
    try:
        item = json.loads(path.read_text(encoding='utf-8'))
        published = _dt(str(item['published']))
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        continue
    for advisory_id in [item.get('id'), *(item.get('aliases') or [])]:
        if advisory_id:
            osv_published[str(advisory_id)] = published

cutoffs = json.loads((ROOT / 'resources/revision/model_knowledge_cutoff.json').read_text(encoding='utf-8'))


@lru_cache(None)
def match(package, version):
    before = vc._collect_cve_ids_for_package_version(old_index.get(package, {}), version)
    after = vc._collect_cve_ids_for_package_version(active_index.get(package, {}), version)
    assert after[0] <= before[0] and after[1] <= before[1], (package, version)
    return before, after


def pair(row):
    return (str(row.get('pypi_name') or '').lower().replace('_', '-'),
            str(row.get('version') or row.get('resolved_version') or row.get('constraint_resolved_version') or ''))


def retained_severity(cve_ids):
    severities, scores = [], []
    for cve in cve_ids:
        _, year, number = cve.split('-')
        path = ROOT / 'resources/2026-02-27_all_CVEs_at_midnight' / year / (number[:-3] + 'xxx') / (cve + '.json')
        if not path.is_file():
            continue
        severity, score = vc._extract_cvss_from_cve_record(json.loads(path.read_text()))
        severities.append(severity)
        if score is not None:
            scores.append(score)
    return vc._max_severity(severities, max(scores) if scores else None)


overlays = {(r['track'], r['mode'], r['model'], r['task_id']): r['record']
            for r in rows(DATA / 'overlays.jsonl')}
outcomes = {(r['track'], r['mode'], r['model'], r['task_id']): r
            for r in rows(DATA / 'task_outcomes.jsonl')}
funnel = defaultdict(Counter)
removed_findings = Counter()
assertion_counts = Counter()

for audit in rows(DATA / 'run_audits.jsonl'):
    if audit['track'] != 'd2' or str(audit['python_version']) != '3.12':
        continue
    run = Path(audit['run_dir'])
    audit['model'] = json.loads((run / 'metrics_summary.json').read_text())['model_name']
    raw_path = next(run / n for n in ['m5_compat_records.json', 'm4_vuln_records.json', 'm3_resolved_records.json'] if (run / n).is_file())
    raw = json.loads(raw_path.read_text())
    for record in raw:
        task = str(record['task_id'])
        if task not in clean:
            continue
        key = ('d2', audit['mode'], audit['model'], task)
        record = overlays.get(key, record)
        taskrow = outcomes[key]
        for kind, origin in [('exact', 'model_exact'), ('range', 'model_constrained_resolver')]:
            group = (audit['mode'], kind, audit['model'])
            bucket = funnel[group]
            bucket['N'] += 1
            bucket['P'] += bool(taskrow.get(f'has_valid_model_{kind}_constraint'))
            findings = [f for f in record.get('vuln_findings', [])
                        if f.get('is_vulnerable') and f.get('constraint_kind') == kind
                        and f.get('resolution_origin') == origin and f.get('dependency_scope') == 'direct']
            before = bool(findings)
            assert before == bool(taskrow.get(f'has_model_{kind}_cve')), key
            after = False
            for finding in findings:
                package, version = pair(finding)
                original, active = match(package, version)
                assert original[0] or original[1], (key, package, version)
                if active[0] or active[1]:
                    after = True
                else:
                    removed_findings[(audit['mode'], kind, package)] += 1
                if original[1] != active[1]:
                    assertion_counts['raw_findings_changed_cve_set'] += 1
            bucket['V_before'] += before
            bucket['V_after'] += after
            bucket['lost_observations'] += before and not after
    del raw

direct_counts = defaultdict(Counter)
removed_advisories = Counter()
severity_before = Counter()
severity_after = Counter()
severity_before_by_model = defaultdict(Counter)
severity_after_by_model = defaultdict(Counter)
va_pairs_before = Counter()
va_pairs_after = Counter()
va_ids_before = defaultdict(set)
va_ids_after = defaultdict(set)
va_dates_before = defaultdict(dict)
va_dates_after = defaultdict(dict)
retained_changed_cve_examples = []

for row in rows(DATA / 'direct_dependencies.jsonl'):
    if (row.get('track') != 'd2' or row.get('task_id') not in clean
            or str(row.get('python_version')) != '3.12' or not row.get('is_direct')
            or not row.get('constraint_valid') or row.get('constraint_kind') not in {'exact', 'range'}
            or not row.get('is_cve_associated')):
        continue
    category = (row['mode'], row['constraint_kind'])
    original, active = match(*pair(row))
    assert original[0] or original[1], pair(row)
    counts = direct_counts[category]
    counts['before'] += 1
    retained = bool(active[0] or active[1])
    counts['after'] += retained
    if not retained:
        removed_advisories.update(row.get('osv_ids') or [])
    if original[1] != active[1]:
        counts['changed_cve_set'] += 1
        if retained:
            retained_changed_cve_examples.append({
                'category': category, 'model': row['model'], 'task_id': row['task_id'],
                'pair': pair(row), 'stored_severity': row.get('max_severity'),
                'removed_cves': sorted(original[1] - active[1]),
                'retained_cves': sorted(active[1]),
            })
    if category == ('inline', 'exact'):
        model = row['model']
        severity = row.get('max_severity') or 'UNKNOWN'
        severity_before[severity] += 1
        severity_before_by_model[model][severity] += 1
        va_pairs_before[pair(row)] += 1
        va_ids_before[model].update(row.get('osv_ids') or [])
        for advisory_id in [*(row.get('cve_ids') or []), *(row.get('osv_ids') or [])]:
            advisory_id = str(advisory_id)
            if advisory_id in osv_published:
                va_dates_before[model][advisory_id] = osv_published[advisory_id]
        if retained:
            new_severity = retained_severity(active[1]) if original[1] != active[1] else severity
            severity_after[new_severity or 'UNKNOWN'] += 1
            severity_after_by_model[model][new_severity or 'UNKNOWN'] += 1
            va_pairs_after[pair(row)] += 1
            va_ids_after[model].update(identifier for identifier in row.get('osv_ids') or []
                                       if alias_map.get(identifier, identifier) in active[0])
            for advisory_id in [*(row.get('cve_ids') or []), *(row.get('osv_ids') or [])]:
                advisory_id = str(advisory_id)
                if advisory_id in osv_published and alias_map.get(advisory_id, advisory_id) in active[0]:
                    va_dates_after[model][advisory_id] = osv_published[advisory_id]

summary = defaultdict(Counter)
per_model = []
for (mode, kind, model), values in sorted(funnel.items()):
    summary[(mode, kind)].update(values)
    per_model.append(dict(mode=mode, kind=kind, model=model, **values))
for (mode, kind), counts in summary.items():
    counts['population_before_pct'] = 100 * counts['V_before'] / counts['N']
    counts['population_after_pct'] = 100 * counts['V_after'] / counts['N']
    counts['conditional_before_pct'] = 100 * counts['V_before'] / counts['P'] if counts['P'] else None
    counts['conditional_after_pct'] = 100 * counts['V_after'] / counts['P'] if counts['P'] else None


def disclosure_rows(va_dates):
    emitted = []
    for model, advisories in sorted(va_dates.items()):
        cutoff = cutoffs.get(model.strip().lower())
        if not cutoff:
            raise KeyError(f'no knowledge cutoff for {model!r}')
        base = _dt(cutoff['date'])
        offsets = (-6, -3, 0, 3, 6) if cutoff['source'] == 'self_reported' else (0,)
        for offset in offsets:
            shifted = base + timedelta(days=offset * 365.25 / 12)
            before_count = sum(published <= shifted for published in advisories.values())
            emitted.append({
                'model': model,
                'offset_months': offset,
                'pre_cutoff': before_count,
                'pre_cutoff_rate': before_count / len(advisories) if advisories else None,
                'source': cutoff['source'],
                'unique_advisories': len(advisories),
            })
    return emitted


disclosure_before_rows = disclosure_rows(va_dates_before)
disclosure_after_rows = disclosure_rows(va_dates_after)

existing_disclosure = defaultdict(dict)
with (EXTENDED / 'cve_disclosure_exact.csv').open(encoding='utf-8', newline='') as handle:
    for rec in csv.DictReader(handle):
        existing_disclosure[rec['model']][int(rec['offset_months'])] = rec
for row in disclosure_before_rows:
    saved = existing_disclosure[row['model']][row['offset_months']]
    assert int(saved['unique_advisories']) == row['unique_advisories'], row
    assert int(saved['pre_cutoff']) == row['pre_cutoff'], row
    assert abs(float(saved['pre_cutoff_rate']) - row['pre_cutoff_rate']) < 1e-9, row

existing_severity = defaultdict(dict)
with (EXTENDED / 'vulnerability_severity_exact.csv').open(encoding='utf-8', newline='') as handle:
    for rec in csv.DictReader(handle):
        existing_severity[rec['model']][rec['severity']] = int(rec['occurrences'])
for model, counts in severity_before_by_model.items():
    assert {k: v for k, v in existing_severity[model].items() if v} == dict(counts), model

with (OUT_DIR / 'vulnerability_severity_exact.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['model', 'occurrences', 'severity', 'unit'])
    writer.writeheader()
    for model in sorted(severity_after_by_model):
        for severity, count in sorted(severity_after_by_model[model].items()):
            if count:
                writer.writerow({'model': model, 'occurrences': count,
                                 'severity': severity, 'unit': 'dependency_occurrence'})

with (OUT_DIR / 'cve_disclosure_exact.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['model', 'offset_months', 'pre_cutoff',
                                                'pre_cutoff_rate', 'source', 'unique_advisories'])
    writer.writeheader()
    writer.writerows(disclosure_after_rows)

result = {
    'scope': 'withdrawal-only sensitivity; original version-join untouched; not a fully corrected release',
    'snapshot': {'records': len(advisories), 'withdrawn': len(withdrawn),
                 'withdrawn_min': min(r['withdrawn'] for r in withdrawn.values()),
                 'withdrawn_max': max(r['withdrawn'] for r in withdrawn.values())},
    'funnel': {':'.join(k): dict(v) for k, v in summary.items()},
    'per_model': per_model,
    'direct_occurrences': {':'.join(k): dict(v) for k, v in direct_counts.items()},
    'removed_raw_findings_by_package': {':'.join(k): v for k, v in removed_findings.items()},
    'removed_stored_advisory_ids': dict(removed_advisories),
    'va_severity_before': dict(severity_before), 'va_severity_after': dict(severity_after),
    'va_severity_after_by_model': {m: dict(c) for m, c in sorted(severity_after_by_model.items())},
    'va_leading_pairs_before': [dict(package=k[0], version=k[1], n=v) for k, v in va_pairs_before.most_common(5)],
    'va_leading_pairs_after': [dict(package=k[0], version=k[1], n=v) for k, v in va_pairs_after.most_common(5)],
    'va_stored_osv_ids_per_model': {m: {'before': len(ids), 'after': len(va_ids_after[m]),
                                        'removed': sorted(ids - va_ids_after[m])}
                                    for m, ids in va_ids_before.items()},
    'va_unique_advisories_per_model': {
        m: {'before': len(va_dates_before[m]), 'after': len(va_dates_after[m])}
        for m in sorted(va_dates_before)},
    'disclosure_after_offset0': {r['model']: {'unique': r['unique_advisories'],
                                              'pre_cutoff': r['pre_cutoff'],
                                              'rate': r['pre_cutoff_rate']}
                                 for r in disclosure_after_rows if r['offset_months'] == 0},
    'disclosure_after_rows': disclosure_after_rows,
    'assertion_counts': dict(assertion_counts),
    'retained_changed_cve_examples': retained_changed_cve_examples,
    'matched_unique_pairs': match.cache_info().currsize,
}
(OUT_DIR / 'withdrawn_impact.json').write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
