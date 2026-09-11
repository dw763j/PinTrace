"""Withdrawal-only recheck of the neighbor-version experiment selection (FIXME 1).

Run on do from the project root with the project virtualenv:

    python -m scripts.audit_neighbor_withdrawn_scope

Writes ``outputs/analysis/rq2/withdrawn_neighbor_scope.json``.

Population: the executed neighbor-version experiment
(outputs/tosem_revision_20260812_neighbor_1f1644e, baseline
``safe_ty_install_error``, mode inline, Python 3.12, all ten models) over the
full 1,000-task PinTrace set. The executed cohort selected task--model
observations that (a) contain a direct ``model_exact`` dependency and
(b) are safe, i.e. carry no vulnerable finding under the stored OSV matching,
while exhibiting a ty install error. The raw run records are read from the
same outputs root recorded in the executed result files.

This script first reproduces the executed selection exactly from the raw
``m5_compat_records.json`` files and asserts task-id equality with the saved
per-task results. It then rebuilds "safe" with a withdrawal-only index (OSV
records with nonempty ``withdrawn`` skipped; no other matching change) and
reports the eligible population after the filter, the retained tested cohort,
and the newly eligible but untested observations. No experiment is re-run and
no stored output is modified.
"""
import inspect
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from paths import ANALYSIS_RQ2, OSV_INDEX
from stages import vuln_checker as vc
import run_neighbor_version_experiment as nve

OSV = OSV_INDEX
RESULTS = sorted((ROOT / 'outputs/tosem_revision_20260812_neighbor_1f1644e/d2')
                 .glob('safe_ty_install_error_inline_*.json'))
assert len(RESULTS) == 10, [p.name for p in RESULTS]
OUT_DIR = ANALYSIS_RQ2
BASELINE = 'safe_ty_install_error'
ORIGIN = 'model_exact'

old_index = vc.load_osv_index(str(OSV))
source = inspect.getsource(vc.load_osv_index)
needle = '            data = json.load(f)\n'
assert source.count(needle) == 1
source = source.replace(needle, needle + '        if data.get("withdrawn"):\n            continue\n')
namespace = dict(vc.__dict__)
exec(compile(source, '<withdrawal-only-index>', 'exec'), namespace)
active_index = namespace['load_osv_index'](str(OSV))


@lru_cache(None)
def still_vulnerable(package, version):
    osv_ids, cve_ids = vc._collect_cve_ids_for_package_version(active_index.get(package, {}), version)
    return bool(osv_ids or cve_ids)


def task_vulnerable_after(record):
    for finding in record.get('vuln_findings') or []:
        if not finding.get('is_vulnerable'):
            continue
        package = str(finding.get('pypi_name') or '').lower().replace('_', '-')
        version = str(finding.get('version') or finding.get('resolved_version') or '')
        if still_vulnerable(package, version):
            return True
    return False


per_model = []
for path in RESULTS:
    data = json.loads(path.read_text(encoding='utf-8'))
    selected = data['run_selection']['selected_run']
    outputs_root = Path(data['experiment_config']['outputs_root'])
    if not outputs_root.is_absolute():
        outputs_root = ROOT / outputs_root
    run_dir = outputs_root / selected['mode'] / selected['run_name'] / selected['python_tag']
    model = json.loads((run_dir / 'metrics_summary.json').read_text(encoding='utf-8'))['model_name']
    raw = json.loads((run_dir / 'm5_compat_records.json').read_text(encoding='utf-8'))

    tested = Counter(str(row.get('task_id', '')).strip() for row in data['task_results'])
    baseline_before = [r for r in raw
                       if nve._is_baseline_record(r, BASELINE)
                       and nve._record_has_direct_origin(r, ORIGIN)]
    recomputed = Counter(str(r.get('task_id', '')).strip() for r in baseline_before)
    assert tested == recomputed, (path.name, len(tested), len(recomputed))

    eligible_after_ids = set()
    for record in raw:
        ty = ((record.get('compat_results') or {}).get('ty') or {})
        if not nve._has_ty_install_error(ty):
            continue
        if not nve._record_has_direct_origin(record, ORIGIN):
            continue
        if task_vulnerable_after(record):
            continue
        eligible_after_ids.add(str(record.get('task_id', '')).strip())
    tested_ids = set(tested)
    assert tested_ids <= eligible_after_ids, (path.name, len(tested_ids - eligible_after_ids))
    summary = data['summary']
    per_model.append({
        'model': model,
        'result_file': path.name,
        'run_name': selected['run_name'],
        'tested': len(tested_ids),
        'eligible_after_withdrawn': len(eligible_after_ids),
        'newly_eligible_untested': len(eligible_after_ids - tested_ids),
        'newly_eligible_task_ids': sorted(eligible_after_ids - tested_ids),
        'recover_no_ty_install_error_tasks': summary.get('recover_no_ty_install_error_tasks'),
        'recover_ty_tasks': summary.get('recover_ty_tasks'),
    })

totals = {
    'tested': sum(m['tested'] for m in per_model),
    'eligible_after_withdrawn': sum(m['eligible_after_withdrawn'] for m in per_model),
    'newly_eligible_untested': sum(m['newly_eligible_untested'] for m in per_model),
    'recover_no_ty_install_error_tasks': sum(m['recover_no_ty_install_error_tasks'] for m in per_model),
    'recover_ty_tasks': sum(m['recover_ty_tasks'] for m in per_model),
}
assert totals['tested'] == 1237, totals
assert totals['eligible_after_withdrawn'] == 1290, totals
assert totals['newly_eligible_untested'] == 53, totals
assert totals['recover_no_ty_install_error_tasks'] == 478, totals
assert totals['recover_ty_tasks'] == 410, totals

result = {
    'scope': 'withdrawal-only neighbor eligibility recheck; no experiment re-run',
    'baseline': BASELINE,
    'origin': ORIGIN,
    'population': 'full 1,000-task PinTrace, 10 models, mode inline, Python 3.12',
    'totals': totals,
    'per_model': per_model,
}
(OUT_DIR / 'withdrawn_neighbor_scope.json').write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('scope', 'totals')}, indent=2))
print('per-model tested/eligible/new:',
      {m['model']: (m['tested'], m['eligible_after_withdrawn'], m['newly_eligible_untested'])
       for m in per_model})
