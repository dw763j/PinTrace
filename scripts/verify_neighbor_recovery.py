"""Verify the neighboring-version completion run (FIXME 1 follow-up).

Run on do from the project root after the completion driver finishes:

    python -m scripts.verify_neighbor_recovery

Checks, per model and in total:
  * every completed output covers exactly the eligible population expected
    after the withdrawal exclusion (1,290 observations over ten models);
  * the executed 1,237 task results are adopted unchanged from the executed
    outputs (dict equality on every shared task id);
  * the newly executed observations are exactly the 53 task ids recorded by
    audit_withdrawn_neighbor_scope_20260908.py;
  * the recomputed summary counts are consistent with the merged task rows.

Writes ``outputs/analysis/neighbor/verification.json`` and prints a
LaTeX-ready per-model recovery table.
"""
import json
from pathlib import Path

from paths import ANALYSIS_NEIGHBOR, ANALYSIS_RQ2

ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / 'outputs/tosem_revision_20260812_neighbor_1f1644e/d2'
NEW_DIR = ANALYSIS_NEIGHBOR / 'models'
SCOPE_JSON = ANALYSIS_RQ2 / 'withdrawn_neighbor_scope.json'
OUT_JSON = ANALYSIS_NEIGHBOR / 'verification.json'

scope = json.loads(SCOPE_JSON.read_text(encoding='utf-8'))
pending_by_run = {row['run_name']: sorted(row['newly_eligible_task_ids']) for row in scope['per_model']}
eligible_by_run = {row['run_name']: row['eligible_after_withdrawn'] for row in scope['per_model']}

new_files = sorted(NEW_DIR.glob('safe_ty_install_error_inline_*.json'))
assert len(new_files) == 10, [p.name for p in new_files]

per_model = []
for path in new_files:
    new = json.loads(path.read_text(encoding='utf-8'))
    selected = new['run_selection']['selected_run']
    run_name = selected['run_name']
    old_path = OLD_DIR / path.name
    old = json.loads(old_path.read_text(encoding='utf-8'))
    assert old['run_selection']['selected_run']['run_name'] == run_name

    new_rows = {str(r['task_id']).strip(): r for r in new['task_results']}
    old_rows = {str(r['task_id']).strip(): r for r in old['task_results']}
    assert len(new_rows) == len(new['task_results']), (run_name, 'duplicate task ids')

    expected_eligible = eligible_by_run[run_name]
    assert len(new_rows) == expected_eligible, (run_name, len(new_rows), expected_eligible)
    assert set(old_rows) <= set(new_rows), (run_name, 'executed rows lost')

    adopted_unchanged = all(new_rows[task] == old_rows[task] for task in old_rows)
    added_ids = sorted(set(new_rows) - set(old_rows))
    expected_added = pending_by_run[run_name]
    assert added_ids == expected_added, (run_name, len(added_ids), len(expected_added))

    summary = new['summary']
    assert summary['baseline_tasks'] == expected_eligible, (run_name, summary['baseline_tasks'])
    recomputed_install = sum(1 for r in new['task_results'] if r.get('first_no_ty_install_error') is not None)
    recomputed_static = sum(1 for r in new['task_results'] if r.get('first_ty_compatible') is not None)
    assert recomputed_install == summary['recover_no_ty_install_error_tasks'], (run_name, 'install mismatch')
    assert recomputed_static == summary['recover_ty_tasks'], (run_name, 'static mismatch')

    per_model.append({
        'run_name': run_name,
        'model': old['run_selection']['selected_run'].get('model_name'),
        'eligible': expected_eligible,
        'adopted': len(old_rows),
        'newly_executed': len(added_ids),
        'adopted_unchanged': adopted_unchanged,
        'install_recovered': recomputed_install,
        'static_recovered': recomputed_static,
    })

totals = {
    'eligible': sum(m['eligible'] for m in per_model),
    'adopted': sum(m['adopted'] for m in per_model),
    'newly_executed': sum(m['newly_executed'] for m in per_model),
    'install_recovered': sum(m['install_recovered'] for m in per_model),
    'static_recovered': sum(m['static_recovered'] for m in per_model),
    'old_install_recovered': 478,
    'old_static_recovered': 410,
}
assert totals['eligible'] == 1290 and totals['adopted'] == 1237 and totals['newly_executed'] == 53, totals
assert all(m['adopted_unchanged'] for m in per_model)

result = {'totals': totals, 'per_model': per_model}
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(totals, indent=2))
for m in per_model:
    inst = 100 * m['install_recovered'] / m['eligible']
    stat = 100 * m['static_recovered'] / m['eligible']
    print(f"{m['model']}: N={m['eligible']} ({m['newly_executed']} new) "
          f"inst {m['install_recovered']}/{m['eligible']} ({inst:.2f}%) "
          f"static {m['static_recovered']}/{m['eligible']} ({stat:.2f}%)")
print('ALL_CHECKS_PASSED')
