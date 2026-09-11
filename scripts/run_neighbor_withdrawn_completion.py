"""Complete the neighboring-version experiment for the 53 observations that
became eligible after excluding withdrawn OSV advisories (FIXME 1 follow-up).

Run on do from the project root with the project virtualenv:

    python -m scripts.run_neighbor_withdrawn_completion

What this driver does, in order:

1.  Rebuild the OSV index with records carrying a nonempty ``withdrawn`` field
    skipped (the same withdrawal-only filter as the two audits in this
    directory; no other matching change, FIXME 2 untouched).
2.  Patch, inside this process only:
      - ``run_neighbor_version_experiment._task_has_vuln`` so baseline safety
        is judged against the active-advisory index (this admits the 53 newly
        eligible observations), and
      - ``run_neighbor_version_experiment.load_osv_index`` so candidate-level
        vulnerability checks use the same active index.
    The original script file is not modified.
3.  Copy the ten executed per-model result files into
    ``outputs/analysis/neighbor/models``
    so ``--resume`` adopts the executed 1,237 task results unchanged and only
    the pending observations are executed.
4.  Assert the pre-flight expectations (per-model eligible totals and the
    executed cohort sizes) before launching.
5.  Invoke the experiment's own ``main()`` with the frozen configuration
    recorded in the executed results (resolver hash inputs, reference date,
    neighborhood and stop-policy parameters), ``--all-models --resume``,
    parallel 6 x 2.

Outputs land only in the new completion directory; the executed outputs under
outputs/tosem_revision_20260812_neighbor_1f1644e are never written to.
"""
import inspect
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from paths import ANALYSIS_NEIGHBOR, EVALUATION, OSV_INDEX
from stages import vuln_checker as vc
import run_neighbor_version_experiment as nve

OSV = OSV_INDEX
OLD_RESULTS_DIR = ROOT / 'outputs/tosem_revision_20260812_neighbor_1f1644e/d2'
NEW_RESULTS_DIR = ANALYSIS_NEIGHBOR / 'models'

EXPECTED_ELIGIBLE = {
    'd2_DeepSeek-V3.2_blind_inline': 163,
    'd2_MiniMax-M2.5_blind_inline': 165,
    'd2_Qwen3-235B-A22B-Instruct-2507_blind_inline': 155,
    'd2_Qwen3-30B-A3B-Instruct-2507_blind_inline': 80,
    'd2_Qwen3.5-397B-A17B_blind_inline': 166,
    'd2_claude-sonnet-4-6_blind_inline': 78,
    'd2_gemini-3.1-pro-preview_blind_inline': 145,
    'd2_gpt-5.4_blind_inline': 54,
    'd2_kimi-k2.5_blind_inline': 133,
    'd2_meta-llama_llama-4-scout_blind_inline': 151,
}

ARGV = [
    'run_neighbor_version_experiment.py',
    '--dataset', 'd2',
    '--outputs-d2', 'outputs/tosem_revision_20260811_1f1644e/main_py312/d2',
    '--canonical-root', str(EVALUATION),
    '--analysis-output-root', str(NEW_RESULTS_DIR),
    '--origin', 'model_exact',
    '--python-version', '3.12',
    '--target-mode', 'inline',
    '--select-by', 'safe_ty_incompat',
    '--baseline', 'safe_ty_install_error',
    '--neighbor-n', '3',
    '--max-candidates-k', '40',
    '--max-searchable-libs', '12',
    '--stop-policy', 'first_pass',
    '--focused-neighbor-n', '10',
    '--reference-date', '2026-02-01T00:00:00+00:00',
    '--python-platform', 'x86_64-unknown-linux-gnu',
    '--uv-bin', 'uv',
    '--resolver-timeout', '600',
    '--bcb-timeout', '180',
    '--uv-install-timeout', '600',
    '--parallel', '6',
    '--parallel-models', '2',
    '--specified-only',
    '--focus-install-failed-tpl',
    '--all-models',
    '--resume',
]

# ---------------------------------------------------------------- active index
old_source = inspect.getsource(vc.load_osv_index)
needle = '            data = json.load(f)\n'
assert old_source.count(needle) == 1
patched_source = old_source.replace(needle, needle + '        if data.get("withdrawn"):\n            continue\n')
namespace = dict(vc.__dict__)
exec(compile(patched_source, '<withdrawal-only-index>', 'exec'), namespace)
active_index = namespace['load_osv_index'](str(OSV))


@lru_cache(None)
def still_vulnerable(package, version):
    osv_ids, cve_ids = vc._collect_cve_ids_for_package_version(active_index.get(package, {}), version)
    return bool(osv_ids or cve_ids)


def task_has_vuln_active(record):
    for finding in record.get('vuln_findings', []) or []:
        if not finding.get('is_vulnerable'):
            continue
        package = str(finding.get('pypi_name') or '').lower().replace('_', '-')
        version = str(finding.get('version') or finding.get('resolved_version') or '')
        if still_vulnerable(package, version):
            return True
    return False


nve._task_has_vuln = task_has_vuln_active
nve.load_osv_index = lambda path: active_index

# ------------------------------------------------------------------ pre-flight
NEW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
old_files = sorted(OLD_RESULTS_DIR.glob('safe_ty_install_error_inline_*.json'))
assert len(old_files) == 10, [p.name for p in old_files]

total_tested = 0
total_eligible = 0
for path in old_files:
    data = json.loads(path.read_text(encoding='utf-8'))
    selected = data['run_selection']['selected_run']
    run_name = selected['run_name']
    assert selected['mode'] == 'inline' and selected['python_tag'] == 'py312', run_name
    assert run_name in EXPECTED_ELIGIBLE, run_name
    outputs_root = Path(data['experiment_config']['outputs_root'])
    if not outputs_root.is_absolute():
        outputs_root = ROOT / outputs_root
    raw = json.loads((outputs_root / 'inline' / run_name / 'py312' / 'm5_compat_records.json')
                     .read_text(encoding='utf-8'))
    eligible = sum(
        1 for record in raw
        if nve._has_ty_install_error(((record.get('compat_results') or {}).get('ty') or {}))
        and nve._record_has_direct_origin(record, 'model_exact')
        and not task_has_vuln_active(record)
    )
    executed = len(data['task_results'])
    assert eligible == EXPECTED_ELIGIBLE[run_name], (run_name, eligible, EXPECTED_ELIGIBLE[run_name])
    assert eligible - executed == {'d2_DeepSeek-V3.2_blind_inline': 7, 'd2_MiniMax-M2.5_blind_inline': 8,
                                   'd2_Qwen3-235B-A22B-Instruct-2507_blind_inline': 6,
                                   'd2_Qwen3-30B-A3B-Instruct-2507_blind_inline': 3,
                                   'd2_Qwen3.5-397B-A17B_blind_inline': 9,
                                   'd2_claude-sonnet-4-6_blind_inline': 2,
                                   'd2_gemini-3.1-pro-preview_blind_inline': 5,
                                   'd2_gpt-5.4_blind_inline': 2, 'd2_kimi-k2.5_blind_inline': 4,
                                   'd2_meta-llama_llama-4-scout_blind_inline': 7}[run_name], (run_name, eligible, executed)
    destination = NEW_RESULTS_DIR / path.name
    if not destination.exists():
        shutil.copy2(path, destination)
    total_tested += executed
    total_eligible += eligible
    print(f'[pre-flight] {run_name}: executed={executed} eligible_after={eligible} pending={eligible - executed}')

assert (total_tested, total_eligible) == (1237, 1290), (total_tested, total_eligible)
print(f'[pre-flight] totals: executed={total_tested} eligible_after={total_eligible} pending=53')
print('[pre-flight] resume sources copied to', NEW_RESULTS_DIR)

with (ANALYSIS_NEIGHBOR / 'driver_config.json').open('w', encoding='utf-8') as handle:
    json.dump({
        'driver': Path(__file__).name,
        'patches': ['nve._task_has_vuln -> active-advisory check',
                    'nve.load_osv_index -> withdrawal-only index'],
        'argv': ARGV,
        'expected_eligible': EXPECTED_ELIGIBLE,
        'resume_sources': [p.name for p in old_files],
    }, handle, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------- launch
print('[driver] launching with argv:', ' '.join(ARGV[1:]), flush=True)
sys.argv = ARGV
nve.main()
print('[driver] DRIVER_DONE')
