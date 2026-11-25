import os
"""Recompute paired contrasts after excluding responses affected by source corrections."""

import collections
import hashlib
import json

import numpy as np
import pandas as pd

from reanalyze import ROOT, OUT, SEED, rows, csvout, dump


def main():
    source = ROOT / 'inputs/main/source_correction_flags.jsonl'
    corrections = list(rows(source))
    flags = pd.read_csv(OUT / 'response_flags.csv', low_memory=False)
    assert len(corrections) == len(flags) == 36260
    affected = {(row['track'], row['model'], row['task_id']) for row in corrections if row['source_changed']}
    results = []
    rng = np.random.default_rng(SEED)
    populations = [
        ('PinTrace complete', flags.track == 'd2', ['static']),
        ('PinTrace clean', (flags.track == 'd2') & flags.clean, ['static', 'exact_associated', 'exact_published_by_cutoff']),
        ('BigCodeBench', flags.track == 'd1', ['dynamic']),
    ]
    for population, mask, metrics in populations:
        selected = flags[mask].copy()
        retained = selected[[tuple(row) not in affected for row in selected[['track', 'model', 'task_id']].itertuples(index=False, name=None)]]
        task_ids = sorted(retained.task_id.unique())
        by_condition = {mode: retained[retained['mode'] == mode] for mode in ['requirements.txt', 'inline']}
        mr_keys = set(by_condition['requirements.txt'][['task_id', 'model']].itertuples(index=False, name=None))
        va_keys = set(by_condition['inline'][['task_id', 'model']].itertuples(index=False, name=None))
        assert mr_keys == va_keys
        counts_per_task = by_condition['inline'].groupby('task_id').size().reindex(task_ids).to_numpy()
        draws = rng.multinomial(len(task_ids), np.ones(len(task_ids)) / len(task_ids), size=10000)
        denominators = draws @ counts_per_task
        for metric in metrics:
            mr, va = [by_condition[mode].groupby('task_id')[metric].sum().reindex(task_ids).to_numpy() for mode in ['requirements.txt', 'inline']]
            intervals = np.quantile(100 * (draws @ (va - mr)) / denominators, [.025, .975])
            results.append(dict(population=population, metric=metric, retained_pairs=int(counts_per_task.sum()),
                                excluded_pairs=len(selected) // 2 - int(counts_per_task.sum()), retained_tasks=len(task_ids),
                                MR_n=int(mr.sum()), VA_n=int(va.sum()), MR_pct=100 * mr.sum() / counts_per_task.sum(),
                                VA_pct=100 * va.sum() / counts_per_task.sum(), delta_pp=100 * (va-mr).sum() / counts_per_task.sum(),
                                delta_ci_low=intervals[0], delta_ci_high=intervals[1]))
    csvout('extraction_sensitivity.csv', results)
    dump('extraction_impact.json', {
        'source_changed_responses': sum(row['source_changed'] for row in corrections),
        'excluded_pairs': len(affected),
        'changed_responses_by_track': dict(collections.Counter(row['track'] for row in corrections if row['source_changed'])),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'flags_sha256': hashlib.sha256((OUT / 'response_flags.csv').read_bytes()).hexdigest(),
        'seed': SEED, 'replicates': 10000,
        'population_rule': 'Exclude both conditions of each task-model pair when either saved response has a corrected code selection, manifest, package constraint input, or author-adjudicated terminal state.',
        'interval_method': 'Paired task-cluster percentile bootstrap retaining each task\'s observed model count in the numerator and denominator.',
    })
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
