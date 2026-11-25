import os
"""Summarize within-package top-three shares from the checked RQ1 figure data."""
import csv
from pathlib import Path

root = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
source = root / 'analysis/rq1_concentration_checked.csv'
with source.open(encoding='utf-8-sig') as handle:
    rows = list(csv.DictReader(handle))

summary = []
for row in rows:
    counts = [int(row[key]) for key in ['rank1', 'rank2', 'rank3']]
    total = int(row['N'])
    assert counts[0] == int(row['modal_count'])
    assert sum(counts) + int(row['other']) == total
    summary.append({
        'package': row['package'],
        'condition': row['condition'],
        'exact_declarations': total,
        'top_three_declarations': sum(counts),
        'top_three_pct': 100 * sum(counts) / total,
        'majority': sum(counts) > total / 2,
    })

destination = root / 'analysis/rq1_top_three_concentration.csv'
with destination.open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

for condition in ['MR', 'VA']:
    subset = [row for row in summary if row['condition'] == condition]
    assert len(subset) == len({row['package'] for row in subset}) == 10
    majority = sum(row['majority'] for row in subset)
    print(f'{condition}: top three versions account for a majority in {majority}/10 packages')
