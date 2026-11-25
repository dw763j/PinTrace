import os
"""Generate the model-level exact-advice advisory table from corrected counts."""
import csv
import hashlib
import json
from pathlib import Path

MODELS = (
    ('gpt-5.4', 'GPT-5.4'),
    ('claude-sonnet-4-6', 'Claude Sonnet 4.6'),
    ('gemini-3.1-pro-preview', 'Gemini 3.1 Pro'),
    ('deepseek-v3.2', 'DeepSeek V3.2'),
    ('kimi-k2.5', 'Kimi K2.5'),
    ('qwen3.5-397b-a17b', 'Qwen3.5 397B'),
    ('qwen3-235b-a22b-instruct-2507', 'Qwen3 235B'),
    ('qwen3-30b-a3b-instruct-2507', 'Qwen3 30B'),
    ('minimax-m2.5', 'MiniMax M2.5'),
    ('meta-llama/llama-4-scout', 'Llama 4 Scout'),
    ('All observations', 'All observations'),
)


def write_advisory_model_table(root, source_rows=None):
    root = Path(root)
    source = root / 'analysis/security_funnels.csv'
    if source_rows is None:
        with source.open(encoding='utf-8-sig', newline='') as handle:
            source_rows = list(csv.DictReader(handle))
    selected = [row for row in source_rows if row['kind'] == 'exact'
                and row['security_scope'] == 'published_by_cutoff']
    index = {(row['model'], row['mode']): row for row in selected}
    assert len(index) == len(selected) == 22
    records, rows = [], []
    for model, name in MODELS:
        record = {'model': model, 'name': name}
        cells = [name]
        for mode, condition in [('requirements.txt', 'MR'), ('inline', 'VA')]:
            row = index[model, mode]
            n, s, p, v = (int(row[key]) for key in ['N', 'S', 'P', 'V'])
            assert 0 < p <= s <= n and 0 <= v <= p
            assert n == (9270 if model == 'All observations' else 927)
            conditional = 100 * v / p
            assert abs(conditional - float(row['conditional_pct'])) < 1e-9
            assert abs(100 * v / n - float(row['population_pct'])) < 1e-9
            record[condition] = dict(N=n, S=s, P=p, V=v, conditional_pct=conditional)
            cells.extend([f'{p:,}', f'{v:,}', f'{conditional:.2f}'])
        records.append(record)
        rows.append(cells)
    for condition in ['MR', 'VA']:
        for key in ['N', 'S', 'P', 'V']:
            assert sum(row[condition][key] for row in records[:-1]) == records[-1][condition][key]
    assert all(row['VA']['conditional_pct'] > 40 for row in records[:-1])

    headers = ['Model', 'MR P', 'MR V', 'MR conditional (%)',
               'VA P', 'VA V', 'VA conditional (%)']
    description = ('Exact-advice advisory associations in clean PinTrace using advisories published by the release cutoff. '
                   'P counts responses with at least one individually satisfiable exact constraint; V counts those with a match. '
                   'The conditional rate is 100 V/P. Each model has 927 responses per condition. '
                   'All observations sums counts across ten models before computing rates.')
    markdown = description + '\n\n| ' + ' | '.join(headers) + ' |\n'
    markdown += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    markdown += ''.join('| ' + ' | '.join(row) + ' |\n' for row in rows)

    latex = r'''% Evidence: analysis/security_funnels.csv, kind=exact, security_scope=published_by_cutoff.
% Reproduction: python scripts/prepare_advisory_model_table.py; sum model counts before computing All observations.
\begin{table}[htbp]
\centering\footnotesize
\caption{\revised{Advisory associations in exact advice by model on clean \myso, using advisories published by the release cutoff.
Each model has 927 responses per condition.
$P_{q,e}$ counts responses with individually satisfiable exact advice, $V_{q,e}$ counts those with an advisory match, and $\nu^\star_{q,e}=V_{q,e}/P_{q,e}$.
All observations sums counts across the ten models before computing rates.}}
\label{tab:rq2_security}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrrrr@{}}
\toprule
\textbf{Model} & $P_{\mathrm{MR},e}$ & $V_{\mathrm{MR},e}$ & $\nu^\star_{\mathrm{MR},e}$ (\%) & $P_{\mathrm{VA},e}$ & $V_{\mathrm{VA},e}$ & $\nu^\star_{\mathrm{VA},e}$ (\%) \\
\midrule
'''
    for cells in rows[:-1]:
        latex += ' & '.join(cells) + r' \\' + '\n'
    latex += '\\midrule\n'
    latex += r'\textbf{All observations}' + ' & ' + ' & '.join(rows[-1][1:]) + r' \\' + '\n'
    latex += '\\bottomrule\n\\end{tabularx}\n\\end{table}\n'

    output = root / 'tables'
    output.mkdir(exist_ok=True)
    (output / 'rq2_security.md').write_text(markdown, encoding='utf-8')
    (output / 'rq2_security.tex').write_text(latex, encoding='utf-8')
    with (output / 'rq2_security_by_model.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'condition', 'N', 'S', 'P', 'V', 'conditional_pct'])
        for record in records:
            for condition in ['MR', 'VA']:
                writer.writerow([record['model'], condition] +
                                [record[condition][key] for key in ['N', 'S', 'P', 'V', 'conditional_pct']])
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    population='927 clean PinTrace tasks, ten models', unit='response',
                    kind='exact', security_scope='published_by_cutoff', records=records, rows=rows)
    (output / 'rq2_security_by_model.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return dict(manifest=manifest, rows=rows, latex=latex)


if __name__ == '__main__':
    result = write_advisory_model_table(Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1])))
    print('Generated ten model rows and All observations from checked response counts.')
