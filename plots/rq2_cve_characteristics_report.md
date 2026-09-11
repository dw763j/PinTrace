# RQ2 CVE-characteristics verification report

## Reproduction command

From the repository root:

`.venv/bin/python -m plots.rq2_cve_characteristics`

## Inputs and population

- severity: `outputs/analysis/rq2/vulnerability_severity_exact.csv`
- disclosure: `outputs/analysis/rq2/cve_disclosure_exact.csv`
- knowledge cutoffs: `resources/revision/model_knowledge_cutoff.json`
- population encoded in the corrected CSVs: D2 clean, mode=`inline` (VA), `resolution_origin=model_exact`, `constraint_valid=true`, `is_cve_associated=true`
- models: 10
- model order: same as Table tab:rq2_cve_association / `plots.model_display.MODEL_LABEL`

## Panel (a): maximum catalogued severity

- analysis unit: CVE-associated exact dependency occurrence
- 100% horizontal stacked bars
- UNKNOWN=194 and NONE=8 merged into Unscored=202
- Critical=894, High=2,193, Medium=1,767, Low=245, Unscored=202
- total occurrences=5,301 (asserted 5,301 before plotting)

- GPT-5.4: n=542; Critical=162, High=176, Medium=161, Low=31, Unscored=12
- Claude-Sonnet-4.6: n=586; Critical=81, High=242, Medium=201, Low=46, Unscored=16
- Gemini-3.1-Pro: n=597; Critical=69, High=269, Medium=200, Low=42, Unscored=17
- DeepSeek-V3.2: n=518; Critical=74, High=232, Medium=175, Low=21, Unscored=16
- Kimi-K2.5: n=650; Critical=104, High=265, Medium=234, Low=28, Unscored=19
- Qwen3.5-397B: n=570; Critical=58, High=247, Medium=205, Low=32, Unscored=28
- Qwen3-235B: n=424; Critical=70, High=156, Medium=163, Low=17, Unscored=18
- Qwen3-30B: n=188; Critical=32, High=93, Medium=49, Low=1, Unscored=13
- MiniMax-M2.5: n=719; Critical=167, High=297, Medium=205, Low=27, Unscored=23
- Llama-4-Scout: n=507; Critical=77, High=216, Medium=174, Low=0, Unscored=40

## Panel (b): unique advisories published by cutoff

- analysis unit: unique associated advisory per model
- repeated dependency occurrences do not increase weight
- provider-published cutoffs: 4 models, filled circles, no sensitivity interval
- self-reported cutoffs: 6 models, hollow diamonds, inner ±3-month and outer ±6-month sensitivity bounds
- these intervals are cutoff-date sensitivity bounds, not confidence intervals
- central rates range 26.28%--86.55%
- models with central rate >= 50%: 9/10

- GPT-5.4: 226/344 (65.70%), provider-published, cutoff=2025-08-31T00:00:00Z
- Claude-Sonnet-4.6: 327/473 (69.13%), provider-published, cutoff=2025-05-01T00:00:00Z
- Gemini-3.1-Pro: 380/590 (64.41%), provider-published, cutoff=2025-01-01T00:00:00Z
- DeepSeek-V3.2: 412/589 (69.95%), self-reported, cutoff=2025-05-01T00:00:00Z; ±3 mo 59.42--76.06%; ±6 mo 54.33--83.36%
- Kimi-K2.5: 144/548 (26.28%), self-reported, cutoff=2024-04-01T00:00:00Z; ±3 mo 17.70--43.80%; ±6 mo 6.02--51.82%
- Qwen3.5-397B: 399/461 (86.55%), self-reported, cutoff=2026-01-01T00:00:00Z; ±3 mo 78.52--100.00%; ±6 mo 73.32--100.00%
- Qwen3-235B: 395/612 (64.54%), self-reported, cutoff=2024-12-01T00:00:00Z; ±3 mo 56.86--66.99%; ±6 mo 43.30--75.82%
- Qwen3-30B: 200/379 (52.77%), self-reported, cutoff=2024-12-01T00:00:00Z; ±3 mo 44.85--55.67%; ±6 mo 26.91--67.02%
- MiniMax-M2.5: 365/725 (50.34%), self-reported, cutoff=2024-06-01T00:00:00Z; ±3 mo 43.31--62.07%; ±6 mo 32.28--68.83%
- Llama-4-Scout: 327/507 (64.50%), provider-published, cutoff=2024-08-01T00:00:00Z

## Outputs

- `plots/rq2_cve_characteristics.pdf`
- `plots/rq2_cve_characteristics.csv`
- `plots/rq2_cve_characteristics_report.md`
- `paper_results/figures/rq2_cve_characteristics.pdf`

## Acceptance checks

- severity totals Critical=894, High=2,193, Medium=1,767, Low=245, Unscored=202: pass
- UNKNOWN=194 and NONE=8 merge into Unscored: pass
- total occurrences=5,301: pass
- central disclosure rates 26.28%--86.55%: pass
- 9 of 10 central rates >= 50%: pass
- cutoff JSON source matches the disclosure CSV for every model: pass
- model order matches Table tab:rq2_cve_association: pass
