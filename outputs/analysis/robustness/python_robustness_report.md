# Cross-Python robustness verification report

## Data sources

- robustness runs (Python 3.8, 3.10, 3.14): `outputs/tosem_revision_20260812_python_1f1644e`
- Python 3.12 corrected canonical table: `outputs/evaluation/task_outcomes.jsonl`
- output directory: `outputs/analysis/robustness`
- classifiers: `stages.failure_classifier.classify_compatibility_layer`, `stages.failure_classifier.classify_ty_error`
- Python 3.8/3.10/3.14 records are classified from each `m5_compat_records.json`. Python 3.12 uses the stored `resolver_status`, `compatibility_layer`, `ty_compatible`, `ty_install_category`, and `bcb_status` fields.
- this script does not rerun model generation, dependency resolution, installation, static checking, or BigCodeBench tests

## Denominators and stage definitions

- request conditions: `requirements.txt` -> MR; `inline` -> VA
- BigCodeBench / D1: 813 tasks per model, condition, and Python version; All observations uses 8,130 observations
- PinTrace / D2: 1,000 tasks per model, condition, and Python version; All observations uses 10,000 observations
- each task--model observation contributes to a stage at most once
- `resolver_input`: resolver status is `resolved` or `failed`; `empty` is excluded
- `complete_lock`: resolver status is `resolved`
- `installation_completed`: `complete_lock` and compatibility layer is not `environment_build` or `infrastructure`
- `static_pass`: `complete_lock`, `installation_completed`, and `ty is_compatible == true`
- `dynamic_pass` (BigCodeBench only): `bcb_test.status == pass` over the full 813-task population, not conditioned on resolution, installation, or static validation
- end-to-end rates use the complete task--model population as the denominator
- stage-conditional rates: `complete_lock / resolver_input`, `installation_completed / complete_lock`, `static_pass / installation_completed`
- All observations sums model-level numerators and denominators before computing the percentage

## Completeness checks

- analysis units: 160 (expected 160)
- models per dataset/condition/Python version: 10
- raw model-name variants unified to paper labels: Claude-Sonnet-4.6, DeepSeek-V3.2, GPT-5.4, Gemini-3.1-Pro, Kimi-K2.5, Llama-4-Scout, MiniMax-M2.5, Qwen3-235B, Qwen3-235B-A22B-Instruct-2507, Qwen3-30B, Qwen3-30B-A3B-Instruct-2507, Qwen3.5-397B, Qwen3.5-397B-A17B, claude-sonnet-4-6, deepseek-v3.2, gemini-3.1-pro-preview, gpt-5.4, kimi-k2.5, meta-llama/llama-4-scout, qwen3.5-397b-a17b
- missing resolver status: 0
- unexpected resolver statuses: none
- duplicate task ids: 0
- unknown compatibility layers: none
- `ty_compatible=true` observations excluded from `static_pass` because they lack a complete lock or completed installation: 7609
- `missing_distutils` outside installation-after-lock failures: 0

## Python 3.12 regression

### PinTrace MR

- n: 10,000 (expected 10,000; pass)
- resolver_input: 9,235 (expected 9,235; pass)
- complete_lock: 8,234 (expected 8,234; pass)
- installation_completed: 7,779 (expected 7,779; pass)
- static_pass: 5,044 (expected 5,044; pass)

### PinTrace VA

- n: 10,000 (expected 10,000; pass)
- resolver_input: 8,868 (expected 8,868; pass)
- complete_lock: 6,889 (expected 6,889; pass)
- installation_completed: 5,274 (expected 5,274; pass)
- static_pass: 3,432 (expected 3,432; pass)

### BigCodeBench dynamic pass

- MR: 3,453/8,130 (expected 3,453/8,130; pass)
- VA: 1,335/8,130 (expected 1,335/8,130; pass)

## Aggregate counts and end-to-end percentages

### PinTrace

#### MR

- Python 3.8: n=10,000; resolver_input=9,231 (92.31%); complete_lock=8,121 (81.21%); installation_completed=7,899 (78.99%); static_pass=5,342 (53.42%); conditional: lock|input=87.98%, install|lock=97.27%, static|install=67.63%
- Python 3.10: n=10,000; resolver_input=9,231 (92.31%); complete_lock=8,340 (83.40%); installation_completed=8,178 (81.78%); static_pass=5,170 (51.70%); conditional: lock|input=90.35%, install|lock=98.06%, static|install=63.22%
- Python 3.12: n=10,000; resolver_input=9,235 (92.35%); complete_lock=8,234 (82.34%); installation_completed=7,779 (77.79%); static_pass=5,044 (50.44%); conditional: lock|input=89.16%, install|lock=94.47%, static|install=64.84%
- Python 3.14: n=10,000; resolver_input=9,237 (92.37%); complete_lock=8,049 (80.49%); installation_completed=7,321 (73.21%); static_pass=4,779 (47.79%); conditional: lock|input=87.14%, install|lock=90.96%, static|install=65.28%

#### VA

- Python 3.8: n=10,000; resolver_input=8,864 (88.64%); complete_lock=6,093 (60.93%); installation_completed=5,967 (59.67%); static_pass=4,041 (40.41%); conditional: lock|input=68.74%, install|lock=97.93%, static|install=67.72%
- Python 3.10: n=10,000; resolver_input=8,864 (88.64%); complete_lock=7,215 (72.15%); installation_completed=7,081 (70.81%); static_pass=4,766 (47.66%); conditional: lock|input=81.40%, install|lock=98.14%, static|install=67.31%
- Python 3.12: n=10,000; resolver_input=8,868 (88.68%); complete_lock=6,889 (68.89%); installation_completed=5,274 (52.74%); static_pass=3,432 (34.32%); conditional: lock|input=77.68%, install|lock=76.56%, static|install=65.07%
- Python 3.14: n=10,000; resolver_input=8,869 (88.69%); complete_lock=6,836 (68.36%); installation_completed=4,184 (41.84%); static_pass=2,667 (26.67%); conditional: lock|input=77.08%, install|lock=61.21%, static|install=63.74%

### BigCodeBench

#### MR

- Python 3.8: n=8,130; resolver_input=8,126 (99.95%); complete_lock=7,875 (96.86%); installation_completed=7,865 (96.74%); static_pass=7,272 (89.45%); dynamic_pass=3,723 (45.79%); conditional: lock|input=96.91%, install|lock=99.87%, static|install=92.46%
- Python 3.10: n=8,130; resolver_input=8,126 (99.95%); complete_lock=8,031 (98.78%); installation_completed=8,000 (98.40%); static_pass=6,394 (78.65%); dynamic_pass=3,689 (45.38%); conditional: lock|input=98.83%, install|lock=99.61%, static|install=79.92%
- Python 3.12: n=8,130; resolver_input=8,126 (99.95%); complete_lock=8,017 (98.61%); installation_completed=7,485 (92.07%); static_pass=6,772 (83.30%); dynamic_pass=3,453 (42.47%); conditional: lock|input=98.66%, install|lock=93.36%, static|install=90.47%
- Python 3.14: n=8,130; resolver_input=8,126 (99.95%); complete_lock=7,987 (98.24%); installation_completed=7,333 (90.20%); static_pass=6,658 (81.89%); dynamic_pass=3,389 (41.69%); conditional: lock|input=98.29%, install|lock=91.81%, static|install=90.80%

#### VA

- Python 3.8: n=8,130; resolver_input=8,130 (100.00%); complete_lock=5,884 (72.37%); installation_completed=5,876 (72.28%); static_pass=5,507 (67.74%); dynamic_pass=2,756 (33.90%); conditional: lock|input=72.37%, install|lock=99.86%, static|install=93.72%
- Python 3.10: n=8,130; resolver_input=8,130 (100.00%); complete_lock=7,892 (97.07%); installation_completed=7,870 (96.80%); static_pass=7,154 (88.00%); dynamic_pass=2,605 (32.04%); conditional: lock|input=97.07%, install|lock=99.72%, static|install=90.90%
- Python 3.12: n=8,130; resolver_input=8,130 (100.00%); complete_lock=7,846 (96.51%); installation_completed=3,341 (41.09%); static_pass=2,947 (36.25%); dynamic_pass=1,335 (16.42%); conditional: lock|input=96.51%, install|lock=42.58%, static|install=88.21%
- Python 3.14: n=8,130; resolver_input=8,130 (100.00%); complete_lock=7,845 (96.49%); installation_completed=1,873 (23.04%); static_pass=1,709 (21.02%); dynamic_pass=735 (9.04%); conditional: lock|input=96.49%, install|lock=23.88%, static|install=91.24%

## Percentage-point changes from Python 3.10

### PinTrace

#### MR

- Complete lock, 3.10 to 3.12: -1.06 pp (decrease); 10/10 models move in the same direction
- Complete lock, 3.10 to 3.14: -2.91 pp (decrease); 10/10 models move in the same direction
- Installation completed, 3.10 to 3.12: -3.99 pp (decrease); 10/10 models move in the same direction
- Installation completed, 3.10 to 3.14: -8.57 pp (decrease); 10/10 models move in the same direction
- Static validation passed, 3.10 to 3.12: -1.26 pp (decrease); 6/10 models move in the same direction
- Static validation passed, 3.10 to 3.14: -3.91 pp (decrease); 10/10 models move in the same direction

#### VA

- Complete lock, 3.10 to 3.12: -3.26 pp (decrease); 10/10 models move in the same direction
- Complete lock, 3.10 to 3.14: -3.79 pp (decrease); 10/10 models move in the same direction
- Installation completed, 3.10 to 3.12: -18.07 pp (decrease); 10/10 models move in the same direction
- Installation completed, 3.10 to 3.14: -28.97 pp (decrease); 10/10 models move in the same direction
- Static validation passed, 3.10 to 3.12: -13.34 pp (decrease); 10/10 models move in the same direction
- Static validation passed, 3.10 to 3.14: -20.99 pp (decrease); 10/10 models move in the same direction

### BigCodeBench

#### MR

- Complete lock, 3.10 to 3.12: -0.17 pp (decrease); 4/10 models move in the same direction
- Complete lock, 3.10 to 3.14: -0.54 pp (decrease); 10/10 models move in the same direction
- Installation completed, 3.10 to 3.12: -6.33 pp (decrease); 9/10 models move in the same direction
- Installation completed, 3.10 to 3.14: -8.20 pp (decrease); 10/10 models move in the same direction
- Static validation passed, 3.10 to 3.12: +4.65 pp (increase); 7/10 models move in the same direction
- Static validation passed, 3.10 to 3.14: +3.25 pp (increase); 7/10 models move in the same direction
- Dynamic test passed, 3.10 to 3.12: -2.90 pp (decrease); 10/10 models move in the same direction
- Dynamic test passed, 3.10 to 3.14: -3.69 pp (decrease); 10/10 models move in the same direction

#### VA

- Complete lock, 3.10 to 3.12: -0.57 pp (decrease); 10/10 models move in the same direction
- Complete lock, 3.10 to 3.14: -0.58 pp (decrease); 10/10 models move in the same direction
- Installation completed, 3.10 to 3.12: -55.71 pp (decrease); 10/10 models move in the same direction
- Installation completed, 3.10 to 3.14: -73.76 pp (decrease); 10/10 models move in the same direction
- Static validation passed, 3.10 to 3.12: -51.75 pp (decrease); 9/10 models move in the same direction
- Static validation passed, 3.10 to 3.14: -66.97 pp (decrease); 10/10 models move in the same direction
- Dynamic test passed, 3.10 to 3.12: -15.62 pp (decrease); 10/10 models move in the same direction
- Dynamic test passed, 3.10 to 3.14: -23.00 pp (decrease); 10/10 models move in the same direction

## Distutils contribution to installation failures

Only observations that produced a complete lock and then failed installation enter `installation_failure_count` and `missing_distutils_count`. `missing_distutils` is the `ty_install_category` returned by `classify_ty_error`. Removing this error would not imply that the task would pass static or dynamic validation.

### PinTrace MR

- Python 3.8: missing_distutils=0 / installation_failure=222 (0.00% of installation failures); 0/10,000 (0.00% of the complete population); complete_lock=8,121
- Python 3.10: missing_distutils=2 / installation_failure=162 (1.23% of installation failures); 2/10,000 (0.02% of the complete population); complete_lock=8,340
- Python 3.12: missing_distutils=184 / installation_failure=455 (40.44% of installation failures); 184/10,000 (1.84% of the complete population); complete_lock=8,234
- Python 3.14: missing_distutils=105 / installation_failure=728 (14.42% of installation failures); 105/10,000 (1.05% of the complete population); complete_lock=8,049

### PinTrace VA

- Python 3.8: missing_distutils=0 / installation_failure=126 (0.00% of installation failures); 0/10,000 (0.00% of the complete population); complete_lock=6,093
- Python 3.10: missing_distutils=2 / installation_failure=134 (1.49% of installation failures); 2/10,000 (0.02% of the complete population); complete_lock=7,215
- Python 3.12: missing_distutils=800 / installation_failure=1,615 (49.54% of installation failures); 800/10,000 (8.00% of the complete population); complete_lock=6,889
- Python 3.14: missing_distutils=676 / installation_failure=2,652 (25.49% of installation failures); 676/10,000 (6.76% of the complete population); complete_lock=6,836

### BigCodeBench MR

- Python 3.8: missing_distutils=0 / installation_failure=10 (0.00% of installation failures); 0/8,130 (0.00% of the complete population); complete_lock=7,875
- Python 3.10: missing_distutils=3 / installation_failure=31 (9.68% of installation failures); 3/8,130 (0.04% of the complete population); complete_lock=8,031
- Python 3.12: missing_distutils=384 / installation_failure=532 (72.18% of installation failures); 384/8,130 (4.72% of the complete population); complete_lock=8,017
- Python 3.14: missing_distutils=383 / installation_failure=654 (58.56% of installation failures); 383/8,130 (4.71% of the complete population); complete_lock=7,987

### BigCodeBench VA

- Python 3.8: missing_distutils=0 / installation_failure=8 (0.00% of installation failures); 0/8,130 (0.00% of the complete population); complete_lock=5,884
- Python 3.10: missing_distutils=5 / installation_failure=22 (22.73% of installation failures); 5/8,130 (0.06% of the complete population); complete_lock=7,892
- Python 3.12: missing_distutils=2,570 / installation_failure=4,505 (57.05% of installation failures); 2,570/8,130 (31.61% of the complete population); complete_lock=7,846
- Python 3.14: missing_distutils=2,537 / installation_failure=5,972 (42.48% of installation failures); 2,537/8,130 (31.21% of the complete population); complete_lock=7,845

## Differences from the previous robustness output

The previous script inferred installation as `total - install_error - infra_error` and counted every `ty_compatible=true` observation as a static pass. That definition does not require `resolver.status == resolved`, so empty no-dependency observations entered both the installation and static-pass numerators. Installation rates could therefore exceed resolution rates, and the static rates were not comparable to the corrected RQ3 end-to-end definition.

All observations previously averaged model-level percentages. This revision sums counts first. Because every model has the same task count, the percentages can match, but the pooled denominators are now auditable.

The previous `distutils_sensitivity.csv` metric excluded `missing_distutils` observations from the denominator and still counted no-dependency static passes. This revision reports only the contribution of `missing_distutils` to post-lock installation failures.

Complete-lock counts should match the previous `resolved` counts. Installation and static-pass counts are lower once empty resolver observations are removed.

- complete_lock matches the previous `resolved` count in every overlapping cell

All-observations installation and static-pass revisions:

- PinTrace MR Python 3.8: installation 8,668 -> 7,899 (86.68% -> 78.99%); static 6,110 -> 5,342 (61.10% -> 53.42%)
- PinTrace MR Python 3.10: installation 8,947 -> 8,178 (89.47% -> 81.78%); static 5,938 -> 5,170 (59.38% -> 51.70%)
- PinTrace MR Python 3.12: installation 8,544 -> 7,779 (85.44% -> 77.79%); static 5,808 -> 5,044 (58.08% -> 50.44%)
- PinTrace MR Python 3.14: installation 8,084 -> 7,321 (80.84% -> 73.21%); static 5,541 -> 4,779 (55.41% -> 47.79%)
- PinTrace VA Python 3.8: installation 7,103 -> 5,967 (71.03% -> 59.67%); static 5,177 -> 4,041 (51.77% -> 40.41%)
- PinTrace VA Python 3.10: installation 8,217 -> 7,081 (82.17% -> 70.81%); static 5,902 -> 4,766 (59.02% -> 47.66%)
- PinTrace VA Python 3.12: installation 6,406 -> 5,274 (64.06% -> 52.74%); static 4,564 -> 3,432 (45.64% -> 34.32%)
- PinTrace VA Python 3.14: installation 5,315 -> 4,184 (53.15% -> 41.84%); static 3,798 -> 2,667 (37.98% -> 26.67%)
- BigCodeBench MR Python 3.8: installation 7,869 -> 7,865 (96.79% -> 96.74%); static 7,275 -> 7,272 (89.48% -> 89.45%)
- BigCodeBench MR Python 3.10: installation 8,004 -> 8,000 (98.45% -> 98.40%); static 6,397 -> 6,394 (78.68% -> 78.65%)
- BigCodeBench MR Python 3.12: installation 7,489 -> 7,485 (92.12% -> 92.07%); static 6,775 -> 6,772 (83.33% -> 83.30%)
- BigCodeBench MR Python 3.14: installation 7,337 -> 7,333 (90.25% -> 90.20%); static 6,661 -> 6,658 (81.93% -> 81.89%)
- BigCodeBench VA Python 3.8: installation 5,876 -> 5,876 (72.28% -> 72.28%); static 5,507 -> 5,507 (67.74% -> 67.74%)
- BigCodeBench VA Python 3.10: installation 7,870 -> 7,870 (96.80% -> 96.80%); static 7,154 -> 7,154 (88.00% -> 88.00%)
- BigCodeBench VA Python 3.12: installation 3,341 -> 3,341 (41.09% -> 41.09%); static 2,947 -> 2,947 (36.25% -> 36.25%)
- BigCodeBench VA Python 3.14: installation 1,873 -> 1,873 (23.04% -> 23.04%); static 1,709 -> 1,709 (21.02% -> 21.02%)

## Reproduction command

```bash
python -m scripts.generate_robustness_results --robustness-root outputs/tosem_revision_20260812_python_1f1644e --py312-task-outcomes outputs/evaluation/task_outcomes.jsonl --output outputs/analysis/robustness
```

