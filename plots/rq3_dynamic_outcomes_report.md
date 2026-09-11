# RQ3 dynamic-execution outcomes verification report

## Input

- path: `outputs/evaluation/task_outcomes.jsonl`
- SHA-256: `88750aa52441dbf1db1eda0f208a60eb0aa3290fed6871b849c8958d9998bba7`
- expected SHA-256: `88750aa52441dbf1db1eda0f208a60eb0aa3290fed6871b849c8958d9998bba7`
- loaded records: 16,260 after filters; full file is hashed above

## Filters

- source: `task_outcomes.jsonl` from the corrected canonical tables
- do not use the obsolete 724-task OSV subset
- do not use any former BigCodeBench summary table
- `track == "d1"`
- `python_version == "3.12"`
- no filter on OSV coverage, resolver status, compatibility layer, constraint presence, or static-validation outcome
- analysis unit: one task--model observation
- uniqueness key: `(track, mode, model, task_id)`

## Status definitions

- dynamic pass: `bcb_status == "pass"`
- dynamic non-pass: `bcb_status in {"fail", "error", "runner_error"}`
- `bcb_status == "fail"` is reported as test failure; the runner combines unittest failures and unittest errors under this status
- `bcb_status == "error"` is an execution error
- `bcb_status == "runner_error"` is a runner error
- static pass: `ty_compatible is true`
- static fail: `ty_compatible is false`
- missing or unexpected `ty_compatible` or `bcb_status` values abort the script

## Mode mapping

- `requirements.txt` -> `MR`
- `inline` -> `VA`

## Model-name mapping

Display names and order come from `plots.model_display.MODEL_LABEL` via `paper_display_label`.
The order matches `plots/rq3_failure_layers.py`.
No new aliases are introduced.

- `gpt-5.4` -> `GPT-5.4`
- `claude-sonnet-4-6` -> `Claude-Sonnet-4.6`
- `gemini-3.1-pro-preview` -> `Gemini-3.1-Pro`
- `DeepSeek-V3.2` -> `DeepSeek-V3.2`
- `kimi-k2.5` -> `Kimi-K2.5`
- `Qwen3.5-397B-A17B` -> `Qwen3.5-397B`
- `Qwen3-235B-A22B-Instruct-2507` -> `Qwen3-235B`
- `Qwen3-30B-A3B-Instruct-2507` -> `Qwen3-30B`
- `MiniMax-M2.5` -> `MiniMax-M2.5`
- `meta-llama_llama-4-scout` -> `Llama-4-Scout`
- `meta-llama/llama-4-scout` is accepted through `paper_display_label` and displayed as `Llama-4-Scout`
- D1 records spell DeepSeek as `DeepSeek-V3.2` under MR and `deepseek-v3.2` under VA; both map to `DeepSeek-V3.2`
- D1 records spell Llama as `meta-llama_llama-4-scout` or `meta-llama/llama-4-scout`; both map to `Llama-4-Scout`

- figure order: GPT-5.4, Claude-Sonnet-4.6, Gemini-3.1-Pro, DeepSeek-V3.2, Kimi-K2.5, Qwen3.5-397B, Qwen3-235B, Qwen3-30B, MiniMax-M2.5, Llama-4-Scout, then `All observations`
- `All observations` sums raw counts across models before computing rates

## Per-model dynamic outcomes

### MR

- GPT-5.4 (`gpt-5.4`): n=813; pass=401 (49.32%); test failure=406; execution error=5; runner error=1
- Claude-Sonnet-4.6 (`claude-sonnet-4-6`): n=813; pass=404 (49.69%); test failure=386; execution error=23; runner error=0
- Gemini-3.1-Pro (`gemini-3.1-pro-preview`): n=813; pass=422 (51.91%); test failure=379; execution error=12; runner error=0
- DeepSeek-V3.2 (`DeepSeek-V3.2`): n=813; pass=263 (32.35%); test failure=307; execution error=242; runner error=1
- Kimi-K2.5 (`kimi-k2.5`): n=813; pass=352 (43.30%); test failure=440; execution error=20; runner error=1
- Qwen3.5-397B (`Qwen3.5-397B-A17B`): n=813; pass=410 (50.43%); test failure=397; execution error=5; runner error=1
- Qwen3-235B (`Qwen3-235B-A22B-Instruct-2507`): n=813; pass=316 (38.87%); test failure=336; execution error=159; runner error=2
- Qwen3-30B (`Qwen3-30B-A3B-Instruct-2507`): n=813; pass=257 (31.61%); test failure=354; execution error=201; runner error=1
- MiniMax-M2.5 (`MiniMax-M2.5`): n=813; pass=361 (44.40%); test failure=432; execution error=19; runner error=1
- Llama-4-Scout (`meta-llama_llama-4-scout`): n=813; pass=267 (32.84%); test failure=435; execution error=110; runner error=1
- All observations (`ALL`): n=8,130; pass=3,453 (42.47%); test failure=3,872; execution error=796; runner error=9

### VA

- GPT-5.4 (`gpt-5.4`): n=813; pass=404 (49.69%); test failure=385; execution error=24; runner error=0
- Claude-Sonnet-4.6 (`claude-sonnet-4-6`): n=813; pass=78 (9.59%); test failure=71; execution error=664; runner error=0
- Gemini-3.1-Pro (`gemini-3.1-pro-preview`): n=813; pass=169 (20.79%); test failure=174; execution error=470; runner error=0
- DeepSeek-V3.2 (`DeepSeek-V3.2`): n=813; pass=134 (16.48%); test failure=143; execution error=536; runner error=0
- Kimi-K2.5 (`kimi-k2.5`): n=813; pass=115 (14.15%); test failure=167; execution error=531; runner error=0
- Qwen3.5-397B (`Qwen3.5-397B-A17B`): n=813; pass=87 (10.70%); test failure=87; execution error=639; runner error=0
- Qwen3-235B (`Qwen3-235B-A22B-Instruct-2507`): n=813; pass=65 (8.00%); test failure=95; execution error=653; runner error=0
- Qwen3-30B (`Qwen3-30B-A3B-Instruct-2507`): n=813; pass=145 (17.84%); test failure=208; execution error=460; runner error=0
- MiniMax-M2.5 (`MiniMax-M2.5`): n=813; pass=65 (8.00%); test failure=93; execution error=655; runner error=0
- Llama-4-Scout (`meta-llama_llama-4-scout`): n=813; pass=73 (8.98%); test failure=144; execution error=595; runner error=1
- All observations (`ALL`): n=8,130; pass=1,335 (16.42%); test failure=1,567; execution error=5,227; runner error=1

## Aggregate dynamic outcomes

### MR

- total: 8,130
- pass: 3,453 (42.47%)
- test failure (`fail`): 3,872
- execution error (`error`): 796 (9.79%)
- runner error (`runner_error`): 9

### VA

- total: 8,130
- pass: 1,335 (16.42%)
- test failure (`fail`): 1,567
- execution error (`error`): 5,227 (64.29%)
- runner error (`runner_error`): 1

- aggregate pass-rate difference: 26.05 percentage points

## Joint static and dynamic outcomes

### MR

- Static pass + dynamic pass: 3,205 (39.42%)
- Static pass + dynamic non-pass: 3,570 (43.91%)
- Static fail + dynamic pass: 248 (3.05%)
- Static fail + dynamic non-pass: 1,107 (13.62%)
- four-category sum: 8,130

### VA

- Static pass + dynamic pass: 1,204 (14.81%)
- Static pass + dynamic non-pass: 1,743 (21.44%)
- Static fail + dynamic pass: 131 (1.61%)
- Static fail + dynamic non-pass: 5,052 (62.14%)
- four-category sum: 8,130

## Conditional dynamic success among statically compatible observations

These rates are reported but are not the denominator in Panel (a).

- MR: statically compatible=6,775; dynamic pass among them=3,205; conditional rate=47.31%
- VA: statically compatible=2,947; dynamic pass among them=1,204; conditional rate=40.86%

## Execution errors and static incompatibility

- MR: execution errors=796; statically incompatible execution errors=737; share=92.59%
- VA: execution errors=5,227; statically incompatible execution errors=4,831; share=92.42%

## Assertions

- 63/63 passed

- input SHA-256 equals 88750aa52441dbf1db1eda0f208a60eb0aa3290fed6871b849c8958d9998bba7: pass (sha256=88750aa52441dbf1db1eda0f208a60eb0aa3290fed6871b849c8958d9998bba7)
- filtered records are Python 3.12: pass (python_version values=['3.12'])
- filtered records have track == "d1": pass (track values=['d1'])
- (track, mode, model, task_id) unique in the filtered set: pass (n=16260 unique=16260)
- bcb_status values are pass, fail, error, or runner_error: pass (bcb_status values=['error', 'fail', 'pass', 'runner_error'])
- ty_compatible values are JSON booleans: pass (ty_compatible values=[False, True])
- MR contains exactly 10 models: pass (n_models=10)
- MR contains exactly 8,130 observations: pass (n=8,130)
- MR GPT-5.4 contains exactly 813 observations: pass (n=813)
- MR Claude-Sonnet-4.6 contains exactly 813 observations: pass (n=813)
- MR Gemini-3.1-Pro contains exactly 813 observations: pass (n=813)
- MR DeepSeek-V3.2 contains exactly 813 observations: pass (n=813)
- MR Kimi-K2.5 contains exactly 813 observations: pass (n=813)
- MR Qwen3.5-397B contains exactly 813 observations: pass (n=813)
- MR Qwen3-235B contains exactly 813 observations: pass (n=813)
- MR Qwen3-30B contains exactly 813 observations: pass (n=813)
- MR MiniMax-M2.5 contains exactly 813 observations: pass (n=813)
- MR Llama-4-Scout contains exactly 813 observations: pass (n=813)
- MR total equals 8,130: pass (total=8,130)
- MR dynamic pass equals 3,453 (42.47%): pass (pass=3,453 (42.47%))
- MR test failure (bcb_status=fail) equals 3,872: pass (fail=3,872)
- MR execution error equals 796 (9.79%): pass (error=796 (9.79%))
- MR runner_error equals 9: pass (runner_error=9)
- MR four joint counts sum to 8,130: pass (sum=8,130)
- MR Static pass + dynamic pass equals 3,205 (39.42%): pass (count=3,205 (39.42%))
- MR Static pass + dynamic non-pass equals 3,570 (43.91%): pass (count=3,570 (43.91%))
- MR Static fail + dynamic pass equals 248 (3.05%): pass (count=248 (3.05%))
- MR Static fail + dynamic non-pass equals 1,107 (13.62%): pass (count=1,107 (13.62%))
- MR statically compatible equals 6,775: pass (static_pass=6,775)
- MR dynamic pass among statically compatible equals 3,205 (47.31%): pass (conditional=3,205/6,775 (47.31%))
- MR execution errors among statically incompatible observations equal 737/796 (92.59%): pass (share=737/796 (92.59%))
- MR model pass-rate range equals 31.61% to 51.91%: pass (range=31.61% to 51.91%)
- VA contains exactly 10 models: pass (n_models=10)
- VA contains exactly 8,130 observations: pass (n=8,130)
- VA GPT-5.4 contains exactly 813 observations: pass (n=813)
- VA Claude-Sonnet-4.6 contains exactly 813 observations: pass (n=813)
- VA Gemini-3.1-Pro contains exactly 813 observations: pass (n=813)
- VA DeepSeek-V3.2 contains exactly 813 observations: pass (n=813)
- VA Kimi-K2.5 contains exactly 813 observations: pass (n=813)
- VA Qwen3.5-397B contains exactly 813 observations: pass (n=813)
- VA Qwen3-235B contains exactly 813 observations: pass (n=813)
- VA Qwen3-30B contains exactly 813 observations: pass (n=813)
- VA MiniMax-M2.5 contains exactly 813 observations: pass (n=813)
- VA Llama-4-Scout contains exactly 813 observations: pass (n=813)
- VA total equals 8,130: pass (total=8,130)
- VA dynamic pass equals 1,335 (16.42%): pass (pass=1,335 (16.42%))
- VA test failure (bcb_status=fail) equals 1,567: pass (fail=1,567)
- VA execution error equals 5,227 (64.29%): pass (error=5,227 (64.29%))
- VA runner_error equals 1: pass (runner_error=1)
- VA four joint counts sum to 8,130: pass (sum=8,130)
- VA Static pass + dynamic pass equals 1,204 (14.81%): pass (count=1,204 (14.81%))
- VA Static pass + dynamic non-pass equals 1,743 (21.44%): pass (count=1,743 (21.44%))
- VA Static fail + dynamic pass equals 131 (1.61%): pass (count=131 (1.61%))
- VA Static fail + dynamic non-pass equals 5,052 (62.14%): pass (count=5,052 (62.14%))
- VA statically compatible equals 2,947: pass (static_pass=2,947)
- VA dynamic pass among statically compatible equals 1,204 (40.86%): pass (conditional=1,204/2,947 (40.86%))
- VA execution errors among statically incompatible observations equal 4,831/5,227 (92.42%): pass (share=4,831/5,227 (92.42%))
- VA model pass-rate range equals 8.00% to 49.69%: pass (range=8.00% to 49.69%)
- VA pass rate is lower for 9 of 10 models: pass (VA lower=Claude-Sonnet-4.6, Gemini-3.1-Pro, DeepSeek-V3.2, Kimi-K2.5, Qwen3.5-397B, Qwen3-235B, Qwen3-30B, MiniMax-M2.5, Llama-4-Scout (9))
- GPT-5.4 is the sole exception: pass (exception=GPT-5.4)
- GPT-5.4 MR pass equals 401/813 (49.32%): pass (pass=401/813 (49.32%))
- GPT-5.4 VA pass equals 404/813 (49.69%): pass (pass=404/813 (49.69%))
- aggregate pass-rate difference equals 26.05 percentage points: pass (difference=26.05 pp)

## Software versions

- Python: `3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]`
- pandas: `3.0.1`
- matplotlib: `3.10.8`

## Reproduction command

From the repository root:

`.venv/bin/python -m plots.rq3_dynamic_outcomes`

## Output geometry and fonts

- PNG: 2102 x 1056 px, 300.0 dpi, 7.01 in x 3.52 in
- PDF MediaBox: 7.01 in x 3.52 in (504.7820275 pt x 253.512975 pt)
- embedded fonts: FXGCJY+LiberationSerif, FXGCJY+LiberationSerif-Bold
- font subtypes: CIDFontType2, Type0
- `pdfinfo` and `pdffonts` were not installed on this host; page size and embedded fonts were read directly from the PDF

## Outputs

- csv: `plots/rq3_dynamic_outcomes.csv`
  - SHA-256: `77b6c49a57664b22ef14e42e9b82576090f9e0d6ea6047871e1b52d064c273c0`
- pdf: `plots/rq3_dynamic_outcomes.pdf`
  - SHA-256: `0b494d137aee3094a62971bf33da863cc95bf69fc570ead72a39fb92e5249376`
- png: `plots/rq3_dynamic_outcomes.png`
  - SHA-256: `90fd0f5a33ad34f31f91c4442676bf9fb1eab16aae3c0c43bc078f70efdf5b65`
- manuscript_pdf: `paper_results/figures/rq3_dynamic_outcomes.pdf`
  - SHA-256: `0b494d137aee3094a62971bf33da863cc95bf69fc570ead72a39fb92e5249376`
- report: `plots/rq3_dynamic_outcomes_report.md`
  - SHA-256 is computed after this file is written and printed to stdout

## Manuscript files

Manuscript source files were not modified.  This script writes the analysis CSV, PDF, PNG, and this report under `plots/`, and copies the PDF to `paper_results/figures/`.
