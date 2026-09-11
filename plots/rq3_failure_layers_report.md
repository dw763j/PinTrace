# RQ3 failure-layer distribution verification report

## Input

- path: `outputs/evaluation/task_outcomes.jsonl`
- SHA-256: `88750aa52441dbf1db1eda0f208a60eb0aa3290fed6871b849c8958d9998bba7`

## Filter

- source: `task_outcomes.jsonl` from the corrected canonical tables
- do not use `compatibility_layers.csv`
- `track == "d2"`
- `resolver_status != "empty"` (drop no-dependency observations)
- `compatibility_layer != "compatible"` (failure-only)
- retain only the four mutually exclusive static failure layers listed below
- stop if `infrastructure`, `other`, a missing layer, or any other unexpected layer is present
- the one MR `api_mismatch_candidate` observation with `resolver_status=empty` is excluded by the empty-resolver filter; the MR API-mismatch candidate count must therefore be 1,558 rather than 1,559

## Mode mapping

- `requirements.txt` -> `MR`
- `inline` -> `VA`

## Model-name mapping

Display names and order come from `plots.model_display.MODEL_LABEL` via `paper_display_label`.
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

- figure order: GPT-5.4, Claude-Sonnet-4.6, Gemini-3.1-Pro, DeepSeek-V3.2, Kimi-K2.5, Qwen3.5-397B, Qwen3-235B, Qwen3-30B, MiniMax-M2.5, Llama-4-Scout, then `All observations`
- `All observations` sums raw counts across models before computing percentages

## Overall counts and shares

### MR

- failure observations: 4,191
- Environment construction: 1,456 (34.74%)
- API-mismatch candidate: 1,558 (37.17%)
- Import/initialization: 645 (15.39%)
- Code/syntax: 532 (12.69%)

### VA

- failure observations: 5,436
- Environment construction: 3,594 (66.11%)
- API-mismatch candidate: 1,068 (19.65%)
- Import/initialization: 385 (7.08%)
- Code/syntax: 389 (7.16%)

## Per-model failure denominators and layer counts

### MR

- GPT-5.4 (`gpt-5.4`): n=318; Environment construction=70 (22.01%); API-mismatch candidate=151 (47.48%); Import/initialization=44 (13.84%); Code/syntax=53 (16.67%)
- Claude-Sonnet-4.6 (`claude-sonnet-4-6`): n=429; Environment construction=151 (35.20%); API-mismatch candidate=170 (39.63%); Import/initialization=72 (16.78%); Code/syntax=36 (8.39%)
- Gemini-3.1-Pro (`gemini-3.1-pro-preview`): n=314; Environment construction=84 (26.75%); API-mismatch candidate=154 (49.04%); Import/initialization=41 (13.06%); Code/syntax=35 (11.15%)
- DeepSeek-V3.2 (`DeepSeek-V3.2`): n=453; Environment construction=196 (43.27%); API-mismatch candidate=164 (36.20%); Import/initialization=49 (10.82%); Code/syntax=44 (9.71%)
- Kimi-K2.5 (`kimi-k2.5`): n=421; Environment construction=123 (29.22%); API-mismatch candidate=168 (39.90%); Import/initialization=54 (12.83%); Code/syntax=76 (18.05%)
- Qwen3.5-397B (`Qwen3.5-397B-A17B`): n=335; Environment construction=108 (32.24%); API-mismatch candidate=134 (40.00%); Import/initialization=58 (17.31%); Code/syntax=35 (10.45%)
- Qwen3-235B (`Qwen3-235B-A22B-Instruct-2507`): n=440; Environment construction=162 (36.82%); API-mismatch candidate=161 (36.59%); Import/initialization=70 (15.91%); Code/syntax=47 (10.68%)
- Qwen3-30B (`Qwen3-30B-A3B-Instruct-2507`): n=519; Environment construction=297 (57.23%); API-mismatch candidate=114 (21.97%); Import/initialization=62 (11.95%); Code/syntax=46 (8.86%)
- MiniMax-M2.5 (`MiniMax-M2.5`): n=420; Environment construction=89 (21.19%); API-mismatch candidate=147 (35.00%); Import/initialization=77 (18.33%); Code/syntax=107 (25.48%)
- Llama-4-Scout (`meta-llama/llama-4-scout`): n=542; Environment construction=176 (32.47%); API-mismatch candidate=195 (35.98%); Import/initialization=118 (21.77%); Code/syntax=53 (9.78%)
- All observations (`ALL`): n=4,191; Environment construction=1,456 (34.74%); API-mismatch candidate=1,558 (37.17%); Import/initialization=645 (15.39%); Code/syntax=532 (12.69%)

### VA

- GPT-5.4 (`gpt-5.4`): n=357; Environment construction=153 (42.86%); API-mismatch candidate=150 (42.02%); Import/initialization=14 (3.92%); Code/syntax=40 (11.20%)
- Claude-Sonnet-4.6 (`claude-sonnet-4-6`): n=418; Environment construction=223 (53.35%); API-mismatch candidate=113 (27.03%); Import/initialization=22 (5.26%); Code/syntax=60 (14.35%)
- Gemini-3.1-Pro (`gemini-3.1-pro-preview`): n=487; Environment construction=359 (73.72%); API-mismatch candidate=94 (19.30%); Import/initialization=11 (2.26%); Code/syntax=23 (4.72%)
- DeepSeek-V3.2 (`DeepSeek-V3.2`): n=583; Environment construction=409 (70.15%); API-mismatch candidate=106 (18.18%); Import/initialization=31 (5.32%); Code/syntax=37 (6.35%)
- Kimi-K2.5 (`kimi-k2.5`): n=527; Environment construction=331 (62.81%); API-mismatch candidate=114 (21.63%); Import/initialization=37 (7.02%); Code/syntax=45 (8.54%)
- Qwen3.5-397B (`Qwen3.5-397B-A17B`): n=539; Environment construction=424 (78.66%); API-mismatch candidate=66 (12.24%); Import/initialization=31 (5.75%); Code/syntax=18 (3.34%)
- Qwen3-235B (`Qwen3-235B-A22B-Instruct-2507`): n=630; Environment construction=460 (73.02%); API-mismatch candidate=92 (14.60%); Import/initialization=45 (7.14%); Code/syntax=33 (5.24%)
- Qwen3-30B (`Qwen3-30B-A3B-Instruct-2507`): n=572; Environment construction=281 (49.13%); API-mismatch candidate=139 (24.30%); Import/initialization=97 (16.96%); Code/syntax=55 (9.62%)
- MiniMax-M2.5 (`MiniMax-M2.5`): n=600; Environment construction=445 (74.17%); API-mismatch candidate=85 (14.17%); Import/initialization=39 (6.50%); Code/syntax=31 (5.17%)
- Llama-4-Scout (`meta-llama/llama-4-scout`): n=723; Environment construction=509 (70.40%); API-mismatch candidate=109 (15.08%); Import/initialization=58 (8.02%); Code/syntax=47 (6.50%)
- All observations (`ALL`): n=5,436; Environment construction=3,594 (66.11%); API-mismatch candidate=1,068 (19.65%); Import/initialization=385 (7.08%); Code/syntax=389 (7.16%)

## Assertions

- 73/73 passed

- filtered records are Python 3.12: pass (python_version values=['3.12'])
- (track, mode, model, task_id) unique in the filtered set: pass (n=9627 unique=9627)
- MR contains exactly 10 models: pass (n_models=10)
- MR failure total equals 4,191: pass (total=4,191)
- MR four-layer sum equals 4,191: pass (sum=4,191)
- MR Environment construction count equals 1,456: pass (count=1,456)
- MR Environment construction share equals 34.74%: pass (share=34.74%)
- MR API-mismatch candidate count equals 1,558: pass (count=1,558)
- MR API-mismatch candidate share equals 37.17%: pass (share=37.17%)
- MR Import/initialization count equals 645: pass (count=645)
- MR Import/initialization share equals 15.39%: pass (share=15.39%)
- MR Code/syntax count equals 532: pass (count=532)
- MR Code/syntax share equals 12.69%: pass (share=12.69%)
- MR GPT-5.4 four-layer sum equals the model failure count: pass (sum=318 n=318)
- MR GPT-5.4 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR Claude-Sonnet-4.6 four-layer sum equals the model failure count: pass (sum=429 n=429)
- MR Claude-Sonnet-4.6 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR Gemini-3.1-Pro four-layer sum equals the model failure count: pass (sum=314 n=314)
- MR Gemini-3.1-Pro unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR DeepSeek-V3.2 four-layer sum equals the model failure count: pass (sum=453 n=453)
- MR DeepSeek-V3.2 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR Kimi-K2.5 four-layer sum equals the model failure count: pass (sum=421 n=421)
- MR Kimi-K2.5 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR Qwen3.5-397B four-layer sum equals the model failure count: pass (sum=335 n=335)
- MR Qwen3.5-397B unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR Qwen3-235B four-layer sum equals the model failure count: pass (sum=440 n=440)
- MR Qwen3-235B unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR Qwen3-30B four-layer sum equals the model failure count: pass (sum=519 n=519)
- MR Qwen3-30B unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR MiniMax-M2.5 four-layer sum equals the model failure count: pass (sum=420 n=420)
- MR MiniMax-M2.5 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR Llama-4-Scout four-layer sum equals the model failure count: pass (sum=542 n=542)
- MR Llama-4-Scout unrounded shares sum to 100%: pass (sum=100.000000000000%)
- MR All observations four-layer sum equals the model failure count: pass (sum=4,191 n=4,191)
- MR All observations unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA contains exactly 10 models: pass (n_models=10)
- VA failure total equals 5,436: pass (total=5,436)
- VA four-layer sum equals 5,436: pass (sum=5,436)
- VA Environment construction count equals 3,594: pass (count=3,594)
- VA Environment construction share equals 66.11%: pass (share=66.11%)
- VA API-mismatch candidate count equals 1,068: pass (count=1,068)
- VA API-mismatch candidate share equals 19.65%: pass (share=19.65%)
- VA Import/initialization count equals 385: pass (count=385)
- VA Import/initialization share equals 7.08%: pass (share=7.08%)
- VA Code/syntax count equals 389: pass (count=389)
- VA Code/syntax share equals 7.16%: pass (share=7.16%)
- VA GPT-5.4 four-layer sum equals the model failure count: pass (sum=357 n=357)
- VA GPT-5.4 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA Claude-Sonnet-4.6 four-layer sum equals the model failure count: pass (sum=418 n=418)
- VA Claude-Sonnet-4.6 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA Gemini-3.1-Pro four-layer sum equals the model failure count: pass (sum=487 n=487)
- VA Gemini-3.1-Pro unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA DeepSeek-V3.2 four-layer sum equals the model failure count: pass (sum=583 n=583)
- VA DeepSeek-V3.2 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA Kimi-K2.5 four-layer sum equals the model failure count: pass (sum=527 n=527)
- VA Kimi-K2.5 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA Qwen3.5-397B four-layer sum equals the model failure count: pass (sum=539 n=539)
- VA Qwen3.5-397B unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA Qwen3-235B four-layer sum equals the model failure count: pass (sum=630 n=630)
- VA Qwen3-235B unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA Qwen3-30B four-layer sum equals the model failure count: pass (sum=572 n=572)
- VA Qwen3-30B unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA MiniMax-M2.5 four-layer sum equals the model failure count: pass (sum=600 n=600)
- VA MiniMax-M2.5 unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA Llama-4-Scout four-layer sum equals the model failure count: pass (sum=723 n=723)
- VA Llama-4-Scout unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA All observations four-layer sum equals the model failure count: pass (sum=5,436 n=5,436)
- VA All observations unrounded shares sum to 100%: pass (sum=100.000000000000%)
- VA: Environment construction is the largest layer in all 10 models: pass (dominant=GPT-5.4=Environment construction 42.86%, Claude-Sonnet-4.6=Environment construction 53.35%, Gemini-3.1-Pro=Environment construction 73.72%, DeepSeek-V3.2=Environment construction 70.15%, Kimi-K2.5=Environment construction 62.81%, Qwen3.5-397B=Environment construction 78.66%, Qwen3-235B=Environment construction 73.02%, Qwen3-30B=Environment construction 49.13%, MiniMax-M2.5=Environment construction 74.17%, Llama-4-Scout=Environment construction 70.40%)
- VA Environment construction minimum is GPT-5.4 at 42.86%: pass (min=GPT-5.4 42.86%)
- VA Environment construction maximum is Qwen3.5-397B at 78.66%: pass (max=Qwen3.5-397B 78.66%)
- MR: API-mismatch candidate is the largest layer in seven models: pass (API-dominant=GPT-5.4, Claude-Sonnet-4.6, Gemini-3.1-Pro, Kimi-K2.5, Qwen3.5-397B, MiniMax-M2.5, Llama-4-Scout; env-dominant=DeepSeek-V3.2, Qwen3-235B, Qwen3-30B; other=)
- MR: Environment construction is the largest layer in the remaining three models: pass (env-dominant=DeepSeek-V3.2, Qwen3-235B, Qwen3-30B)

## Cross-model pattern checks

- VA: Environment construction is the largest failure layer in all 10 models:
  - GPT-5.4: 42.86% (153/357)
  - Claude-Sonnet-4.6: 53.35% (223/418)
  - Gemini-3.1-Pro: 73.72% (359/487)
  - DeepSeek-V3.2: 70.15% (409/583)
  - Kimi-K2.5: 62.81% (331/527)
  - Qwen3.5-397B: 78.66% (424/539)
  - Qwen3-235B: 73.02% (460/630)
  - Qwen3-30B: 49.13% (281/572)
  - MiniMax-M2.5: 74.17% (445/600)
  - Llama-4-Scout: 70.40% (509/723)
- VA Environment construction range: 42.86% (GPT-5.4) to 78.66% (Qwen3.5-397B)
- MR models whose largest failure layer is API-mismatch candidate:
  - GPT-5.4, Claude-Sonnet-4.6, Gemini-3.1-Pro, Kimi-K2.5, Qwen3.5-397B, MiniMax-M2.5, Llama-4-Scout (7)
- MR models whose largest failure layer is Environment construction:
  - DeepSeek-V3.2, Qwen3-235B, Qwen3-30B (3)

## Software versions

- Python: `3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]`
- pandas: `3.0.1`
- matplotlib: `3.10.8`

## Reproduction command

From the repository root:

`.venv/bin/python -m plots.rq3_failure_layers`

## Output geometry and fonts

- PNG: 3218 x 2202 px, 300.0 dpi, 10.73 in x 7.34 in
- PDF MediaBox: 10.73 in x 7.34 in (772.5733 pt x 528.163375 pt)
- embedded fonts: FXGCJY+LiberationSerif, FXGCJY+LiberationSerif-Bold
- font subtypes: CIDFontType2, Type0
- `pdfinfo` and `pdffonts` were not installed on this host; page size and embedded fonts were read directly from the PDF

## Outputs

- csv: `plots/rq3_failure_layers.csv`
  - SHA-256: `c2c577639882a2012fe31994776aa64cfbb137bfc8dbf455f14492a37e3a7faf`
- pdf: `plots/rq3_failure_layers.pdf`
  - SHA-256: `7b09390555d7a1cadd26910aa165b6fd924593858fbceb02a8816272bf4bdc1a`
- png: `plots/rq3_failure_layers.png`
  - SHA-256: `2c39bbb42cd428a8a72a3eae29f16d3f523417956c7a74a16789d54932a88fe5`
- report: `plots/rq3_failure_layers_report.md`
  - SHA-256 is computed after this file is written and printed to stdout

## Manuscript files

Manuscript files were not modified.  This script writes only the analysis CSV, PDF, PNG, and this report under `plots/`.
