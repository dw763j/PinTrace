# RQ2 constraint-validity verification report

## Data files

- task outcomes: `outputs/evaluation/task_outcomes.jsonl`
- direct dependencies: `outputs/evaluation/direct_dependencies.jsonl`
- clean task ids: `resources/revision/pintrace_clean_task_ids.jsonl`
- frozen exact funnel: `outputs/analysis/funnels/constraint_funnel_exact.csv`
- frozen range funnel: `outputs/analysis/funnels/constraint_funnel_range.csv`
- population: track=`d2`, clean PinTrace tasks=`849`, models=`10`

## Observation key

- `(task_id, model, request condition)`
- MR = `requirements.txt`, VA = `inline`
- observations loaded from `task_outcomes.jsonl`: 16980
- expected: 8490 per condition, 16980 total

## Exact/range inclusion filter

Reuse the canonical funnel filter from `evaluate.build_canonical_tables`:

- `declaration_source` in `{manifest, inline_annotation}`
- `constraint_kind` in `{exact, range}`
- do **not** filter by `resolution_origin`
- exclude bare/unconstrained declarations, code-import-only records, test-harness dependencies, and transitive dependencies
- type-specific `P/S` includes only `constraint_kind` in `{exact, range}`
- response-level `A/S_any` additionally includes model-origin `constraint_kind=invalid` records as failures

## Constraint validity field

- primary: `constraint_valid`
- fallback if `constraint_valid` is missing: `version_exists`
- this matches `_has_valid_constraint` in the canonical generator
- in-scope exact/range records with missing `constraint_valid`: 0

## Invalid or unparsed records

- `constraint_kind=invalid` records in D2 clean: 3
- observations involved: 1
- MR / Claude-Sonnet-4.6 / `stackexchange-62898915`: 3 records ['org.springframework.boot:spring-boot-starter-web', 'org.springframework.boot:spring-boot-starter-data-jpa', 'com.h2database:h2']
- canonical exact/range funnel inclusion: **excluded**, because `constraint_kind` is not `exact` or `range`
- response-level `A/S_any` inclusion: **included as a failure**, because the model-origin declaration cannot identify a satisfiable release
- the affected observation contains only invalid Maven-style coordinates and therefore increases MR `S_any` by one without increasing `A`

## Reproduction of frozen P/S

- reproduced: yes
- MR exact All observations: 1207/1271 (94.96%)
- MR range All observations: 2187/2202 (99.32%)
- VA exact All observations: 6389/6997 (91.31%)
- VA range S: 0 (expected 0)
- every per-model and aggregate P/S cell matches the frozen funnel tables

## All-constraint validity

- MR `S_any`: 3446 (not `S_exact + S_range` = 3473; overlap both-kinds = 28)
- MR `A/S_any`: 3187/3446 (92.48%)
- MR partition all_valid/mixed/none: 3187/187/72 (sum 3446)
- VA `S_any`: 6997 (equals `S_exact`=6997; VA has no range constraints)
- VA `A/S_any`: 5510/6997 (78.75%)
- VA partition all_valid/mixed/none: 5510/879/608 (sum 6997)
- VA `A` is recomputed from record-level validity and is not assumed equal to `P_exact`

## Sanity checks

- 8,490 task--model observations per request condition: pass
- 849 observations per model per condition: pass
- model set is the ten paper models: pass
- VA range `S_range` = 0: pass
- `A <= S_any`: pass
- `all_valid + mixed_validity + none_valid = S_any`: pass
- aggregate percentages sum counts first: pass
- percentages use two decimal places: pass
- frozen P/S table fully reproduced: pass
