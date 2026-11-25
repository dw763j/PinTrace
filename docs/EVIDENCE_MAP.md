# Paper evidence map

The manuscript fingerprint and display labels are in `config/manuscript.json`. Paths below refer to materialized inputs and generated outputs under the build directory, except the explicitly identified reference and code paths. The table CSVs use one analytical unit per record. Their column layouts can differ from the typeset tables. `tables/verification.json` checks the numbers shown in the manuscript.

## Tables

| Label | Question | Supporting input | Computation | Export |
| --- | --- | --- | --- | --- |
| `tab:pintrace_selection` | Stack Overflow selection | `inputs/collection/*.stats.json` | `export_tables.py` | `tables/tab_pintrace_selection.csv` |
| `tab:task_populations` | Task populations | `inputs/assessments/partition; inputs/tasks; inputs/main/task_outcomes.jsonl` | `export_tables.py` | `tables/tab_task_populations.csv` |
| `tab:models` | Model roster | `results/reference/tables/tab_models.tex; results/reference/*.bib` | `curated model descriptions` | `tables/tab_models.csv` |
| `tab:rq1_constraint_forms` | Declaration forms | `inputs/main/source_carrier_declarations.jsonl` | `prepare_displays.py` | `tables/tab_rq1_constraint_forms.csv` |
| `tab:rq1_model_variation` | Complete manifest profiles | `inputs/main/source_carrier_declarations.jsonl; inputs/main/task_outcomes.jsonl` | `summarize_mr_constraint_profiles.py` | `tables/tab_rq1_model_variation.csv` |
| `tab:rq2_validity` | Individual and complete advice validity | `inputs/main/direct_dependencies.jsonl; inputs/main/task_outcomes.jsonl` | `reanalyze.py; prepare_rq2_displays.py` | `tables/tab_rq2_validity.csv` |
| `tab:release_states` | Release existence | `inputs/main/direct_dependencies.jsonl; inputs/registries/pypi*.json` | `reanalyze.py` | `tables/tab_release_states.csv` |
| `tab:rq2_security` | Per-model exact-advice associations | `inputs/main/direct_dependencies.jsonl; inputs/registries/osv; clean-task partition` | `reanalyze.py; prepare_advisory_model_table.py` | `tables/tab_rq2_security.csv` |
| `tab:rq3_outcomes` | Environment stages | `inputs/main/task_outcomes.jsonl` | `reanalyze.py; analyze_rq3_results.py` | `tables/tab_rq3_outcomes.csv` |
| `tab:rq3_installed_diagnostics` | Installed-environment diagnostics | `inputs/main/task_outcomes.jsonl` | `summarize_rq3_compatibility.py` | `tables/tab_rq3_installed_diagnostics.csv` |
| `tab:neighbor_recovery` | Neighbor recovery | `inputs/assessments/neighbor; inputs/search_archive/outputs/analysis/neighbor/models` | `analyze_neighbor.py` | `tables/tab_neighbor_recovery.csv` |
| `tab:fixed_code_comparison` | Fixed-code policies | `experiments/fixed_code/comparisons.json` | `analyze_fixed_code.py` | `tables/tab_fixed_code_comparison.csv` |
| `tab:runtime_sensitivity` | Four-runtime comparison | `experiments/runtime/observations.jsonl` | `analyze_runtime.py` | `tables/tab_runtime_sensitivity.csv` |
| `tab:related_comparison` | Related-work classification | `results/reference/tables/tab_related_comparison.tex; results/reference/*.bib` | `curated literature classification` | `tables/tab_related_comparison.csv` |
| `tab:assessment_rules` | Assessment rules | `results/reference/tables/tab_assessment_rules.tex; engine/tools/build_canonical_dependencies.py` | `curated protocol rules` | `tables/tab_assessment_rules.csv` |
| `tab:dynamic_by_model` | Per-model task tests | `inputs/main/task_outcomes.jsonl` | `reanalyze.py` | `tables/tab_dynamic_by_model.csv` |
| `tab:five_repeat_static` | Repeated static outcomes | `experiments/repeated/evaluations.jsonl` | `analyze_repeated.py` | `tables/tab_five_repeat_static.csv` |
| `tab:five_repeat_tests` | Repeated task tests | `experiments/repeated/evaluations.jsonl` | `analyze_repeated.py` | `tables/tab_five_repeat_tests.csv` |
| `tab:five_repeat_exact` | Repeated exact advice | `experiments/repeated/evaluations.jsonl` | `analyze_repeated.py` | `tables/tab_five_repeat_exact.csv` |

## Figures

| Figure | Purpose | Plot input | Generator |
| --- | --- | --- | --- |
| 1 | Study framework | `inputs/artwork/pipeline.pdf` | `restore_overview_figure.py` |
| 2 | Version distributions | `analysis/rq1_concentration_checked.csv` | `plot_rq1_version_choices.py` |
| 3 | Release age | `analysis/version_age_summary.csv` | `plot_release_age.py` |
| 4 | Advisory publication scopes | `analysis/security_funnels.csv` | `plot_advisory_scopes.py` |
| 5 | Construction and static stages | `analysis/rq3_stage_outcomes.csv` | `plot_rq3_results.py` |
| 6 | Task execution | `analysis/rq3_execution_outcomes.csv` | `plot_rq3_results.py` |
| 7 | Neighbor search budget | `analysis/neighbor_budget_prefix.csv` | `plot_neighbor_budget.py` |
| 8 | Repeated generation | `experiments/repeated/analysis_summary.json` | `plot_five_model_generation.py` |

## Scoped findings

`claims.json` checks the findings against their actual units and populations. Package-level version shares retain their package denominator. The recurring-version checks distinguish a within-package mode from the most frequent package-version pair across packages.

`experiments/runtime/runtime_failure_composition.csv` reports the share without an installed environment among responses without a task-test pass. Each row identifies the Python runtime and request condition. The 70.33% result belongs to VA under Python 3.12. The corresponding VA shares are 41.45% under Python 3.8, 4.07% under Python 3.10, and 84.50% under Python 3.14.

`analysis/extraction_sensitivity.csv` contains the paired source-exclusion comparisons reported in the measurement-validation section. The checks cover the retained pair counts, difference estimates, and interval directions for static pass, task-test pass, and cutoff-scoped advisory association.

The 52.88% neighboring-version static recovery uses 815 selected VA installation failures under Python 3.12. Its qualification requires a complete baseline lock, a recorded installation failure, a direct model-written exact constraint, and no baseline advisory match.
