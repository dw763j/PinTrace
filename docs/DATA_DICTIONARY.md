# Data dictionary

Files are stored under `data/` as gzip-compressed JSON, JSONL, or CSV. `config/data_manifest.json` records their materialized paths, hashes, sizes, and roles. Running the reproduction command expands analysis inputs into `build/`.

## Keys and populations

The main response key is `(track, mode, model, task_id)`. `track=d2` means PinTrace and `track=d1` means BigCodeBench. `mode=requirements.txt` means manifest request (MR), and `mode=inline` means version annotation (VA). Request conditions differ in what they ask and where advice appears. `request_id` is a stable response identifier. `response_sha256` identifies the exact answer text.

| Population | Tasks | Responses per condition | Role |
| --- | ---: | ---: | --- |
| Complete PinTrace | 1,000 | 10,000 | Advice incidence and environment compatibility |
| Clean PinTrace | 927 | 9,270 | Advice quality and advisory associations |
| PinTrace with target-library versions | 73 | 730 | Complement of the clean partition |
| BigCodeBench subset | 813 | 8,130 | Environment compatibility and executable task tests |

Counts across the two complete datasets sum to 36,260 main responses. Declaration counts use a different unit and must not be divided by response counts without defining an incidence measure.

## Main inputs

| Materialized path | Unit and contents |
| --- | --- |
| `inputs/generation/main.jsonl` | One main response, including its exact prompt, answer, extraction evidence, and source hash |
| `inputs/main/task_outcomes.jsonl` | One main response with independently recorded resolution, installation, static, and dynamic states |
| `inputs/main/direct_dependencies.jsonl` | One merged direct declaration, 83,241 records |
| `inputs/main/source_carrier_declarations.jsonl` | One declaration in its original output carrier, 80,092 records |
| `inputs/main/harness_dependencies.jsonl` | One test-harness dependency, 2,036 records, separate from model advice |
| `inputs/main/excluded_imports.jsonl` | Imports classified outside third-party advice, with their classification |
| `inputs/main/joint_locks.jsonl` | Response-level joint-resolution status, selected pin map, and lock/source hashes |
| `inputs/main/source_correction_flags.jsonl` | Per-response source sensitivity flags used in the exclusion check |
| `inputs/assessments/constraints/results.jsonl` | Unique individual requirement checks, with commands, outputs, statuses, and resolved pins |
| `inputs/assessments/source_decisions/` | Source-selection decisions and structural classifications tied to response hashes |
| `inputs/assessments/identity/` | Distribution-identity decisions with supporting evidence |
| `inputs/assessments/partition/` | Final task partition, source-grounded labels, exact PinTrace prompts, and prompt hashes |
| `inputs/tasks/bigcodebench-v0.1.4.jsonl` | All 1,140 original BigCodeBench tasks, including tests and library metadata |
| `inputs/tasks/bigcodebench-selected.jsonl` | The 813-task subset used by the study |
| `inputs/manual/` | Human judgments and their analysis |

The full PinTrace task records include Stack Overflow question/answer identifiers, text, code, tags, and target packages under `inputs/search_archive/outputs/dataset_builder/`. Models receive the questions. Accepted answers support collection and package identification.

`prepared/main.jsonl` is reconstructed by source verification. Each record contains selected code, companion files, the primary filename, complete environment requirements, and the same response identifiers. The code hash is checked against the final runtime observation for Python 3.12.

## Declaration fields

`constraint_kind` is `exact`, `range`, `bare`, or `invalid`. Exact advice specifies one release. Other nonempty constraints and direct references are ranged advice. Bare advice leaves the version open. A response can contain multiple forms.

`pypi_name` is the identified distribution. `specified_version` is the model-written exact string. RQ1 preserves those strings. `resolution_origin` separates model-exact versions, resolver choices within a model constraint, and unconstrained resolver choices. `constraint_valid` is the individual operational satisfiability result under the study configuration. `frozen_assessment_eligible` identifies active, parseable, identified constraints that can enter that assessment. Source and test-harness attribution are preserved separately.

`declaration_id`, `merged_declaration_ids`, and matching-manifest fields connect merged advice to its original carriers. These records support both occurrence-level distributions and response-level incidence without treating the two units as interchangeable.

## Outcome fields

`environment_eligible` records whether the response supplies usable input for joint resolution. `complete_lock`, `installation_pass`, `static_pass`, and `dynamic_pass` identify successive outcomes. Task tests run after installation, including when static diagnostics occur. A task-test pass therefore does not require `static_pass=true`.

Execution status fields distinguish pass, failure, error, timeout, and stages that were not run. `static_executed` and `dynamic_executed` distinguish observed nonpasses from stages not executed. Main records separately identify incomplete dependency inputs and structurally unusable responses. Those responses remain in the end-to-end response denominator.

## Reference data

`inputs/registries/` contains PyPI release metadata, additional package metadata, the pre-balancing package-incidence counts, and 18,315 OSV records. Advisory matching excludes 397 withdrawn records. Release admission and age use 1 February 2026. Advisory matching uses the 1 March 2026 snapshot, with separate publication scopes for the cutoff and the complete snapshot. Version-level advisory matching is a software-composition-analysis signal.

`inputs/collection/` contains screening statistics and the December 2025 top-PyPI download ranking. The collection counts are 405,707 package-matched records, 132,048 import-supported records, 55,797 records after date screening, and 55,711 parseable candidates before the 1,000-task sample.

## Follow-up experiments

| Path | Unit and scope |
| --- | --- |
| `inputs/assessments/neighbor/` | Qualification of main PinTrace VA responses, ordered candidate plans, and additional search outcomes |
| `inputs/search_archive/outputs/analysis/neighbor/models/` | Original recorded search outcomes used after matching against the final qualification |
| `experiments/fixed_code/comparisons.json` | 120 paired policy cases, comprising 80 PinTrace and 40 BigCodeBench responses, with original, unpinned, and neighboring environments |
| `experiments/fixed_code/selected.json` | Selected programs, requirements, baseline states, and sampling groups |
| `experiments/runtime/runtime_failure_composition.csv` | Runtime-specific environment-availability shares among BigCodeBench responses without a task-test pass |
| `experiments/runtime/observations.jsonl` | 145,040 response/runtime observations across Python 3.8, 3.10, 3.12, and 3.14 |
| `experiments/repeated/evaluations.jsonl` | 2,400 final observations across five models, two request conditions, three rounds, 60 PinTrace tasks, and 20 BigCodeBench tasks |
| `experiments/repeated/requests.jsonl` | Planned repeated-generation requests with exact prompts and round identifiers |
| `experiments/repeated/responses.jsonl` | Final response per planned repeated request, excluding account/billing fields |
| `experiments/repeated/prepared.jsonl` | Source files and environment inputs used by the repeated-generation evaluation |
| `experiments/repeated/tasks.json` | Fixed task list with package groups and the sampling-time group counts |

The final neighboring-version analysis includes 815 installation-failing responses. Candidate ranks refer to the recorded forty-candidate evaluation order. The original search records are retained because they define that order. The qualification file determines which records contribute to the final rates.

Repeated-generation tasks are a fixed package-group sample. Their sampling-time metadata describes the selection frame used when requests were prepared. The main quality partition remains the final 927/73 split. Analyses use the explicit task lists rather than interpreting every sampling-time count as the main-study denominator.
