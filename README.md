# PinTrace

Companion artifact for the manuscript:

> **Dependency-Version Advice in LLM-Generated Code: Incidence, Quality, and Consequences**

Repository: [https://github.com/dw763j/PinTrace](https://github.com/dw763j/PinTrace)

This artifact ships the PinTrace task set, the completed evaluation records used in the paper, figure generators, and the extraction–resolution–validation pipeline. Regenerating the paper tables and figures from those records does not call model APIs and does not require `uv` or `ty`.

## Study design

Third-party versions affect whether generated code can be built and maintained securely. We study what version advice large language models produce, how current and valid that advice is, and what happens when it is adopted.

Two request conditions on the same tasks:

- **MR** (manifest request): generate code and a `requirements.txt` file.
- **VA** (version annotation): generate code and attach `# VERSION=x.y.z` to each third-party import.

| Population | Size | Used for |
|---|---:|---|
| PinTrace (Stack Overflow tasks) | 1,000 | RQ1 incidence, RQ3 static outcomes |
| PinTrace clean subset (no target-dependency versions in the prompt) | 849 | RQ2 quality and vulnerability association |
| BigCodeBench tasks with third-party imports | 813 | RQ3 execution tests |

Each condition is evaluated on 10 models, giving 10,000 PinTrace observations and 8,130 execution-test observations per condition. Resolution uses a frozen cutoff of 1 February 2026, `uv` 0.10.7, Python 3.12, and `x86_64`. Vulnerability matching uses the OSV PyPI snapshot of 1 March 2026 and excludes withdrawn advisories.

## Experimental line

The numbers in the manuscript come from one completed evaluation, already stored here:

```
PinTrace / BigCodeBench tasks
        ↓  MR or VA, 10 models
evaluation records (M6): outputs/evaluation/task_outcomes.jsonl
        ├── RQ1  incidence of exact / range / bare advice
        ├── RQ2  constraint validity
        ├── RQ2  known-vulnerability association after dropping withdrawn OSV records
        ├── RQ3  resolve → install → ty; BigCodeBench tests on D1
        ├── neighboring-version recovery on VA install failures (1,290 tasks)
        └── Python 3.8 / 3.10 / 3.12 / 3.14 robustness aggregates
        ↓
paper_results/tables  and  paper_results/figures
```

Raw model dumps, per-task lockfiles, and PyPI/OSV caches are not part of this public artifact.

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the paper tables and figures

```bash
python -m scripts.generate_paper_tables
```

Writes LaTeX and CSV to `paper_results/tables/` and checks the manuscript headline counts:

- RQ1: MR exact 17.75%, MR range 26.15%, VA exact 81.19%
- RQ2 validity: 1,207/1,271 (MR exact), 2,187/2,202 (MR range), 6,389/6,997 (VA exact)
- RQ2 known-vulnerability association after excluding withdrawn advisories: 9.46% (MR exact), 7.57% (MR range), 39.49% (VA exact)
- RQ3 end-to-end static pass: 50.44% (MR), 34.32% (VA)
- Neighboring-version recovery: 504/1,290 installation, 429/1,290 static

Submitted PDFs are in `paper_results/figures/`. Regenerate the scripted figures with:

```bash
python -m plots.d2_tpl_version_distribution_revised
python -m plots.rq3_failure_layers
python -m plots.rq3_dynamic_outcomes
python -m plots.rq2_cve_characteristics
```

`pipeline.pdf` is a schematic and has no generator. `model_exact_version_age.pdf` is shipped. The CVE-characteristics figure uses the official withdrawn-filtered totals in `outputs/analysis/rq2/` (5,301 associated occurrences).

```bash
python -m pytest -q \
  tests/test_revision_metrics.py \
  tests/test_revision_failure_classifier.py \
  tests/test_isolated_bcb_eval.py \
  tests/test_alias_mapping_converter.py \
  tests/test_vuln_check.py \
  tests/test_version_resolver_requirements.py \
  tests/test_test_utilized_tpl_detection.py
```

## Pipeline (how the records were produced)

```
Task text
    ├─ MR: code + requirements.txt
    └─ VA: code with # VERSION=x.y.z on third-party imports
M1  Model generation
M2  Extract dependency declarations (exact / range / bare / invalid)
M3  Time-aligned resolution (whole-environment lock + per-constraint satisfiability)
M4  Match resolved releases against the frozen OSV index
M5  Install the hashed lock, run ty, and run BigCodeBench tests where available
M6  Build canonical tables → outputs/evaluation/
    then paper analysis → outputs/analysis/{rq2,funnels,neighbor,robustness}
```

| Stage | Research question | Code / records |
|---|---|---|
| M2 | RQ1: incidence of exact, range, and bare advice | `evaluate/` → `outputs/evaluation/` |
| M3 + release dates + M4 | RQ2: constraint validity, currency, known-vulnerability association | `outputs/analysis/rq2/` (withdrawn advisories excluded) |
| M3 + M5 | RQ3: resolution, installation, static validation, execution tests | `outputs/evaluation/task_outcomes.jsonl` |
| Python 3.8 / 3.10 / 3.12 / 3.14 | Runtime sensitivity | `outputs/analysis/robustness/` |
| Neighboring-version search | Whether changing model-produced exact pins recovers failed environments | `scripts/run_neighbor_version_experiment.py` → `outputs/analysis/neighbor/` |

After the per-run stages, paper analysis is the same pipeline, not a side tree:

```bash
python -m evaluate.build_canonical_tables --outputs-root <m2-m5-dumps> --modes inline requirements.txt
python -m evaluate.generate_analysis_tables
python -m scripts.generate_extended_results          # writes outputs/analysis/rq2/unfiltered/
python -m scripts.audit_osv_withdrawn                # official RQ2 CSVs (needs OSV snapshot)
python -m scripts.generate_robustness_results        # needs the cross-Python dumps
python -m scripts.generate_paper_tables
```

`build_canonical_tables`, `generate_extended_results`, neighbor completion, and robustness aggregation need the private M2–M5 dumps. Table and figure regeneration from the shipped records does not.

## Dataset

PinTrace is in

```
outputs/dataset_builder/so_e2e_balanced/records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl
```

Task splits:

- `resources/revision/pintrace_clean_task_ids.jsonl` (849)
- `resources/revision/pintrace_contaminated_task_ids.jsonl` (151)
- `resources/revision/bcb_tpl_task_ids.jsonl` (813)

The 849/151 split is an LLM-assisted partition of whether the Stack Overflow question already names a target-dependency version, followed by deterministic cleanup. It is not a human gold standard.

Rebuilding PinTrace from the Stack Overflow dump is documented in [`dataset_builder/README.md`](dataset_builder/README.md).

## Repository layout

```
dataset_builder/     PinTrace construction from Stack Overflow
evaluate/            M1–M6 pipeline, including M6 canonical tables
stages/              Extraction, uv resolution, OSV matching, ty / tests
plots/               Figure scripts
scripts/             Table generation, neighbor experiment, RQ2 audits
paper_results/       Manuscript figures and regenerated tables
resources/revision/  Task-id splits and model knowledge-cutoff dates
outputs/dataset_builder/   PinTrace JSONL
outputs/evaluation/        M6 task_outcomes and direct_dependencies
outputs/analysis/rq2/      Official RQ2 CSVs (withdrawn advisories excluded)
outputs/analysis/funnels/  Constraint-validity funnels
outputs/analysis/neighbor/ Neighboring-version recovery
outputs/analysis/robustness/  Cross-Python aggregates
tests/               Unit tests
```

## Paper artifacts

| Item | Generator | Frozen file |
|---|---|---|
| Fig. pipeline | — | `paper_results/figures/pipeline.pdf` |
| Fig. version concentration | `python -m plots.d2_tpl_version_distribution_revised` | `paper_results/figures/d2_tpl_version_distribution_revised.pdf` |
| Fig. version age | `python -m scripts.generate_extended_results` | `paper_results/figures/model_exact_version_age.pdf` |
| Fig. CVE characteristics | `python -m plots.rq2_cve_characteristics` | `paper_results/figures/rq2_cve_characteristics.pdf` |
| Fig. failure layers | `python -m plots.rq3_failure_layers` | `paper_results/figures/rq3_failure_layers.pdf` |
| Fig. dynamic outcomes | `python -m plots.rq3_dynamic_outcomes` | `paper_results/figures/rq3_dynamic_outcomes.pdf` |
| Tab. dataset temporal / content | `python -m scripts.generate_paper_tables` | `paper_results/tables/dataset_*.tex` |
| Tab. RQ1–RQ3, robustness, neighbor | `python -m scripts.generate_paper_tables` | `paper_results/tables/` |

`tab:dataset_temporal` is frozen because the Stack Overflow answer-time index is not shipped. `tab:models` and `tab:related_comparison` are manuscript tables, not generated from code.

## License

Code is released under the [MIT License](LICENSE).

PinTrace questions are derived from Stack Overflow (CC BY-SA). BigCodeBench tasks remain under their upstream license. OSV records remain under the OSV project terms. Please cite those sources if you redistribute the derived task files.

## Citation

If you use this artifact, please cite the manuscript:

```bibtex
@unpublished{pintrace2026,
  title = {Dependency-Version Advice in {LLM}-Generated Code: Incidence, Quality, and Consequences},
  note  = {Manuscript under review. Artifacts: https://github.com/dw763j/PinTrace}
}
```
