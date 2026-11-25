# Reproduction guide

## Offline analysis

`python reproduce.py` is the complete offline entry point. The default bootstrap count is 10,000. The task is the resampling unit, with both request conditions and the model observations kept together. Main-study analyses use seed 20261109. Repeated-generation analyses use seed 20261110 and keep all three rounds for each selected task. Percentile 2.5% and 97.5% limits reproduce the paper's intervals.

The workflow reads `config/data_manifest.json`. Every compressed input has a SHA-256 checksum. Uncompressed file hashes are also checked during materialization. Analysis files are written to `build/`, so packaged inputs and reference results remain unchanged.

For a different output location, run:

```sh
python reproduce.py --workspace /path/to/pintrace-build
```

The default workflow needs roughly 2 GB of working disk space in addition to the repository. Unpacking all raw execution evidence requires additional space. The full Stack Overflow XML dump and package distribution binaries are external acquisition inputs. The included task sample, collection statistics, and pre-balancing package-incidence counts support the reported sample and weighting calculations.

## Analysis sequence

The entry point runs the following stages in dependency order:

1. `reanalyze.py` assesses the canonical declarations against the frozen PyPI and OSV inputs. It computes response-level quality metrics, release ages, advisory associations, package reweighting, and paired bootstrap comparisons.
2. `extraction_sensitivity.py` computes comparisons after excluding source-selection-sensitive task/model pairs.
3. `analyze_neighbor.py` checks the candidate ordering and the selected 815-response search population against the recorded search results. It produces recovery rates and budget-prefix results.
4. `analyze_fixed_code.py`, `analyze_runtime.py`, and `analyze_repeated.py` aggregate their respective response-level inputs.
5. `prepare_displays.py`, `summarize_version_concentration.py`, and `export_tables.py` construct table/plot data and check the numbers shown in the manuscript.
6. `verify_source_inputs.py` verifies raw-response and selected-code hashes, reconstructs the final source files, exports environment inputs, and verifies the BigCodeBench subset from its library metadata.
7. `verify_claims.py` checks headline rates with explicit numerators and denominators.
8. Plot scripts regenerate all eight paper figures. Figure 1 uses the supplied vector artwork and label-update code. The other figures are computed from result CSVs.

Numerical CSV comparisons allow floating-point roundoff below 1e-9. Table checks compare the rounded values displayed in the manuscript. Figure checks compare all extracted labels and numerical text with the exact manuscript assets. The reference PDFs preserve those assets. PDF font embedding and tight bounding boxes can vary slightly when plots are rendered again.

For individual stages, set `PINTRACE_WORKSPACE` to the materialized build directory. For example:

```sh
export PINTRACE_WORKSPACE="$PWD/build"
python analysis_code/analyze_runtime.py
python analysis_code/export_tables.py
```

In PowerShell, use `$env:PINTRACE_WORKSPACE = (Resolve-Path build).Path`.

## Inspect raw execution evidence

```sh
python reproduce.py --evidence
```

The runtime archive is stored in ordered parts smaller than GitHub's per-file limit. The entry point verifies each part and the combined archive before unpacking it. Archive paths are checked before extraction. Raw files appear under `build/raw_evidence/`. Their command paths describe the recorded Linux executions. Active analysis code uses repository-relative inputs and does not require those original directories.

The runtime evidence includes commands, outputs, protocol settings, source assessments, the final observation records, and reevaluation records. The source-assessment archive supports independent constraint checks and the additional neighboring-version searches. Complete joint-lock pin maps are included for the main study. Textual lock artifacts are available where captured in the execution evidence.

## Reevaluate a program

Reevaluation installs third-party packages and runs model-written programs. Run it on a disposable Linux worker. The offline result-reproduction command does not perform these operations.

The main environment uses Python 3.12, uv 0.10.7, ty 0.0.23, Linux x86_64, and the release cutoff `2026-02-01T00:00:00Z`. Resolution and installation each allow 600 seconds. The project-files evaluator allows 120 seconds for static checks and 200 seconds for BigCodeBench execution. BigCodeBench execution uses bubblewrap with a separate network namespace. Original evaluation branches preserve their actual settings in the code and archived commands.

Install the engine dependencies and uv on the disposable worker. The evaluator expects `uv` on `PATH`, or a path in `PINTRACE_UV`. Install bubblewrap for BigCodeBench tests. From the repository root:

```sh
python -m pip install -r engine/requirements.txt
python -m pip install uv==0.10.7
python engine/tools/evaluate_one.py \
  --prepared build/prepared/main.jsonl \
  --request-id REQUEST_ID \
  --out build/new-evaluations/REQUEST_ID \
  --bigcodebench data/inputs/tasks/bigcodebench-selected.jsonl.gz
```

The helper preserves the final selected source files and dependency input. It writes a new evaluation record without replacing recorded outcomes. `REQUEST_ID` is the identifier in `prepared/main.jsonl`. For PinTrace, omit `--bigcodebench`.

## Collect tasks or generate new answers

`engine/dataset_builder/` contains the Stack Overflow collection and screening code. The December 2025 PyPI download ranking is included in the data inventory. The original Stack Overflow dump is obtained separately. `engine/evaluate/inference.py` implements the two request conditions. The repeated-generation task lists, exact prompts, and model request records are packaged with the experiment inputs.

New model generation needs credentials and incurs provider charges. The archived responses are the inputs used for paper-result reproduction. The engine reads credentials from the environment or an interactive secret prompt; no credentials are supplied in this repository.
