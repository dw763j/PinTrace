# PinTrace

This repository contains the data and code for **Dependency-Version Advice in LLM-Generated Code: Incidence, Quality, and Consequences**. It connects the model responses and environment assessments to the numerical findings, tables, and figures in the paper.

The main study contains 1,000 PinTrace tasks and 813 BigCodeBench tasks. Ten models answer every task under two dependency request conditions, producing 36,260 responses. Advice-quality analyses use the 927 PinTrace tasks without target-library version information. The repository also includes the fixed-code experiments, neighboring-version search, four-runtime comparison, human checks, and five-model repeated-generation experiment.

Start with the [reviewer guide](REVIEWER_GUIDE.md) for the reproduction commands and the main verification outputs.

## Reproduce the paper results

Use Python 3.12. From this directory, run:

```sh
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
python reproduce.py
```

The command checks packaged input hashes, recomputes the analyses with 10,000 paired task-cluster bootstrap samples, exports table data, generates all eight paper figures, and compares results with the paper's reference outputs. It also verifies the response-to-code lineage and exports the complete inputs for reevaluating each program. This workflow runs offline after dependency installation. It does not call model APIs or execute generated programs.

Outputs appear in `build/`:

| Output | Contents |
| --- | --- |
| `analysis/` | RQ1–RQ3 metrics, declarations, response flags, and paired comparisons |
| `tables/` | Data for each paper table and checks of the displayed numerical values |
| `figures/` | Eight figures, including vector PDFs and image previews |
| `experiments/` | Fixed-code, neighboring-version, runtime, and repeated-generation results |
| `prepared/main.jsonl` | Selected source files and complete dependency inputs for 36,260 responses |
| `comparison.json` | Comparisons with reference result CSVs |
| `source_verification.json` | Response hashes, selected-code hashes, and BigCodeBench subset verification |
| `claims.json` | Headline findings with explicit populations, denominators, runtimes, counts, and paired differences |
| `logs/` | Output from every analysis step |

`python reproduce.py --verify-only` checks all packaged data without unpacking it. `python reproduce.py --evidence` additionally unpacks the full execution evidence, including commands, diagnostic output, and available lock artifacts. See [the reproduction guide](docs/REPRODUCTION.md) for individual steps and environment reevaluation.

## Find the supporting material

| Directory | Purpose |
| --- | --- |
| `data/` | Compressed, checksummed experimental inputs and execution evidence |
| `analysis_code/` | Offline analysis, table export, figure generation, and verification |
| `engine/` | Dataset collection, generation, extraction, resolution, and execution code |
| `config/` | Study settings, file inventory, manuscript fingerprint, and evidence mapping |
| `results/reference/` | Numerical results, table source blocks, and exact figure assets used by the manuscript |
| `docs/` | Reproduction, data schemas, provenance, and table/figure mapping |
| `licenses/` | Third-party notices and attribution |

[The evidence map](docs/EVIDENCE_MAP.md) links every numbered table and figure to its inputs and computation. [The data dictionary](docs/DATA_DICTIONARY.md) explains record keys, populations, and measurement states. [The provenance guide](docs/PROVENANCE.md) identifies source snapshots and the relationship to the [PinTrace repository](https://github.com/dw763j/PinTrace).

The [maintenance guide](docs/MAINTENANCE.md) explains how to synchronize a paper version, its figure generators, and the verification inventory.

## Licenses and citation

PinTrace code is distributed under the [MIT license](LICENSE). Stack Overflow content, BigCodeBench tasks, advisory records, and other third-party material retain their source terms. [Third-party notices](licenses/THIRD_PARTY.md) describe these sources and the per-post attribution records.

Please use [CITATION.cff](CITATION.cff) to cite this work.
