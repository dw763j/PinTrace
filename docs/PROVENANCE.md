# Provenance

The repository corresponds to the manuscript fingerprint in `config/manuscript.json`. That file lists the table labels and exact figure assets referenced by the manuscript. The paper itself is maintained separately.

The code builds on [dw763j/PinTrace](https://github.com/dw763j/PinTrace), commit `ce86cc0c5cfb657c19b2af75aad0c83d2bbd54ea`. The MIT license is preserved. Dataset collection and the generation/evaluation engine are included under `engine/`. The paper analysis and source-grounding helpers are supplied alongside them.

`config/data_manifest.json` records packaged-file hashes, expanded-file hashes, original source-file hashes where applicable, and export transformations. Canonical main data are copied byte for byte. Public exports of repeated-generation responses omit account/billing fields. The final repeated evaluations apply the source-grounded code and dependency inputs used for the paper. The analysis consumes those final records directly.

Main response text is preserved exactly. Source-selection decisions are tied to answer hashes. Source verification reconstructs each selected program and checks its code hash against the final evaluation input. The complete environment requirements are exported from the corresponding Python 3.12 observations. Model advice and harness additions keep separate attribution.

BigCodeBench v0.1.4 was obtained from the project's dataset distribution. Its library metadata reproduces the same 813-task subset. Full task records, entry points, and tests are included. The December 2025 PyPI ranking comes from the `2025.12` tag of `hugovk/top-pypi-packages`.

Stack Overflow source identifiers and post URLs accompany the task data. The attribution export records public author information and license labels returned by the Stack Exchange API on 25 November 2026. One accepted-answer ID was not returned by that API. Its original source URL remains in the attribution file. The frozen text is the task-collection source, while current API metadata supplies attribution information.

The large runtime archive is split without changing its bytes. Its parts and combined hash are recorded in the inventory. Original commands and source paths remain as execution provenance inside raw evidence. They are not required filesystem locations for offline analysis.
