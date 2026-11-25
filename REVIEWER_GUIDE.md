# Reviewer guide

This artifact supports **Dependency-Version Advice in LLM-Generated Code: Incidence, Quality, and Consequences**. The manuscript SHA-256 fingerprint and display inventory are recorded in `config/manuscript.json`.

## Reproduce the results

Use Python 3.12 and run these commands from the repository root:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements.lock.txt
python reproduce.py
```

The command verifies the packaged input hashes, recomputes the analyses, exports the table data, regenerates all eight figures, and compares the results with the manuscript reference outputs. It uses the archived responses and does not call model APIs or execute generated programs. Outputs and logs are written to `build/`.

| Review question | Supporting material |
| --- | --- |
| Which inputs support each table and figure? | `docs/EVIDENCE_MAP.md` |
| What are the record units, fields, and denominators? | `docs/DATA_DICTIONARY.md` |
| Where are the reference results and exact manuscript figures? | `results/reference/` |
| What verification has been performed? | `results/verification/` |
| Do the recomputed result CSVs agree with the reference outputs? | `build/comparison.json` |
| Do the displayed numerical table values agree with their inputs? | `build/tables/verification.json` |
| Do the response and selected-code hashes agree with the evaluation inputs? | `build/source_verification.json` |
| What supports each headline rate? | `build/claims.json` |
| Where are the selected programs and complete dependency inputs? | `build/prepared/main.jsonl` |

The main study contains 1,000 PinTrace tasks, an 813-task BigCodeBench subset, ten models, and two request conditions. Quality analyses use the 927 PinTrace tasks without target-library version information. Neighboring-version search, fixed-code comparisons, runtime comparisons, and repeated generation have separate input and result records.

Run `python reproduce.py --verify-only` to check packaged data integrity. Add `--evidence` to the full reproduction command to unpack recorded commands, diagnostic output, and available lock artifacts. See `docs/REPRODUCTION.md` for individual analysis stages and reevaluation on a disposable Linux worker.

Code is licensed under MIT. Third-party tasks and reference data retain their source terms, as documented in `licenses/THIRD_PARTY.md`. Original research inputs preserve their source language and exact contents.
