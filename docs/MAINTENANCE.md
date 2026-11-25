# Keeping the artifact aligned with the paper

The manuscript fingerprint, table source blocks, and figure PDFs describe one paper version. Refresh them together when the paper changes. The sync command reads the manuscript and its referenced assets without modifying the paper or copying its full text into this repository.

```sh
python tools/sync_paper.py --manuscript /path/to/latex/sn-article.tex
python reproduce.py
```

The source manuscript can keep its figure PDFs beside the TeX file or in a `figures` subdirectory. The artifact stores the exact referenced PDFs under `results/reference/figures/` and the numbered table source blocks under `results/reference/tables/`. Source line numbers and file hashes are recorded in `config/manuscript.json`.

Figure generators must produce the same labels and values as the reference PDFs. Figure 1's editable label mapping is in `analysis_code/restore_overview_figure.py`. Figure 8 uses `analysis_code/plot_five_model_generation.py`, with static pass in panel A, task-test pass in panel B, and exact advice in panel C.

When a finding changes, check its population and denominator in `analysis_code/verify_claims.py`. Update the matching evidence-map entry and regenerate its analysis output. Table-label or figure-count changes require updating the evidence map and verification contracts before syncing a new manuscript snapshot.

After reproduction succeeds, refresh the verification reports and repository inventory:

```sh
python tools/sync_paper.py --verification-workspace build --refresh-inventory
python verify_repository.py
python -m unittest discover -s tests -v
```

The inventory preserves exact file hashes across platforms through `.gitattributes`. Reproduction outputs stay in the ignored build directory. Packaged research inputs remain separately checksummed in `config/data_manifest.json`.
