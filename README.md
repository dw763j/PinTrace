# Correct Code, Vulnerable Dependencies 📊

Official code release for the paper **“Correct Code, Vulnerable Dependencies: A Large Scale Measurement Study of LLM-Specified Library Versions.”**  
The pipelines reproduce and extend large-scale measurements on **PinTrace(1,000 curated Stack Overflow tasks)** and **BigCodeBench**.

---

## What the paper studies 🧭

Modern LLMs routinely emit Python with **third-party library (TPL) imports** annotated with **explicit version identifiers**. Those choices directly affect **security and compatibility**, yet version-level risk in LLM-generated code had not been measured at scale.

This work presents the first large-scale measurement of that risk. Highlights:

| Theme | Finding |
|--------|---------|
| **Specification rate** | With **inline-comment** prompting, models specify versions on **26.83%–95.18%** of library references. When asked for a standalone **`requirements.txt`** manifest instead, the same models specify only **6.45%–59.19%**—showing that version discipline is **format-triggered**, not a stable engineering habit. |
| **Vulnerability exposure** | Among tasks where models **do** pin versions, **36.70%–55.70%** contain at least one **known-CVE** version; **62.75%–74.51%** of those vulnerable pins carry **Critical or High** severity. |
| **Disclosure timing** | **72.27%–91.37%** of the associated CVEs were **publicly disclosed before** each model’s **knowledge cutoff**. |
| **Systemic bias** | Closed- and open-weight models **converge on the same small set of risky releases**, pointing to **systemic bias** rather than isolated mistakes. |
| **Static compatibility** | Static compatibility spans **19.70%–63.20%**, with **installation failure** as a dominant driver. |
| **Dynamic verification** | On an execution-based suite, pass rates fall to **6.49%–48.62%** as version incompatibilities block runs **before** tests execute. |
| **Attribution** | A controlled diagnosis experiment shows failures stem from **version selection**, not from the quality of the generated code. |
| **Mitigation** | Natural-language “safety” instructions are weak; **external version anchoring** substantially reduces vulnerable exposure. |

We disclosed these results to the evaluated model teams and coding-assistant vendors; several confirmed that **no built-in CVE check** exists at the **version-selection** stage today. The paper elevates **LLM version selection** as a **first-class, previously overlooked risk surface** in LLM-assisted development.

---

## Repository layout 📁

> D1 -> BigCodeBench dataset. D2 -> PinTrace dataset.

| Path | Role |
|------|------|
| `evaluate/` | D1/D2 inference and end-to-end pipeline entrypoints (`pipeline`, `inference`, …) |
| `stages/` | Version resolution, vulnerability checks, compatibility, clustering, summaries |
| `dataset_builder/` | Build the D2 Stack Exchange dataset from dumps |
| `scripts/` | Batch runners, statistics, neighbor-version experiments, utilities |
| `plots/` | Paper figures and LaTeX table generators |
| `paths.py` | **Single source of truth** for data roots, caches, and outputs |
| `experiments_steps.md` | Command cookbook (D1 / D2 / ablations) |
| `dataset_builder/README.md` | Stage-by-stage dataset construction |

Each Python module documents how to run it from the **repository root** (`python -m …`) or how to import it.

---

## Quick start ⚡

```bash
cd llm_tpl_open_source
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
cp .env.example .env        # optional: email notifications, etc.
```

Configure vendor **API keys** via `.env` or the environment—**never commit real secrets**.

---

## `paths.py` and data you must supply 📥

`paths.py` defines four top-level areas under the repo root:

- **`global_cache/`** — shared caches (PyPI version lists, `mapping.json`, CVE cache, `osv_version_matrix.json`, …)
- **`resources/`** — **raw or pre-built inputs** you provide (see table below)
- **`outputs/`** — run artifacts (JSONL outputs, per-stage JSON, `metrics_summary.json`, …)
- **`logs/`**, **`.cache/`** — logs and app caches (uv, Hugging Face, …)

The following variables point to **conventionally named paths under `resources/`** (dates in folder names reflect the **snapshots** used in our study; you may substitute equivalent dumps and update `paths.py` accordingly):

| `paths.py` variable | Default path | What it is / where to get it |
|---------------------|----------------|------------------------------|
| `PYPI_INFO` | `resources/pypi_info/` | Per-package PyPI JSON mirrors (`pypi#<name>.json`). Fetched on demand by the tooling, or pre-seeded. |
| `OSV_INDEX` | `resources/osv#pypi#vulns#20260301/` | OSV **PyPI ecosystem** JSON advisories. Align with exports from [OSV](https://google.osv.dev). |
| `STACKOVERFLOW` | `resources/stackoverflow_20251231/` | Stack Overflow **data dump** (e.g. `Posts.xml`). Source: [Stack Exchange Data Dump](https://archive.org/details/stackexchange). |
| `TOP_PYPI_PACKAGES` | `resources/top-pypi-packages.min.json` | Top-downloaded PyPI packages JSON (e.g. derived from community projects such as `hugovk/top-pypi-packages`). |
| `CVE_DUMP` | `resources/2026-02-27_all_CVEs_at_midnight/` | Full CVE corpus directory (name matches our snapshot date); commonly sourced from **NVD** releases for CVSS enrichment. |
| `BIGCODEBENCH` | `resources/BigCodeBench-v0.1.4-local.jsonl` | **BigCodeBench** JSONL for D1 (use the **v0.1.4** split to match the paper). |
| `STACKEXCHANGE_INDEX_DB` | `resources/stackexchange_index.sqlite3` | SQLite index produced in dataset builder **Pass1** (see `dataset_builder/README.md`). |
| `ANSWER_TIME_INDEX_DB` | `resources/answer_time_index.sqlite3` | Answer creation-time index for temporal filtering/stats (`dataset_builder`, `stats_so_e2e_dataset_content`). |

`global_cache/mapping.json` is populated automatically by `stages.utils.load_mapping` (pipreqs-style mapping URL)—no manual edit required, but you need **one online fetch** or a vendored mapping file.

Missing any critical path above will cause stages to **fail fast or skip**; read `dataset_builder/README.md` and `experiments_steps.md` in order before large runs.

---

## Typical workflows 🔧

1. **Configure** — Edit snapshot names in `paths.py` if your `resources/` layout differs.  
2. **Build D2 data** — `python -m dataset_builder.build_stackoverflow_dataset --help` (see `dataset_builder/README.md`).  
3. **Run D2 pipeline** — `python -m evaluate.d2.pipeline_d2 --help` (batch examples in `experiments_steps.md`).  
4. **Run D1 pipeline** — `python -m evaluate.d1.pipeline_d1 --help` and `python -m scripts.batch_pipeline_d1 --help`.  
5. **Figures & tables** — `python -m plots.<module> --help` (e.g. `plots.generate_pipeline_d1_latex_tables`, `plots.d2_cve_disclosure`).

---

## Citation 📚

If you use this repository in academic work, please cite the paper above (use the official venue metadata once available).

---

## Security 🔐

- Do **not** commit **API keys**, **SMTP passwords**, or private **endpoints**—use environment variables or a local `.env` that stays untracked.  
- Strings like `sk-xxx` in docs and sample job JSON are **placeholders only**.

For reproducibility questions, cross-check `experiments_steps.md` or open an issue.
