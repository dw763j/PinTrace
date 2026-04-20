# Experiment runbook

## D1

### D1 workflow

#### 1. Environment setup

```bash
apt update
apt install python3.12-dev build-essential python3-dev libxml2-dev libxslt1-dev cargo cmake ninja-build -y

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

#### Configuration

- Most filesystem paths are configured in `paths.py` at the repository root.
- Email notification-related environment variables live in `.env` at the repository root.

#### 2. Call the LLM (stage 1) or run the full pipeline (stages 1–6)

```bash
# Default examples use Python 3.12 for LLM output and downstream stages.
python -m evaluate.d1.pipeline_d1 \
    --workers 10 --run-name "xxx_inline" \
    --stages 1 \
    --filter-third-party \
    --api-mode openrouter \
    --api-base-url https://openrouter.ai/api/v1 \
    --api-key sk-or-v1-xxx \
    --prompt-mode "inline" \
    --api-model xxx/xxx \
    --outputs-d1 outputs/d1
```

#### 3. Run pipeline stages 2–6 to produce `metrics_summary.json` (multiple Python versions supported)

```bash
nohup python -m scripts.batch_pipeline_d1 --outputs-d1 outputs/d1 --parallel 8 --python-version 3.8 --stages 2 3 4 5 6 > logs/batch_pipeline_d1_py38_nohup.out 2>&1 &

python -m scripts.batch_pipeline_d1 --outputs-d1 outputs/d1_backup_0315 --parallel 8 --python-version 3.12 --stages 6 --pipeline-run-name stage6_rerun_py312

python -m scripts.batch_pipeline_d1 --outputs-d1 outputs/d1 --parallel 8 --python-version 3.12 --stages 6 --pipeline-run-name stage6_rerun_py312
```

#### 4. Generate LaTeX tables (from stage-6 outputs)

```bash
python -m plots.generate_pipeline_d1_latex_tables --outputs-d1 outputs/d1 --python-version 3.12 --output-tex outputs/d1/pipeline_d1_paper_tables_py312.tex
```

#### 5. Neighbor-version experiment

```bash
# Baselines include: safe_ty_incompat, safe_ty_install_error, safe_bcb_fail
python -m scripts.run_neighbor_version_experiment \
  --dataset d1 \
  --target-mode inline \
  --python-version 3.12 \
  --select-by safe_ty_install_error \
  --baseline safe_ty_install_error \
  --neighbor-n 3 \
  --max-candidates-k 40 \
  --parallel 40 \
  --focus-install-failed-tpl \
  --focused-neighbor-n 10 \
  --specified-only \
  --notify \
  --all-models
```

## D2

### D2 workflow

#### 1. Build the dataset

```bash
python -m dataset_builder.build_stackoverflow_dataset \
  --run-name so_e2e_balanced \
  --posts-xml-path resources/stackoverflow_20251231/Posts.xml \
  --top-limit 15000 \
  --match-mode balanced \
  --cutoff-date 2020-01-01T00:00:00 \
  --date-filter-mode exact \
  --pair-pass-mode any_or_merged \
  --balanced-total 1000 \
  --balanced-seed 42 \
  --pass2-workers 8
```

#### 2. Single entry: run M1–M6 (inference + analysis)

```bash
source .venv/bin/activate

python -m evaluate.d2.pipeline_d2 \
  --stages 1 2 3 4 5 6 \
  --dataset outputs/dataset_builder/so_e2e_balanced/records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl \
  --api-mode openrouter \
  --api-url https://openrouter.ai/api/v1 \
  --api-key sk-or-v1-xxx \
  --api-model deepseek/deepseek-chat \
  --prompt-mode blind \
  --workers 8 \
  --python-version 3.12 \
  --outputs-d2 outputs/d2
```

#### 3. Reuse existing LLM output; run analysis M2–M6 only

```bash
source .venv/bin/activate

python -m evaluate.d2.pipeline_d2 \
  --run-name deepseek-v3.2_blind \
  --stages 2 3 4 5 6 \
  --llm-output-path outputs/d2/deepseek-v3.2_blind.jsonl \
  --workers 8 \
  --python-version 3.12 \
  --outputs-d2 outputs/d2
```

#### 4. Inference M1 only (to produce `llm_output` first)

```bash
source .venv/bin/activate

python -m evaluate.d2.pipeline_d2 \
  --stages 1 \
  --dataset outputs/dataset_builder/so_e2e_balanced/records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl \
  --api-mode openrouter \
  --api-base-url https://openrouter.ai/api/v1 \
  --api-key sk-or-v1-xxx \
  --api-model deepseek/deepseek-chat \
  --prompt-mode blind \
  --workers 8 \
  --outputs-d2 outputs/d2 \
  --python-version 3.12 \
  --timeout 180
```

#### 5. Compatibility stages only (skip vulnerability)

```bash
source .venv/bin/activate

python -m evaluate.d2.pipeline_d2 \
  --run-name deepseek-v3.2_blind_py312 \
  --stages 2 3 5 6 \
  --llm-output-path outputs/d2/deepseek-v3.2_blind.jsonl \
  --skip-vuln \
  --workers 8 \
  --python-version 3.12 \
  --outputs-d2 outputs/d2
```

#### 6. Batch `pipeline_d2` via jobs JSONL

```bash
source .venv/bin/activate

python -m scripts.batch_pipeline_d2 --jobs-jsonl resources/pipeline_d2_inline_stage2_6_py312.jsonl --parallel 8
```

#### 7. Neighbor-version experiment (D2; no BCB path)

```bash
source .venv/bin/activate

# D2 supports: safe_ty_incompat / safe_ty_install_error
python -m scripts.run_neighbor_version_experiment \
  --dataset d2 \
  --modes inline \
  --target-mode inline \
  --python-version 3.12 \
  --select-by safe_ty_install_error \
  --baseline safe_ty_install_error \
  --neighbor-n 3 \
  --max-candidates-k 40 \
  --parallel 20 \
  --parallel-models 4 \
  --focus-install-failed-tpl \
  --focused-neighbor-n 10 \
  --specified-only \
  --all-models \
  --resume \
  --notify
```

```bash
source .venv/bin/activate

# requirements.txt pinning mode
python -m scripts.run_neighbor_version_experiment \
  --dataset d2 \
  --outputs-d2 outputs/d2 \
  --modes inline requirements.txt \
  --target-mode requirements.txt \
  --python-version 3.12 \
  --select-by safe_ty_incompat \
  --baseline safe_ty_incompat \
  --neighbor-n 3 \
  --max-candidates-k 40 \
  --stop-policy first_pass \
  --specified-only \
  --parallel 8 \
  --notify
```

#### 8. Generate LaTeX tables

```bash
python -m plots.generate_pipeline_d2_latex_tables --python-version 3.12
```

## Ablation

#### Build ablation prompt datasets

```bash
python -m stages.ablation_prompts \
  --run-batch \
  --d1-dataset bigcode/bigcodebench \
  --d1-split v0.1.4 \
  --d2-dataset outputs/dataset_builder/so_e2e_balanced/records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl \
  --pinning-mode inline \
  --d2-context-mode blind \
  --max-examples 1000 \
  --top-k-apis 10 \
  --workers 10
```

#### Run ablation (D1 example)

```bash
python -m evaluate.pipeline \
  --track d1 \
  --inference-mode ablation \
  --prompt-mode inline_safe_version \
  --ablation-prompts-jsonl outputs/ablation_prompts/d1/inline_safe_version.jsonl \
  --run-name gpt-5.4_inline_safe_version \
  --api-model gpt-5.4 \
  --api-mode openrouter \
  --api-base-url https://openrouter.ai/api/v1 \
  --api-key sk-xxx \
  --max-examples 2 \
  --stages 1
```
