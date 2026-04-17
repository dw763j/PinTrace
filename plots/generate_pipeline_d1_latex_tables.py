#!/usr/bin/env python3
"""
Generate LaTeX tables for pipeline_d1 experiment results.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m plots.generate_pipeline_d1_latex_tables --help

It scans metrics_summary.json under:
- outputs/d1/requirements.txt
- outputs/d1/inline

Task lists come from ``outputs/d1/bcb_tasks_tpl_in_osv_matrix.jsonl`` (see ``scripts.stats_d1_bcb_osv_tasks``).
Unless ``--osv-bcb-tasks-jsonl`` points elsewhere, every table recomputes paper metrics by applying
``stages.metrics.aggregate_metrics`` to ``m5_compat_records.json`` rows filtered to that task-id set
(so BCB, ty, vulnerability, and error taxonomies stay on one consistent subset).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from plots.model_display import order_model_keys, paper_display_label
from stages.metrics import aggregate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
PLOTS = PROJECT_ROOT / "plots"
DEFAULT_OSV_BCB_TASKS_JSONL = OUTPUTS / "d1" / "bcb_tasks_tpl_in_osv_matrix.jsonl"
PYTHON_VERSIONS: tuple[str, ...] = ("3.8", "3.10", "3.12", "3.14")
ABLATION_PYTHON_VERSION = "3.12"
ABLATION_MODES: tuple[str, ...] = ("inline", "inline_no_vuln", "inline_safe_version", "inline_api_rag")
D1_OSV_SUBSET_CAPTION_TAIL = ""
# (
#     r" Metrics aggregate over the BigCodeBench task subset whose TPLs appear in the OSV version matrix "
#     r"(\texttt{bcb\_tasks\_tpl\_in\_osv\_matrix.jsonl})."
# )
FIXME_CELL = r"\FIXME{}"


@dataclass(frozen=True)
class ModelRow:
    model_key: str
    model: str
    requirements: dict[str, Any] | None
    inline: dict[str, Any] | None

    @property
    def group(self) -> str:
        lower = self.model_key.lower()
        if lower.startswith(("gpt", "gemini", "claude")):
            return "Commercial"
        return "Open Source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from outputs/d1 metrics_summary.json."
    )
    parser.add_argument(
        "--outputs-d1",
        type=Path,
        default=OUTPUTS / "d1",
        help="Root outputs/d1 directory.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=PLOTS / "pipeline_d1_paper_tables.tex",
        help="Path to write generated LaTeX tables.",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default=None,
        help=(
            "Python version whose metrics to read, e.g. 3.12. "
            "When set, metrics are loaded from py<ver>/metrics_summary.json subdirectories. "
            "When unset, legacy (non-versioned) metrics_summary.json files are used."
        ),
    )
    parser.add_argument(
        "--osv-bcb-tasks-jsonl",
        type=Path,
        default=DEFAULT_OSV_BCB_TASKS_JSONL,
        help=(
            "Task id list (default: outputs/d1/bcb_tasks_tpl_in_osv_matrix.jsonl). "
            "All D1 tables re-aggregate m5_compat_records.json to this subset."
        ),
    )
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    return name.strip().lower()


def latex_escape_header(label: str) -> str:
    """Escape LaTeX-special characters for dynamic header labels (currently only _ is expected)."""
    return label.replace("_", r"\_")


def load_mode_metrics(mode_dir: Path, python_version: str | None = None) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    if not mode_dir.is_dir():
        return data

    patterns: list[str]
    if python_version:
        py_tag = f"py{python_version.replace('.', '')}"
        patterns = [f"*/{py_tag}/metrics_summary.json", "*/metrics_summary.json"]
    else:
        # Auto-discover both new (py-tagged) and legacy layouts.
        patterns = ["*/py*/metrics_summary.json", "*/metrics_summary.json"]

    found_paths: list[Path] = []
    for pat in patterns:
        found_paths.extend(sorted(mode_dir.glob(pat)))

    # If no metrics found with the requested python_version, gracefully fall back
    # to legacy layout (useful when only old runs exist).
    if not found_paths and python_version:
        found_paths = sorted(mode_dir.glob("*/metrics_summary.json"))

    for metrics_path in found_paths:
        with metrics_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        model_name = obj.get("model_name")
        if not model_name:
            continue
        key = normalize_model_name(str(model_name))
        data[key] = obj
    return data


def discover_d1_m5_paths(mode_dir: Path, python_version: str | None) -> dict[str, Path]:
    """Map normalized model name -> m5_compat_records.json (sibling of metrics_summary.json)."""
    out: dict[str, Path] = {}
    if not mode_dir.is_dir():
        return out
    if python_version:
        py_tag = f"py{python_version.replace('.', '')}"
        metric_paths = sorted(mode_dir.glob(f"*/{py_tag}/metrics_summary.json"))
        metric_paths.extend(sorted(mode_dir.glob("*/metrics_summary.json")))
    else:
        metric_paths = sorted(mode_dir.glob("*/py*/metrics_summary.json"))
        metric_paths.extend(sorted(mode_dir.glob("*/metrics_summary.json")))
    for ms_path in metric_paths:
        m5_path = ms_path.parent / "m5_compat_records.json"
        if not m5_path.is_file():
            continue
        try:
            with ms_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        model_name = obj.get("model_name")
        if not model_name:
            continue
        out[normalize_model_name(str(model_name))] = m5_path
    return out


def load_osv_bcb_task_ids(jsonl_path: Path) -> set[str]:
    """Task ids produced by stats_d1_bcb_osv_tasks (TPL appears in OSV version matrix)."""
    ids: set[str] = set()
    if not jsonl_path.is_file():
        return ids
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("task_id")
            if tid is not None:
                ids.add(str(tid))
    return ids


def _load_m5_record_list(m5_path: Path | None) -> list[dict]:
    if not m5_path or not m5_path.is_file():
        return []
    try:
        data = json.loads(m5_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def metrics_from_m5_osv_subset(
    m5_path: Path | None,
    osv_task_ids: set[str],
    model_name: str,
) -> dict[str, Any]:
    """Recompute metrics_summary-shaped dict from m5 rows whose task_id lies in osv_task_ids."""
    if not osv_task_ids:
        return aggregate_metrics([], model_name)
    records = [
        r
        for r in _load_m5_record_list(m5_path)
        if isinstance(r, dict) and str(r.get("task_id") or "") in osv_task_ids
    ]
    return aggregate_metrics(records, model_name)


def augment_mode_metrics_with_osv_subset(
    metrics_map: dict[str, dict[str, Any]],
    m5_map: dict[str, Path],
    osv_task_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Replace each model's metrics_summary with aggregate_metrics on the OSV task subset."""
    out: dict[str, dict[str, Any]] = {}
    for key, summary in metrics_map.items():
        mn = str(summary.get("model_name", key))
        out[key] = metrics_from_m5_osv_subset(m5_map.get(key), osv_task_ids, mn)
    return out


def pick_display_name(
    req_obj: dict[str, Any] | None, inline_obj: dict[str, Any] | None, normalized_name: str
) -> str:
    raw_name = normalized_name
    for obj in (req_obj, inline_obj):
        if obj and obj.get("model_name"):
            raw_name = str(obj["model_name"])
            break
    return paper_display_label(normalized_name, raw_name)


def build_rows(req_metrics: dict[str, dict[str, Any]], inline_metrics: dict[str, dict[str, Any]]) -> list[ModelRow]:
    model_keys = order_model_keys(set(req_metrics) | set(inline_metrics))
    rows: list[ModelRow] = []
    for key in model_keys:
        req_obj = req_metrics.get(key)
        inline_obj = inline_metrics.get(key)
        rows.append(
            ModelRow(
                model_key=key,
                model=pick_display_name(req_obj, inline_obj, key),
                requirements=req_obj,
                inline=inline_obj,
            )
        )
    return rows


def build_rows_from_versioned_metrics(
    req_by_version: dict[str, dict[str, dict[str, Any]]],
    inline_by_version: dict[str, dict[str, dict[str, Any]]],
) -> list[ModelRow]:
    model_keys: set[str] = set()
    for vmap in req_by_version.values():
        model_keys.update(vmap.keys())
    for vmap in inline_by_version.values():
        model_keys.update(vmap.keys())

    rows: list[ModelRow] = []
    for key in order_model_keys(model_keys):
        req_obj: dict[str, Any] | None = None
        inline_obj: dict[str, Any] | None = None
        for ver in PYTHON_VERSIONS:
            req_obj = req_obj or req_by_version.get(ver, {}).get(key)
            inline_obj = inline_obj or inline_by_version.get(ver, {}).get(key)
        rows.append(
            ModelRow(
                model_key=key,
                model=pick_display_name(req_obj, inline_obj, key),
                requirements=req_obj,
                inline=inline_obj,
            )
        )
    return rows


def load_ablation_mode_metrics(outputs_d1: Path, mode: str, python_version: str) -> dict[str, dict[str, Any]]:
    mode_dir = outputs_d1 / mode
    metrics = load_mode_metrics(mode_dir, python_version)
    if metrics:
        return metrics

    # Fallback for runs that only have paper bundle artifacts.
    out: dict[str, dict[str, Any]] = {}
    for bundle_path in sorted(mode_dir.glob("*/py*/paper/analysis_ready_bundle.json")):
        try:
            obj = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        msum = obj.get("metrics_summary")
        if not isinstance(msum, dict) or not msum:
            continue
        model_name = msum.get("model_name")
        if not model_name:
            continue
        out[normalize_model_name(str(model_name))] = msum
    return out


def fmt_count(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "--"


def fmt_rate_pct(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value) * 100.0:.2f}"
    except (TypeError, ValueError):
        return "--"


def fmt_rate_pct_or_fixme(value: Any) -> str:
    s = fmt_rate_pct(value)
    return FIXME_CELL if s == "--" else s


def fmt_delta_pp_or_fixme(curr_rate: Any, base_rate: Any) -> str:
    try:
        c = float(curr_rate)
        b = float(base_rate)
    except (TypeError, ValueError):
        return FIXME_CELL
    return f"{(c - b) * 100.0:+.2f}"


def fmt_rate_pct_with_delta_pp_inline(curr_rate: Any, base_rate: Any) -> str:
    """
    Ablation cell: ``$rate(\\Delta pp)$`` (percentage points vs baseline).
    If the rate is missing, returns \\FIXME; if only Δ cannot be computed, returns the rate alone.
    """
    rate_s = fmt_rate_pct_or_fixme(curr_rate)
    if rate_s == FIXME_CELL:
        return FIXME_CELL
    try:
        c = float(curr_rate)
        b = float(base_rate)
    except (TypeError, ValueError):
        return rate_s
    dpp = (c - b) * 100.0
    delta_str = f"{dpp:+.2f}"
    return rf"${rate_s}({delta_str})$"


def fmt_high_risk_ratio(metrics: dict[str, Any] | None) -> str:
    """
    Format high-risk vulnerability ratio as 'x/y(zz.zz%)',
    where x = CRITICAL+HIGH counts, y = total vuln findings.
    """
    if not metrics:
        return "--"
    vuln = metrics.get("vulnerability") if isinstance(metrics, dict) else None
    sev = vuln.get("severity_dist") if isinstance(vuln, dict) else None
    if not isinstance(sev, dict):
        return "--"
    try:
        # Keys in severity_dist are uppercase strings like "CRITICAL"/"HIGH"/"MEDIUM"/"LOW".
        critical = int(sev.get("CRITICAL", 0) or 0)
        high = int(sev.get("HIGH", 0) or 0)
        total = sum(int(v or 0) for v in sev.values())
    except (TypeError, ValueError):
        return "--"
    if total <= 0:
        return "--"
    high_risk = critical + high
    pct = high_risk * 100.0 / total
    return f"{high_risk}/{total}({pct:.2f}\\%)"


def get_field(metrics: dict[str, Any] | None, field: str) -> Any:
    if not metrics:
        return None
    return metrics.get(field)


def get_nested_field(metrics: dict[str, Any] | None, parent: str, child: str) -> Any:
    if not metrics:
        return None
    parent_obj = metrics.get(parent)
    if not isinstance(parent_obj, dict):
        return None
    return parent_obj.get(child)


def get_version_field(metrics: dict[str, Any] | None, field: str) -> Any:
    return get_nested_field(metrics, "version_results", field)


def get_vulnerability_field(metrics: dict[str, Any] | None, field: str) -> Any:
    return get_nested_field(metrics, "vulnerability", field)


def get_compat_security_rate(metrics: dict[str, Any] | None, key: str) -> Any:
    cs = metrics.get("compat_security") if isinstance(metrics, dict) else None
    if not isinstance(cs, dict):
        return None
    total = cs.get("task_total")
    value = cs.get(key)
    try:
        total_i = int(total)
        value_i = int(value)
    except (TypeError, ValueError):
        return None
    if total_i <= 0:
        return None
    return value_i / total_i


def get_ty_error_block(metrics: dict[str, Any] | None) -> dict[str, Any]:
    nested = get_nested_field(metrics, "compat_ty", "ty_error")
    if isinstance(nested, dict):
        return nested
    return {}


def get_bcb_error_block(metrics: dict[str, Any] | None) -> dict[str, Any]:
    nested = get_nested_field(metrics, "compat_bcb", "bcb_error")
    if isinstance(nested, dict):
        return nested
    return {}


def latex_escape_model(name: str) -> str:
    return name.replace("_", r"\_")


def ordered_rows_commercial_first(rows: list[ModelRow]) -> list[ModelRow]:
    return [r for r in rows if r.group == "Commercial"] + [r for r in rows if r.group == "Open Source"]


def last_commercial_index(ordered: list[ModelRow]) -> int:
    last = -1
    for idx, r in enumerate(ordered):
        if r.group == "Commercial":
            last = idx
    return last


def render_dual_mode_table_body(
    rows: list[ModelRow],
    value_fn: Callable[[ModelRow, str], list[str]],
    cmidrule_from_col: int,
    cmidrule_to_col: int,
) -> str:
    ordered = ordered_rows_commercial_first(rows)
    if not ordered:
        return ""

    last_ci = last_commercial_index(ordered)
    lines: list[str] = []

    for mode_label, mode_key in (("Explicit", "requirements"), ("Inline", "inline")):
        first_row_in_group = True
        for idx, row in enumerate(ordered):
            left_col = ""
            if first_row_in_group:
                left_col = rf"\multirow{{{len(ordered)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{mode_label}}}}}}}"
                first_row_in_group = False

            vals = value_fn(row, mode_key)
            model_col = latex_escape_model(row.model)
            lines.append(f"{left_col} & {model_col} & " + " & ".join(vals) + r" \\")

            if last_ci >= 0 and idx == last_ci:
                lines.append(rf"\cmidrule[0.5pt](lr){{{cmidrule_from_col}-{cmidrule_to_col}}}")

        if mode_label == "Explicit":
            lines.append(r"\midrule")

    while lines and (
        lines[-1] == r"\midrule" or lines[-1].startswith(r"\cmidrule[0.5pt](lr)")
    ):
        lines.pop()
    return "\n".join(lines)


def render_rows_with_group(
    rows: list[ModelRow],
    value_renderer,
) -> str:
    lines: list[str] = []
    commercial_rows = [r for r in rows if r.group == "Commercial"]
    open_rows = [r for r in rows if r.group == "Open Source"]

    grouped = [("Commercial", commercial_rows), ("Open Source", open_rows)]
    first_group_printed = False
    for group_name, group_rows in grouped:
        if not group_rows:
            continue
        for idx, row in enumerate(group_rows):
            group_col = ""
            if idx == 0:
                group_col = rf"\multirow{{{len(group_rows)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{group_name}}}}}}}"
            model_col = latex_escape_model(row.model)
            values = value_renderer(row)
            lines.append(f"{group_col} & {model_col} & " + " & ".join(values) + r" \\")
        if first_group_printed:
            lines.append(r"\midrule")
        first_group_printed = True

    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return "\n".join(lines)


def table1(rows: list[ModelRow]) -> str:
    # We render two row-groups on the left: "Explicit" and "Inline".
    # Inside each group, we keep the model order, and insert a midrule between
    # commercial and open-source models (without extra group labels).
    def render_body() -> str:
        # Preserve current ordering: Commercial rows first, then Open Source.
        ordered_rows = [r for r in rows if r.group == "Commercial"] + [
            r for r in rows if r.group == "Open Source"
        ]
        if not ordered_rows:
            return ""

        # Find split index between Commercial and Open Source for midrule.
        last_commercial_idx = -1
        for idx, r in enumerate(ordered_rows):
            if r.group == "Commercial":
                last_commercial_idx = idx

        all_lines: list[str] = []

        for mode_label, mode_key in (("Explicit", "requirements"), ("Inline", "inline")):
            group_rows = ordered_rows
            if not group_rows:
                continue

            first_row_in_group = True
            for idx, row in enumerate(group_rows):
                left_col = ""
                if first_row_in_group:
                    left_col = rf"\multirow{{{len(group_rows)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{mode_label}}}}}}}"
                    first_row_in_group = False

                metrics = row.inline if mode_key == "inline" else row.requirements
                values = [
                    fmt_count(get_field(metrics, "total_tasks")),
                    fmt_count(get_version_field(metrics, "total_lib")),
                    fmt_rate_pct(get_version_field(metrics, "lib_version_spec_rate")),
                    fmt_rate_pct(get_version_field(metrics, "lib_version_validity_rate")),
                    fmt_rate_pct(get_vulnerability_field(metrics, "lib_vuln_rate")),
                    fmt_high_risk_ratio(metrics),
                    fmt_rate_pct(get_vulnerability_field(metrics, "task_vuln_exposure")),
                ]

                model_col = latex_escape_model(row.model)
                all_lines.append(f"{left_col} & {model_col} & " + " & ".join(values) + r" \\")

                if last_commercial_idx >= 0 and idx == last_commercial_idx:
                    # Separate commercial and open-source with a cmidrule over
                    # the Model + metric columns (columns 2-9).
                    all_lines.append(r"\cmidrule[0.5pt](lr){2-9}")

            # Separate the two mode groups with a full midrule.
            if mode_label == "Explicit":
                all_lines.append(r"\midrule")

        # Remove trailing rules if they ended the last group.
        while all_lines and all_lines[-1] in {r"\midrule", r"\cmidrule[0.5pt](lr){2-9}"}:
            all_lines.pop()
        return "\n".join(all_lines)

    body = render_body()
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Descriptive statistics of Pipeline D1 by model.
\#Tasks counts tasks in the OSV-aligned subset; \#LibUses denotes third-party library mentions summed over those tasks.
Specified(\%) reports the proportion of mentions with an explicit version declaration,
Valid(\%) reports the proportion of declared versions that can be resolved to valid releases,
LibVuln(\%) reports the share of resolved library versions with known vulnerabilities,
and TaskVuln(\%) reports the proportion of tasks exposed to at least one vulnerable dependency."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_basic_and_validity}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{llrrrrrrr}}
\toprule
& \textbf{{Model}} & \textbf{{\#Tasks}} & \textbf{{\#LibUses}} & \textbf{{Specified(\%)}} & \textbf{{Valid(\%)}} & \textbf{{LibVuln(\%)}} & \textbf{{HighRisk(\%)}} & \textbf{{TaskVuln(\%)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table2(rows: list[ModelRow]) -> str:
    # Table2 has been merged into table1 (see table1) to present
    # Specified(\%) and Valid(\%) / vulnerability metrics together.
    # Keep this function as a no-op to avoid import/usage errors.
    return ""


def table3(rows: list[ModelRow]) -> str:
    def values(row: ModelRow, mode_key: str) -> list[str]:
        metrics = row.inline if mode_key == "inline" else row.requirements
        return [
            fmt_rate_pct(get_nested_field(metrics, "compat_ty", "task_compat_rate")),
            fmt_rate_pct(get_compat_security_rate(metrics, "task_safe_ty_compat")),
            fmt_rate_pct(get_compat_security_rate(metrics, "task_unsafe_ty_compat")),
            fmt_rate_pct(get_compat_security_rate(metrics, "task_safe_ty_incompat")),
            fmt_rate_pct(get_compat_security_rate(metrics, "task_unsafe_ty_incompat")),
        ]

    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=7)
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Measurement results of compatibility between LLM code generation outputs and its specified third-party library versions. 
Compat(\%) represents the percentage of tasks that pass the static type-checking. 
Quadrant entries categorize tasks based on the intersection of type-checking compatibility (Compat vs.\ Incompat) and vulnerability status (Safe vs.\ Unsafe)."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_compat_quadrants}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{llrrrrr}}
\toprule
& \textbf{{Model}} & \textbf{{Compat(\%)}} & \textbf{{Safe$\cap$Compat}} & \textbf{{Unsafe$\cap$Compat}} & \textbf{{Safe$\cap$Incompat}} & \textbf{{Unsafe$\cap$Incompat}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table_bcb_osv_matrix_tasks(rows: list[ModelRow]) -> str:
    """BigCodeBench outcomes (rows must already carry OSV-subset aggregates)."""

    def values(row: ModelRow, mode_key: str) -> list[str]:
        metrics = row.inline if mode_key == "inline" else row.requirements
        return [
            fmt_count(get_nested_field(metrics, "compat_bcb", "task_pass")),
            fmt_count(get_nested_field(metrics, "compat_bcb", "task_fail")),
            fmt_count(get_nested_field(metrics, "compat_bcb", "task_error_like")),
            fmt_count(get_nested_field(metrics, "compat_bcb", "task_total")),
            fmt_rate_pct(get_nested_field(metrics, "compat_bcb", "pass_rate")),
        ]

    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=7)
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Measurement results of BigCodeBench test case execution.
\#Pass, \#Fail, and \#Error report task-level test outcomes, 
\#Total denotes the number of measured tasks with a non-empty BCB status, 
and PassRate(\%) is the pass rate among those tasks."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_bcb_results_osv_subset}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{llrrrrr}}
\toprule
& \textbf{{Model}} & \textbf{{\#Pass}} & \textbf{{\#Fail}} & \textbf{{\#Error}} & \textbf{{\#Total}} & \textbf{{PassRate(\%)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table_ty_errors(rows: list[ModelRow]) -> str:
    """Table: ty error taxonomy split into env/install vs other."""

    # install_error second-level clusters (keys are already descriptive strings).
    def collect_install_keys() -> list[str]:
        keys: set[str] = set()
        for row in rows:
            for metrics in (row.inline, row.requirements):
                if not metrics:
                    continue
                te = get_ty_error_block(metrics)
                clusters = te.get("install_error_clusters") or {}
                if isinstance(clusters, dict):
                    keys.update(str(k) for k in clusters.keys())
        return sorted(keys)

    install_keys = collect_install_keys()

    def render_body() -> str:
        ordered_rows = [r for r in rows if r.group == "Commercial"] + [
            r for r in rows if r.group == "Open Source"
        ]
        if not ordered_rows:
            return ""

        last_commercial_idx = -1
        for idx, r in enumerate(ordered_rows):
            if r.group == "Commercial":
                last_commercial_idx = idx

        all_lines: list[str] = []
        for mode_label, mode_key in (("Explicit", "requirements"), ("Inline", "inline")):
            group_rows = ordered_rows
            if not group_rows:
                continue
            first_row_in_group = True
            for idx, row in enumerate(group_rows):
                left_col = ""
                if first_row_in_group:
                    left_col = rf"\multirow{{{len(group_rows)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{mode_label}}}}}}}"
                    first_row_in_group = False

                metrics = row.inline if mode_key == "inline" else row.requirements
                ty_error = get_ty_error_block(metrics)
                task_dist = ty_error.get("task_dist") or {}
                install_clusters = ty_error.get("install_error_clusters") or {}

                # Group 1: env_error + install_error (env_error counts toward total only, not its own column)
                group1_env = int(task_dist.get("env_error", 0))
                group1_install = int(task_dist.get("install_error", 0))
                group1_total = group1_env + group1_install
                install_vals = [fmt_count(install_clusters.get(k)) for k in install_keys]

                # Group 2: other ty errors
                group2_typecheck = int(task_dist.get("typecheck_error", 0))
                group2_infra = int(task_dist.get("infra_error", 0))
                group2_other = int(task_dist.get("other_error", 0))
                group2_total = group2_typecheck + group2_infra + group2_other

                model_col = latex_escape_model(row.model)
                values = [
                    # Env+Install group: install + clusters + total
                    fmt_count(group1_install),
                    *install_vals,
                    fmt_count(group1_total),
                    # Other-ty group: typecheck + infra + other + total
                    fmt_count(group2_typecheck),
                    fmt_count(group2_infra),
                    fmt_count(group2_other),
                    fmt_count(group2_total),
                ]
                all_lines.append(f"{left_col} & {model_col} & " + " & ".join(values) + r" \\")

                if last_commercial_idx >= 0 and idx == last_commercial_idx:
                    right_col = 2 + len(values)
                    all_lines.append(rf"\cmidrule[0.5pt](lr){{2-{right_col}}}")

            if mode_label == "Explicit":
                all_lines.append(r"\midrule")

        while all_lines and (
            all_lines[-1].startswith(r"\midrule")
            or all_lines[-1].startswith(r"\cmidrule[0.5pt](lr)")
        ):
            all_lines.pop()
        return "\n".join(all_lines)

    body = render_body()
    # Columns: mode, model, [Env+Install group] [Other-ty group]
    # Env+Install group: install_error, each install cluster, total (env_error folded into total only)
    env_install_cols = 2 + len(install_keys)  # install + clusters + total
    # Other-ty group: typecheck, infra, other, total
    other_ty_cols = 4
    col_spec = "ll" + "r" * (env_install_cols + other_ty_cols)

    # First header row: grouped heads
    top_header = (
        r"& \textbf{Model}"
        + rf" & \multicolumn{{{env_install_cols}}}{{c}}{{Env+Install errors}}"
        + rf" & \multicolumn{{{other_ty_cols}}}{{c}}{{Other ty errors}} \\"
    )
    # Horizontal rules under the two groups (columns are: 1=mode, 2=model, 3..)
    group_rules = (
        rf"\cmidrule[0.5pt](lr){{3-{2 + env_install_cols}}}"
        + " "
        + rf"\cmidrule[0.5pt](lr){{{3 + env_install_cols}-{2 + env_install_cols + other_ty_cols}}}"
    )

    # Second header row: concrete columns under each group
    header_parts = [
        "",  # mode column
        "",  # model column (already labeled above)
        r"\textbf{install\_error}",
        *[rf"\textbf{{{latex_escape_header(k)}}}" for k in install_keys],
        r"\textbf{total}",
        r"\textbf{typecheck\_error}",
        r"\textbf{infra\_error}",
        r"\textbf{other\_error}",
        r"\textbf{total}",
    ]
    second_header = " & ".join(header_parts) + r" \\"
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Overview of the static compatibility check results. 
The table separates environment/dependency-installation failures from other static-checking failures, 
and further decomposes installation failures into fine-grained clusters to characterize dominant failure patterns."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_ty_errors}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{{col_spec}}}
\toprule
\multicolumn{{2}}{{c}}{{}} & \multicolumn{{{env_install_cols}}}{{c}}{{}} & \multicolumn{{{other_ty_cols}}}{{c}}{{}} \\\\
{top_header}
{group_rules} \\
{second_header}
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table_bcb_errors(rows: list[ModelRow]) -> str:
    """Table: BCB error taxonomy, including ty-install/ty-env counts."""

    bcb_error_keys = ["semantic_fail", "import_error", "infra_error", "other_error"]

    def render_body() -> str:
        ordered_rows = [r for r in rows if r.group == "Commercial"] + [
            r for r in rows if r.group == "Open Source"
        ]
        if not ordered_rows:
            return ""

        last_commercial_idx = -1
        for idx, r in enumerate(ordered_rows):
            if r.group == "Commercial":
                last_commercial_idx = idx

        all_lines: list[str] = []
        for mode_label, mode_key in (("Explicit", "requirements"), ("Inline", "inline")):
            group_rows = ordered_rows
            if not group_rows:
                continue
            first_row_in_group = True
            for idx, row in enumerate(group_rows):
                left_col = ""
                if first_row_in_group:
                    left_col = rf"\multirow{{{len(group_rows)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{mode_label}}}}}}}"
                    first_row_in_group = False

                metrics = row.inline if mode_key == "inline" else row.requirements
                bcb_error = get_bcb_error_block(metrics)
                from_ty_install = bcb_error.get("from_ty_install_error", 0)
                from_ty_env = bcb_error.get("from_ty_env_error", 0)
                fail_dist = bcb_error.get("fail_error_type_dist") or {}

                vals = [
                    fmt_count(from_ty_install),
                    fmt_count(from_ty_env),
                    *[fmt_count(fail_dist.get(k)) for k in bcb_error_keys],
                ]

                model_col = latex_escape_model(row.model)
                all_lines.append(f"{left_col} & {model_col} & " + " & ".join(vals) + r" \\")

                if last_commercial_idx >= 0 and idx == last_commercial_idx:
                    right_col = 2 + len(vals)
                    all_lines.append(rf"\cmidrule[0.5pt](lr){{2-{right_col}}}")

            if mode_label == "Explicit":
                all_lines.append(r"\midrule")

        while all_lines and (
            all_lines[-1].startswith(r"\midrule")
            or all_lines[-1].startswith(r"\cmidrule[0.5pt](lr)")
        ):
            all_lines.pop()
        return "\n".join(all_lines)

    body = render_body()
    col_spec = "ll" + "r" * (2 + len(bcb_error_keys))
    header_parts = [
        r"\textbf{\#ty\_install\_error}",
        r"\textbf{\#ty\_env\_error}",
        r"\textbf{semantic\_fail}",
        r"\textbf{import\_error}",
        r"\textbf{infra\_error}",
        r"\textbf{other\_error}",
    ]
    header_cols = " & ".join(header_parts)
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Taxonomy of BigCodeBench error outcomes. 
The table reports the number of tasks affected by ty-derived installation/environment failures and the distribution of BCB failure categories (semantic, import, infrastructure, and other)."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_bcb_error_taxonomy}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{{col_spec}}}
\toprule
& \textbf{{Model}} & {header_cols} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def _metrics_for_version(
    row: ModelRow,
    mode_key: str,
    version: str,
    req_by_version: dict[str, dict[str, dict[str, Any]]],
    inline_by_version: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    by_ver = inline_by_version if mode_key == "inline" else req_by_version
    return by_ver.get(version, {}).get(row.model_key)


def table_ty_compat_by_python_versions(
    rows: list[ModelRow],
    versions: Sequence[str],
    req_by_version: dict[str, dict[str, dict[str, Any]]],
    inline_by_version: dict[str, dict[str, dict[str, Any]]],
) -> str:
    def values(row: ModelRow, mode_key: str) -> list[str]:
        return [
            fmt_rate_pct(get_nested_field(_metrics_for_version(row, mode_key, ver, req_by_version, inline_by_version), "compat_ty", "task_compat_rate"))
            for ver in versions
        ]

    right_col = 2 + len(versions)
    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=right_col)
    header_cols = " & ".join(rf"\textbf{{{ver}}}" for ver in versions)
    ver_join = ", ".join(versions)
    return (
        rf"""
\begin{{table*}}[t]
\centering
\caption{{Ty compatibility rate under different Python versions. Columns report task-level ty compatibility rate (\%) for Python {ver_join}."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_ty_compat_by_python}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{ll{"r" * len(versions)}}}
\toprule
& \textbf{{Model}} & {header_cols} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table_bcb_pass_by_python_versions(
    rows: list[ModelRow],
    versions: Sequence[str],
    req_by_version: dict[str, dict[str, dict[str, Any]]],
    inline_by_version: dict[str, dict[str, dict[str, Any]]],
) -> str:
    def values(row: ModelRow, mode_key: str) -> list[str]:
        return [
            fmt_rate_pct(get_nested_field(_metrics_for_version(row, mode_key, ver, req_by_version, inline_by_version), "compat_bcb", "pass_rate"))
            for ver in versions
        ]

    right_col = 2 + len(versions)
    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=right_col)
    header_cols = " & ".join(rf"\textbf{{{ver}}}" for ver in versions)
    ver_join = ", ".join(versions)
    return (
        rf"""
\begin{{table*}}[t]
\centering
\caption{{BCB pass rate under different Python versions. Columns report task-level BCB pass rate (\%) for Python {ver_join}."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_bcb_pass_by_python}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{ll{"r" * len(versions)}}}
\toprule
& \textbf{{Model}} & {header_cols} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table_compat_by_python_versions(
    rows: list[ModelRow],
    versions: Sequence[str],
    req_by_version: dict[str, dict[str, dict[str, Any]]],
    inline_by_version: dict[str, dict[str, dict[str, Any]]],
) -> str:
    def values(row: ModelRow, mode_key: str) -> list[str]:
        out: list[str] = []
        for ver in versions:
            m = _metrics_for_version(row, mode_key, ver, req_by_version, inline_by_version)
            out.extend(
                [
                    fmt_rate_pct(get_nested_field(m, "compat_ty", "task_compat_rate")),
                    fmt_rate_pct(get_nested_field(m, "compat_bcb", "pass_rate")),
                ]
            )
        return out

    right_col = 2 + len(versions) * 2
    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=right_col)
    col_spec = "ll" + "r" * (len(versions) * 2)
    group_headers = " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{ver}}}}}" for ver in versions)
    group_rules = "".join(
        rf"\cmidrule(lr){{{3 + i * 2}-{4 + i * 2}}}" for i in range(len(versions))
    )
    metric_headers = " & ".join(r"$\tau_C$ & BCB" for _ in versions)
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Cross-Python compatibility results. Columns report static compatibility ($\tau_C$(\%)), BigCodeBench pass rate (\%) for Python 3.8--3.14."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:pipeline_d1_compat_by_python}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{{col_spec}}}
\toprule
& \textbf{{Model}} & {group_headers} \\
{group_rules}
& & {metric_headers} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table_ablation_security(rows: list[ModelRow], mode_metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    ordered = ordered_rows_commercial_first(rows)
    variants: tuple[tuple[str, str], ...] = (
        ("Baseline", "inline"),
        ("abl-instruct", "inline_no_vuln"),
        ("abl-version", "inline_safe_version"),
        ("abl-rag", "inline_api_rag"),
    )

    lines: list[str] = []
    last_ci = last_commercial_index(ordered)
    for idx, row in enumerate(ordered):
        vals: list[str] = []
        for _, mode in variants:
            m = mode_metrics.get(mode, {}).get(row.model_key)
            vals.extend(
                [
                    fmt_rate_pct_or_fixme(get_vulnerability_field(m, "lib_vuln_rate")),
                    fmt_rate_pct_or_fixme(get_vulnerability_field(m, "task_vuln_exposure")),
                ]
            )
        lines.append(f"{latex_escape_model(row.model)} & " + " & ".join(vals) + r" \\")
        if last_ci >= 0 and idx == last_ci:
            lines.append(r"\cmidrule{1-9}")
    while lines and lines[-1].startswith(r"\cmidrule"):
        lines.pop()
    body = "\n".join(lines)
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Security metrics across ablation conditions (inline mode, all models).
$\rho_U$: library vulnerability rate. $\tau_U$: task vulnerability exposure."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:ablation_security}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{\textbf{{Baseline}}}} & \multicolumn{{2}}{{c}}{{\textbf{{abl-instruct}}}} & \multicolumn{{2}}{{c}}{{\textbf{{abl-version}}}} & \multicolumn{{2}}{{c}}{{\textbf{{abl-rag}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}\cmidrule(lr){{8-9}}
\textbf{{Model}} & $\rho_U$(\%) & $\tau_U$(\%) & $\rho_U$(\%) & $\tau_U$(\%) & $\rho_U$(\%) & $\tau_U$(\%) & $\rho_U$(\%) & $\tau_U$(\%) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def table_ablation_compat(rows: list[ModelRow], mode_metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    ordered = ordered_rows_commercial_first(rows)
    variants: tuple[tuple[str, str], ...] = (
        ("abl-instruct", "inline_no_vuln"),
        ("abl-version", "inline_safe_version"),
        ("abl-rag", "inline_api_rag"),
    )

    lines: list[str] = []
    last_ci = last_commercial_index(ordered)
    for idx, row in enumerate(ordered):
        base = mode_metrics.get("inline", {}).get(row.model_key)
        base_ty = get_nested_field(base, "compat_ty", "task_compat_rate")
        base_bcb = get_nested_field(base, "compat_bcb", "pass_rate")
        vals = [fmt_rate_pct_or_fixme(base_ty), fmt_rate_pct_or_fixme(base_bcb)]
        for _, mode in variants:
            cur = mode_metrics.get(mode, {}).get(row.model_key)
            cur_ty = get_nested_field(cur, "compat_ty", "task_compat_rate")
            cur_bcb = get_nested_field(cur, "compat_bcb", "pass_rate")
            vals.extend(
                [
                    fmt_rate_pct_with_delta_pp_inline(cur_ty, base_ty),
                    fmt_rate_pct_with_delta_pp_inline(cur_bcb, base_bcb),
                ]
            )
        lines.append(f"{latex_escape_model(row.model)} & " + " & ".join(vals) + r" \\")
        if last_ci >= 0 and idx == last_ci:
            lines.append(r"\cmidrule{1-9}")
    while lines and lines[-1].startswith(r"\cmidrule"):
        lines.pop()
    body = "\n".join(lines)
    return (
        r"""
\begin{table*}[t]
\centering
\caption{Static compatibility ($\tau_C$, \%) and BigCodeBench dynamic pass rate (\%) across ablation conditions (inline mode, all models).
Values in parentheses give percentage-point change versus Baseline."""
        + D1_OSV_SUBSET_CAPTION_TAIL
        + rf"""}}
\label{{tab:ablation_compat}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{lrr rr rr rr}}
\toprule
 & \multicolumn{{2}}{{c}}{{\textbf{{Baseline}}}}
 & \multicolumn{{2}}{{c}}{{\textbf{{abl-instruct}}}}
 & \multicolumn{{2}}{{c}}{{\textbf{{abl-version}}}}
 & \multicolumn{{2}}{{c}}{{\textbf{{abl-rag}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}\cmidrule(lr){{8-9}}
\textbf{{Model}}
  & $\tau_C$ & BCB
  & $\tau_C$ & BCB
  & $\tau_C$ & BCB
  & $\tau_C$ & BCB \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""
    ).strip()


def main() -> None:
    args = parse_args()
    outputs_d1: Path = args.outputs_d1

    osv_task_ids = load_osv_bcb_task_ids(args.osv_bcb_tasks_jsonl)
    m5_req = discover_d1_m5_paths(outputs_d1 / "requirements.txt", args.python_version)
    m5_inl = discover_d1_m5_paths(outputs_d1 / "inline", args.python_version)

    req_metrics = augment_mode_metrics_with_osv_subset(
        load_mode_metrics(outputs_d1 / "requirements.txt", args.python_version),
        m5_req,
        osv_task_ids,
    )
    inline_metrics = augment_mode_metrics_with_osv_subset(
        load_mode_metrics(outputs_d1 / "inline", args.python_version),
        m5_inl,
        osv_task_ids,
    )
    rows = build_rows(req_metrics, inline_metrics)

    req_by_version = {
        ver: augment_mode_metrics_with_osv_subset(
            load_mode_metrics(outputs_d1 / "requirements.txt", ver),
            discover_d1_m5_paths(outputs_d1 / "requirements.txt", ver),
            osv_task_ids,
        )
        for ver in PYTHON_VERSIONS
    }
    inline_by_version = {
        ver: augment_mode_metrics_with_osv_subset(
            load_mode_metrics(outputs_d1 / "inline", ver),
            discover_d1_m5_paths(outputs_d1 / "inline", ver),
            osv_task_ids,
        )
        for ver in PYTHON_VERSIONS
    }
    rows_by_version = build_rows_from_versioned_metrics(req_by_version, inline_by_version)

    ablation_mode_metrics = {
        mode: augment_mode_metrics_with_osv_subset(
            load_ablation_mode_metrics(outputs_d1, mode, ABLATION_PYTHON_VERSION),
            discover_d1_m5_paths(outputs_d1 / mode, ABLATION_PYTHON_VERSION),
            osv_task_ids,
        )
        for mode in ABLATION_MODES
    }
    ablation_rows = build_rows(
        {},
        {
            k: v
            for mode_data in ablation_mode_metrics.values()
            for k, v in mode_data.items()
        },
    )

    sections = [
        "% Auto-computed by plots/generate_pipeline_d1_latex_tables.py",
        table1(rows),
        "",
        table3(rows),
        "",
        table_compat_by_python_versions(rows_by_version, PYTHON_VERSIONS, req_by_version, inline_by_version),
        "",
        table_bcb_osv_matrix_tasks(rows),
        "",
        table_ty_errors(rows),
        "",
        table_bcb_errors(rows),
        "",
        table_ablation_security(ablation_rows, ablation_mode_metrics),
        "",
        table_ablation_compat(ablation_rows, ablation_mode_metrics),
        "",
    ]
    content = "\n".join(sections)

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(content, encoding="utf-8")

    print(content)
    print(f"\n% Written to: {args.output_tex}")


if __name__ == "__main__":
    main()
