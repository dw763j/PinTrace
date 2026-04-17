#!/usr/bin/env python3
"""
Generate LaTeX tables for pipeline_d2 experiment results.

D2 has two pinning layouts, for example:
- ``outputs/d2/inline/d2_<model>_blind_inline/py<ver>/metrics_summary.json``
- ``outputs/d2/requirements.txt/d2_<model>_blind_requirements_txt/py<ver>/metrics_summary.json``

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m plots.generate_pipeline_d2_latex_tables --python-version 3.12
    python -m plots.generate_pipeline_d2_latex_tables \\
        --neighbor-agg-json outputs/d2/neighbor_experiments/all_models_safe_ty_install_error_py312.json

The neighbor-version aggregate table defaults to
``outputs/d2/neighbor_experiments/all_models_safe_ty_install_error_py312.json``; if missing, that
table is skipped and a LaTeX comment line is emitted.

Ty overview columns come from ``aggregate_d2_ty_primary_rule_counts`` /
``build_ty_rule_overview_stats``: scan ``m5_compat_records.json`` under ``inline`` and
``requirements.txt``, count the first ``compat_results.ty.errors[0].rule`` on each failing task,
aggregate globally, then take top-K rules as primary columns.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from plots.model_display import MODEL_LABEL, normalize_model_key, order_model_keys, paper_display_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
PLOTS = PROJECT_ROOT / "plots"
# Neighbor-version experiment aggregate (``run_neighbor_version_experiment.py --all-models``).
DEFAULT_NEIGHBOR_AGG_JSON = (
    OUTPUTS / "d2" / "neighbor_experiments" / "all_models_safe_ty_install_error_py312.json"
)
# Stable row order for the paper table (matches ``MODEL_LABEL`` insertion order).
NEIGHBOR_TABLE_MODEL_KEYS: tuple[str, ...] = tuple(normalize_model_key(k) for k in MODEL_LABEL)
PYTHON_VERSIONS: tuple[str, ...] = ("3.8", "3.10", "3.12", "3.14")
ABLATION_PYTHON_VERSION = "3.12"
FIXME_CELL = r"\FIXME{}"

# When no m5_compat_records are found, ty overview columns fall back to this order (typical D2 ranking).
_DEFAULT_TY_RULE_OVERVIEW: tuple[str, ...] = (
    "ty-install-error",
    "unresolved-attribute",
    "unresolved-import",
    "invalid-syntax",
    "invalid-argument-type",
    "unresolved-reference",
)


@dataclass(frozen=True)
class D2ModelRow:
    model_key: str
    model: str
    requirements: dict[str, Any] | None
    inline: dict[str, Any] | None

    @property
    def group(self) -> str:
        lower = self.model.lower()
        if lower.startswith(("gpt", "gemini", "claude")):
            return "Commercial"
        return "Open Source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from outputs/d2 metrics_summary.json."
    )
    parser.add_argument(
        "--outputs-d2",
        type=Path,
        default=OUTPUTS / "d2",
        help="Root outputs/d2 directory.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=PLOTS / "pipeline_d2_paper_tables.tex",
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
        "--ty-rule-top-k",
        type=int,
        default=6,
        help=(
            "Number of primary ty diagnostic rules to show as columns in ty error overview "
            "(chosen by global frequency over m5_compat_records.json for inline + requirements.txt)."
        ),
    )
    parser.add_argument(
        "--neighbor-agg-json",
        type=Path,
        default=DEFAULT_NEIGHBOR_AGG_JSON,
        help=(
            "Path to all_models_*.json written by run_neighbor_version_experiment.py "
            "(neighbor search on safe_ty_install_error tasks). When missing, the neighbor table is omitted."
        ),
    )
    return parser.parse_args()


def normalize_model_name(name: str) -> str:
    return name.strip().lower()


def latex_escape_header(label: str) -> str:
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


def pick_display_name(
    req_obj: dict[str, Any] | None, inline_obj: dict[str, Any] | None, normalized_name: str
) -> str:
    raw_name = normalized_name
    for obj in (req_obj, inline_obj):
        if obj and obj.get("model_name"):
            raw_name = str(obj["model_name"])
            break
    return paper_display_label(normalized_name, raw_name)


def build_rows(req_metrics: dict[str, dict[str, Any]], inline_metrics: dict[str, dict[str, Any]]) -> list[D2ModelRow]:
    model_keys = order_model_keys(set(req_metrics) | set(inline_metrics))
    rows: list[D2ModelRow] = []
    for key in model_keys:
        req_obj = req_metrics.get(key)
        inline_obj = inline_metrics.get(key)
        rows.append(
            D2ModelRow(
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
) -> list[D2ModelRow]:
    model_keys: set[str] = set()
    for vmap in req_by_version.values():
        model_keys.update(vmap.keys())
    for vmap in inline_by_version.values():
        model_keys.update(vmap.keys())

    rows: list[D2ModelRow] = []
    for key in order_model_keys(model_keys):
        req_obj: dict[str, Any] | None = None
        inline_obj: dict[str, Any] | None = None
        for ver in PYTHON_VERSIONS:
            req_obj = req_obj or req_by_version.get(ver, {}).get(key)
            inline_obj = inline_obj or inline_by_version.get(ver, {}).get(key)
        rows.append(
            D2ModelRow(
                model_key=key,
                model=pick_display_name(req_obj, inline_obj, key),
                requirements=req_obj,
                inline=inline_obj,
            )
        )
    return rows


def load_mode_metrics_by_versions(
    mode_dir: Path,
    versions: tuple[str, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {ver: load_mode_metrics(mode_dir, ver) for ver in versions}


def load_ablation_mode_metrics(outputs_d2: Path, mode: str, python_version: str) -> dict[str, dict[str, Any]]:
    mode_dir = outputs_d2 / mode
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


def discover_m5_paths_by_model(mode_dir: Path, python_version: str | None) -> dict[str, Path]:
    """Map normalized model name -> m5_compat_records.json next to metrics_summary.json."""
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


def primary_ty_rule_counts_from_m5(m5_path: Path) -> Counter[str]:
    """Per task, count the first ty error's ``rule`` (same convention as failure classification)."""
    c: Counter[str] = Counter()
    try:
        raw = m5_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return c
    if not isinstance(data, list):
        return c
    for rec in data:
        ty = (rec.get("compat_results") or {}).get("ty") if isinstance(rec.get("compat_results"), dict) else None
        if not isinstance(ty, dict):
            continue
        if ty.get("is_compatible"):
            continue
        errors = ty.get("errors") or []
        if not errors or not isinstance(errors[0], dict):
            c["(no_rule)"] += 1
            continue
        rule = errors[0].get("rule")
        c[str(rule if rule is not None else "(no_rule)")] += 1
    return c


def aggregate_d2_ty_primary_rule_counts(
    outputs_d2: Path,
    python_version: str | None = None,
    *,
    modes: tuple[tuple[str, str], ...] = (
        ("requirements", "requirements.txt"),
        ("inline", "inline"),
    ),
) -> dict[str, Any]:
    """
    Count primary ``ty`` diagnostic rules on failing tasks (``compat_results.ty.errors[0].rule``).

    Returns:
      - ``global_primary_rule_totals``: summed counts per rule across scanned runs
      - ``by_mode_and_model``: mode -> normalized_model -> rule -> count
    """
    global_totals: Counter[str] = Counter()
    by_mode_and_model: dict[str, dict[str, dict[str, int]]] = {}

    for mode_key, subdir in modes:
        mode_dir = outputs_d2 / subdir
        paths = discover_m5_paths_by_model(mode_dir, python_version)
        by_mode_and_model.setdefault(mode_key, {})
        for mk, m5 in paths.items():
            ctr = primary_ty_rule_counts_from_m5(m5)
            by_mode_and_model[mode_key][mk] = dict(ctr)
            global_totals.update(ctr)

    return {
        "global_primary_rule_totals": dict(global_totals),
        "by_mode_and_model": by_mode_and_model,
        "python_version": python_version,
        "outputs_d2": str(outputs_d2.resolve()),
    }


@dataclass(frozen=True)
class TyRuleOverviewStats:
    """For LaTeX ty overview: top global rules + per (mode, model) primary-rule counts."""

    main_rules: tuple[str, ...]
    per_mode_model: dict[tuple[str, str], Counter[str]]


def build_ty_rule_overview_stats(
    outputs_d2: Path,
    python_version: str | None,
    *,
    top_k: int,
) -> TyRuleOverviewStats:
    """Pick top-K rules by global frequency; attach per-run counters for table cells."""
    raw = aggregate_d2_ty_primary_rule_counts(outputs_d2, python_version)
    global_totals: Counter[str] = Counter(raw.get("global_primary_rule_totals") or {})
    if global_totals:
        main_rules = tuple(r for r, _ in global_totals.most_common(top_k))
    else:
        n = min(max(0, top_k), len(_DEFAULT_TY_RULE_OVERVIEW))
        main_rules = _DEFAULT_TY_RULE_OVERVIEW[:n] if n else tuple()

    per: dict[tuple[str, str], Counter[str]] = {}
    bmm = raw.get("by_mode_and_model") or {}
    if isinstance(bmm, dict):
        for mode_key, models in bmm.items():
            if not isinstance(models, dict):
                continue
            for mk, rule_map in models.items():
                if not isinstance(rule_map, dict):
                    continue
                per[(str(mode_key), str(mk))] = Counter({str(k): int(v or 0) for k, v in rule_map.items()})

    return TyRuleOverviewStats(main_rules=main_rules, per_mode_model=per)


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


def fmt_count_pct_vs_qualifying(count: Any, qualifying: Any) -> str:
    """Format ``count (pct\\%)`` with percentage relative to qualifying baseline tasks."""
    try:
        c = int(count)
        q = int(qualifying)
    except (TypeError, ValueError):
        return "--"
    if q <= 0:
        return str(c)
    pct = 100.0 * c / q
    return f"{c} ({pct:.2f}\\%)"


def fmt_float2(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "--"


def fmt_high_risk_ratio(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "--"
    vuln = metrics.get("vulnerability") if isinstance(metrics, dict) else None
    sev = vuln.get("severity_dist") if isinstance(vuln, dict) else None
    if not isinstance(sev, dict):
        return "--"
    try:
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


def tpl_lib_uses_count(metrics: dict[str, Any] | None) -> int:
    """#LibUses in table1: ``version_results.total_lib`` (TPL mentions per model/mode)."""
    v = get_version_field(metrics, "total_lib")
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


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


def latex_escape_model(name: str) -> str:
    return name.replace("_", r"\_")


def model_commercial_open_group_from_display(display: str) -> str:
    lower = display.lower()
    if lower.startswith(("gpt", "gemini", "claude")):
        return "Commercial"
    return "Open Source"


def display_label_for_neighbor_model_key(mk: str) -> str:
    for canon, label in MODEL_LABEL.items():
        if normalize_model_key(canon) == mk:
            return label
    return mk


def infer_neighbor_run_model_key(run: dict[str, Any]) -> str:
    sr = run.get("selected_run")
    if not isinstance(sr, dict):
        return ""
    mn = str(sr.get("model_name") or "").strip()
    if mn.startswith("d2_") and mn.endswith("_blind"):
        short = mn[3:-6]
    else:
        short = mn
    return normalize_model_key(short)


def load_neighbor_agg_by_model_key(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    runs = data.get("runs")
    if not isinstance(runs, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        mk = infer_neighbor_run_model_key(run)
        if mk:
            out[mk] = run
    return out


def last_commercial_index_in_neighbor_table_order() -> int:
    last = -1
    for idx, mk in enumerate(NEIGHBOR_TABLE_MODEL_KEYS):
        disp = display_label_for_neighbor_model_key(mk)
        if model_commercial_open_group_from_display(disp) == "Commercial":
            last = idx
    return last


def neighbor_run_table_cells(run: dict[str, Any] | None) -> list[str]:
    """One row: Qual., Install (count+pct), Compat (count+pct), Vuln@Reco, mean tested candidates."""
    if not run or not isinstance(run, dict):
        return ["--", "--", "--", "--", "--"]
    summary = run.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    qual: Any = summary.get("baseline_tasks")
    if qual is None:
        sr = run.get("selected_run")
        if isinstance(sr, dict):
            qual = sr.get("safe_ty_install_error_count")
    inst = summary.get("recover_no_ty_install_error_tasks")
    compat = summary.get("recover_ty_tasks")
    vuln = summary.get("recover_no_ty_install_error_vuln_tasks")
    avg_test = summary.get("avg_tested_candidates_per_task")
    return [
        fmt_count(qual),
        fmt_count_pct_vs_qualifying(inst, qual),
        fmt_count_pct_vs_qualifying(compat, qual),
        fmt_count(vuln),
        fmt_float2(avg_test),
    ]


def table_neighbor_ty_install_error_recovery(neighbor_by_key: dict[str, dict[str, Any]]) -> str:
    """
    Neighbor-search recovery on ``safe_ty_install_error`` tasks (inline, py3.12 aggregate JSON).

    - **Qual.**: qualifying baseline tasks (ty-install-error under safe+ty-incompat selection).
    - **Install.**: recovered tasks without ty-install-error in the neighborhood; pct of Qual.
    - **Compat.**: recovered tasks that pass ty; pct of Qual.
    - **Vuln@Reco**: recovered install without ty-install-error but flagged vulnerable (security regression).
    - **Nbar_test**: mean evaluated neighbor candidates per qualifying task ($\\overline{N}_{\\mathrm{test}}$ in the table).
    """
    ncols = 1 + len(neighbor_run_table_cells(None))
    lines: list[str] = []
    last_ci = last_commercial_index_in_neighbor_table_order()
    for idx, mk in enumerate(NEIGHBOR_TABLE_MODEL_KEYS):
        disp = latex_escape_model(display_label_for_neighbor_model_key(mk))
        cells = neighbor_run_table_cells(neighbor_by_key.get(mk))
        lines.append(f"{disp} & " + " & ".join(cells) + r" \\")
        if last_ci >= 0 and idx == last_ci:
            lines.append(rf"\cmidrule[0.5pt](lr){{1-{ncols}}}")
    while lines and lines[-1].startswith(r"\cmidrule"):
        lines.pop()
    body = "\n".join(lines)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Neighbor-version search on tasks that has package installation error under the Stack Overflow dataset.
\textbf{{Qualifying}} is the number of qualifying tasks;
\textbf{{Installable}} counts tasks with a neighbor that removes the package installation error (percentage of Qualifying tasks);
\textbf{{Compatible}} counts tasks with a neighbor that passes static type-checking (percentage of Qualifying tasks);
\textbf{{Vuln@Reco}} counts install recoveries that introduce vulnerable dependencies;
$\overline{{N}}_{{\mathrm{{test}}}}$ is the mean number of neighbor candidates statically evaluated per qualifying task.
Reported recoveries are lower bounds under the configured search budget and ranking.}}
\label{{tab:neighbor_ty_install_error_recovery_d2_py312}}
\begin{{tabular}}{{lrrrrr}}
\toprule
\textbf{{Model}} & \textbf{{Qualifying}} & \textbf{{Installable}} & \textbf{{Compatible}} & \textbf{{Vuln@Reco}} & $\mathbf{{\overline{{N}}_{{\mathrm{{test}}}}}}$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip()


def ordered_rows_commercial_first(rows: list[D2ModelRow]) -> list[D2ModelRow]:
    return [r for r in rows if r.group == "Commercial"] + [r for r in rows if r.group == "Open Source"]


def last_commercial_index(ordered: list[D2ModelRow]) -> int:
    last = -1
    for idx, r in enumerate(ordered):
        if r.group == "Commercial":
            last = idx
    return last


def render_dual_mode_table_body(
    rows: list[D2ModelRow],
    value_fn: Callable[[D2ModelRow, str], list[str]],
    cmidrule_from_col: int,
    cmidrule_to_col: int,
) -> str:
    """Rows grouped by mode column: Explicit + Inline."""
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


def table1(rows: list[D2ModelRow]) -> str:
    def values(row: D2ModelRow, mode_key: str) -> list[str]:
        m = row.inline if mode_key == "inline" else row.requirements
        return [
            fmt_count(get_field(m, "total_tasks")),
            fmt_count(get_version_field(m, "total_lib")),
            fmt_rate_pct(get_version_field(m, "lib_version_spec_rate")),
            fmt_rate_pct(get_version_field(m, "lib_version_validity_rate")),
            fmt_rate_pct(get_vulnerability_field(m, "lib_vuln_rate")),
            fmt_high_risk_ratio(m),
            fmt_rate_pct(get_vulnerability_field(m, "task_vuln_exposure")),
        ]

    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=9)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Measurement results of version specification behavior and vulnerability exposure in the LLM code generation process. \#LibUses denotes the total number of third-party library mentions across tasks. Specified(\%) reports the proportion of mentions with explicit version declarations, Valid(\%) reports the proportion of declared versions that are resolvable to valid releases, LibVuln(\%) reports the share of resolved library versions with known vulnerabilities, and TaskVuln(\%) reports the proportion of tasks exposed to at least one vulnerable dependency.}}
\label{{tab:rq1_rq2_combined}}
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
""".strip()


def print_table1_tpl_totals(rows: list[D2ModelRow]) -> None:
    """Sum ``total_lib`` over all models for each pinning mode (matches table1 #LibUses)."""
    explicit = sum(tpl_lib_uses_count(r.requirements) for r in rows)
    inline = sum(tpl_lib_uses_count(r.inline) for r in rows)
    print(
        "[table1] Total TPL usages (#LibUses = version_results.total_lib, summed over models): "
        f"Explicit {explicit}, Inline {inline}, combined {explicit + inline}"
    )


def table_compat(rows: list[D2ModelRow]) -> str:
    def values(row: D2ModelRow, mode_key: str) -> list[str]:
        m = row.inline if mode_key == "inline" else row.requirements
        return [
            fmt_rate_pct(get_nested_field(m, "compat_ty", "task_compat_rate")),
            fmt_rate_pct(get_compat_security_rate(m, "task_safe_ty_compat")),
            fmt_rate_pct(get_compat_security_rate(m, "task_unsafe_ty_compat")),
            fmt_rate_pct(get_compat_security_rate(m, "task_safe_ty_incompat")),
            fmt_rate_pct(get_compat_security_rate(m, "task_unsafe_ty_incompat")),
        ]

    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=7)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Measurement results of compatibility between LLM code generation outputs and its specified third-party library versions. 
Compat(\%) represents the percentage of tasks that pass the static type-checking. 
Quadrant entries categorize tasks based on the intersection of type-checking compatibility (Compat vs.\ Incompat) and vulnerability status (Safe vs.\ Unsafe).}}
\label{{tab:pipeline_d2_compat_quadrants}}
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
""".strip()


def _metrics_for_version(
    row: D2ModelRow,
    mode_key: str,
    version: str,
    req_by_version: dict[str, dict[str, dict[str, Any]]],
    inline_by_version: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    by_ver = inline_by_version if mode_key == "inline" else req_by_version
    return by_ver.get(version, {}).get(row.model_key)


def table_ty_compat_by_python_versions(
    rows: list[D2ModelRow],
    versions: tuple[str, ...],
    req_by_version: dict[str, dict[str, dict[str, Any]]],
    inline_by_version: dict[str, dict[str, dict[str, Any]]],
) -> str:
    def values(row: D2ModelRow, mode_key: str) -> list[str]:
        return [
            fmt_rate_pct(
                get_nested_field(
                    _metrics_for_version(row, mode_key, ver, req_by_version, inline_by_version),
                    "compat_ty",
                    "task_compat_rate",
                )
            )
            for ver in versions
        ]

    right_col = 2 + len(versions)
    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=right_col)
    header_cols = " & ".join(rf"\textbf{{{ver}}}" for ver in versions)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Ty compatibility rate under different Python versions. Columns report task-level ty compatibility rate (\%) for Python {", ".join(versions)}.}}
\label{{tab:pipeline_d2_ty_compat_by_python}}
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
""".strip()


def table_ablation_security(rows: list[D2ModelRow], mode_metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
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
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Security metrics across ablation conditions.
$\rho_U$(\%): library vulnerability rate. $\tau_U$(\%): task vulnerability exposure.}}
\label{{tab:ablation_security}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{\textbf{{Baseline}}}} & \multicolumn{{2}}{{c}}{{\textbf{{abl-instruct}}}} & \multicolumn{{2}}{{c}}{{\textbf{{abl-version}}}} & \multicolumn{{2}}{{c}}{{\textbf{{abl-rag}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}\cmidrule(lr){{8-9}}
\textbf{{Model}} & $\rho_U$ & $\tau_U$ & $\rho_U$ & $\tau_U$ & $\rho_U$ & $\tau_U$ & $\rho_U$ & $\tau_U$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
""".strip()


# Install-error cluster keys (metrics ``install_error_clusters``); remainder -> others; last column = total install_error tasks.
INSTALL_ERROR_DETAIL_MAIN_KEYS: tuple[str, ...] = (
    "build_backend_failure",
    "dependency_resolve_no_solution",
    "missing_distutils",
)

INSTALL_ERROR_DETAIL_DISPLAY_LABELS: tuple[str, ...] = (
    "Build Backend Failure",
    "Dependency Resolve",
    "Missing Distutils",
)


def _install_error_others_count(task_dist: dict[str, Any], clusters: dict[str, Any]) -> int:
    """All cluster mass not in the main keys, plus uncategorized install_error tasks."""
    install_total = int(task_dist.get("install_error", 0) or 0)
    main_sum = sum(int((clusters or {}).get(k, 0) or 0) for k in INSTALL_ERROR_DETAIL_MAIN_KEYS)
    return max(0, install_total - main_sum)


def table_ty_error_overview(rows: list[D2ModelRow], rule_stats: TyRuleOverviewStats) -> str:
    """Primary ty diagnostic rule (first error rule per failing task) from m5_compat_records; top-K global + others + total."""

    main_rules = rule_stats.main_rules
    per = rule_stats.per_mode_model
    n_data = len(main_rules) + 2

    def values(row: D2ModelRow, mode_key: str) -> list[str]:
        ctr = per.get((mode_key, row.model_key), Counter())
        out = [fmt_count(ctr.get(r, 0)) for r in main_rules]
        main_sum = sum(int(ctr.get(r, 0) or 0) for r in main_rules)
        total = sum(int(v or 0) for v in ctr.values())
        others = max(0, total - main_sum)
        out.extend([fmt_count(others), fmt_count(total)])
        return out

    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=1 + n_data + 1)
    header_cells = [rf"\textbf{{{latex_escape_header(r)}}}" for r in main_rules] + [r"\textbf{Others}", r"\textbf{Total}"]
    header_rest = " & ".join(header_cells)
    col_spec = "ll" + "r" * n_data
    rules_line = rf" & \textbf{{Model}} & {header_rest} \\"
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Overview of the static compatibility check results. For each incompatible task, the first reported rule is used as its primary diagnostic label. The first {len(main_rules)} columns correspond to the globally most frequent rules, \textbf{{Others}} aggregates remaining low-frequency rules, and \textbf{{Total}} denotes the number of all the ty-incompatible tasks.}}
\label{{tab:pipeline_d2_ty_error_overview}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{{col_spec}}}
\toprule
{rules_line}
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
""".strip()


def table_ty_install_error_detail(rows: list[D2ModelRow]) -> str:
    """Fixed install clusters + others + total (install_error task count from metrics)."""

    def render_body() -> str:
        ordered = ordered_rows_commercial_first(rows)
        if not ordered:
            return ""

        last_ci = last_commercial_index(ordered)
        lines: list[str] = []
        n_data_cols = len(INSTALL_ERROR_DETAIL_MAIN_KEYS) + 2

        for mode_label, mode_key in (("Explicit", "requirements"), ("Inline", "inline")):
            first_row_in_group = True
            for idx, row in enumerate(ordered):
                left_col = ""
                if first_row_in_group:
                    left_col = rf"\multirow{{{len(ordered)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{mode_label}}}}}}}"
                    first_row_in_group = False

                m = row.inline if mode_key == "inline" else row.requirements
                ty_error = get_nested_field(m, "compat_ty", "ty_error")
                task_dist = (ty_error or {}).get("task_dist") or {}
                install_clusters = (ty_error or {}).get("install_error_clusters") or {}
                if not isinstance(install_clusters, dict):
                    install_clusters = {}

                install_total = int(task_dist.get("install_error", 0) or 0)
                main_vals = [fmt_count(install_clusters.get(k)) for k in INSTALL_ERROR_DETAIL_MAIN_KEYS]
                others_c = _install_error_others_count(task_dist, install_clusters)

                model_col = latex_escape_model(row.model)
                vals = [*main_vals, fmt_count(others_c), fmt_count(install_total)]
                lines.append(f"{left_col} & {model_col} & " + " & ".join(vals) + r" \\")

                if last_ci >= 0 and idx == last_ci:
                    right_col = 2 + n_data_cols
                    lines.append(rf"\cmidrule[0.5pt](lr){{2-{right_col}}}")

            if mode_label == "Explicit":
                lines.append(r"\midrule")

        while lines and (
            lines[-1] == r"\midrule" or lines[-1].startswith(r"\cmidrule[0.5pt](lr)")
        ):
            lines.pop()
        return "\n".join(lines)

    body = render_body()
    n_data_cols = len(INSTALL_ERROR_DETAIL_MAIN_KEYS) + 2
    col_spec = "ll" + "r" * n_data_cols

    sub_headers = [rf"\textbf{{{label}}}" for label in INSTALL_ERROR_DETAIL_DISPLAY_LABELS] + [
        r"\textbf{Others}",
        r"\textbf{Total}",
    ]
    header_rest = " & ".join(sub_headers)

    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Breakdown of dependency-installation failures. The major installation-failure clusters are reported explicitly, \textbf{{Others}} aggregates remaining uncategorized installation failures, and \textbf{{Total}} denotes all tasks with installation failures under static checking.}}
\label{{tab:pipeline_d2_ty_install_error_detail}}
\resizebox{{\linewidth}}{{!}}{{
\begin{{tabular}}{{{col_spec}}}
\toprule
& \textbf{{Model}} & {header_rest} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table*}}
""".strip()


def table_bcb(rows: list[D2ModelRow]) -> str:
    """Retained for future use; D2 currently has no BCB test results."""

    def values(row: D2ModelRow, mode_key: str) -> list[str]:
        m = row.inline if mode_key == "inline" else row.requirements
        return [
            fmt_count(get_nested_field(m, "compat_bcb", "task_pass")),
            fmt_count(get_nested_field(m, "compat_bcb", "task_fail")),
            fmt_count(get_nested_field(m, "compat_bcb", "task_error_like")),
            fmt_count(get_nested_field(m, "compat_bcb", "task_total")),
            fmt_rate_pct(get_nested_field(m, "compat_bcb", "pass_rate")),
        ]

    body = render_dual_mode_table_body(rows, values, cmidrule_from_col=2, cmidrule_to_col=7)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{BigCodeBench (BCB) execution outcomes by model. \#Pass, \#Fail, and \#Error are task-level test outcomes, \#Total is the number of evaluated tasks, and PassRate(\%) is the percentage of passing tasks among evaluated tasks.}}
\label{{tab:pipeline_d2_bcb_results}}
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
""".strip()


def latex_comment_block(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        if line:
            out.append("% " + line)
        else:
            out.append("%")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    root: Path = args.outputs_d2

    req_metrics = load_mode_metrics(root / "requirements.txt", args.python_version)
    inline_metrics = load_mode_metrics(root / "inline", args.python_version)
    rows = build_rows(req_metrics, inline_metrics)
    req_by_version = load_mode_metrics_by_versions(root / "requirements.txt", PYTHON_VERSIONS)
    inline_by_version = load_mode_metrics_by_versions(root / "inline", PYTHON_VERSIONS)
    rows_by_version = build_rows_from_versioned_metrics(req_by_version, inline_by_version)
    ablation_mode_metrics = {
        "inline": load_ablation_mode_metrics(root, "inline", ABLATION_PYTHON_VERSION),
        "inline_no_vuln": load_ablation_mode_metrics(root, "inline_no_vuln", ABLATION_PYTHON_VERSION),
        "inline_safe_version": load_ablation_mode_metrics(root, "inline_safe_version", ABLATION_PYTHON_VERSION),
        "inline_api_rag": load_ablation_mode_metrics(root, "inline_api_rag", ABLATION_PYTHON_VERSION),
    }
    ablation_rows = build_rows(
        {},
        {
            k: v
            for mode_data in ablation_mode_metrics.values()
            for k, v in mode_data.items()
        },
    )
    rule_stats = build_ty_rule_overview_stats(root, args.python_version, top_k=args.ty_rule_top_k)

    neighbor_by_key = load_neighbor_agg_by_model_key(args.neighbor_agg_json)

    t1_tex = table1(rows)
    print_table1_tpl_totals(rows)
    sections = [
        "% Auto-computed by plots/generate_pipeline_d2_latex_tables.py",
        t1_tex,
        "",
        table_compat(rows),
        "",
        table_ty_compat_by_python_versions(rows_by_version, PYTHON_VERSIONS, req_by_version, inline_by_version),
        "",
        table_ablation_security(ablation_rows, ablation_mode_metrics),
        "",
        table_ty_error_overview(rows, rule_stats),
        "",
        table_ty_install_error_detail(rows),
        "",
    ]
    if neighbor_by_key:
        sections.extend(
            [
                table_neighbor_ty_install_error_recovery(neighbor_by_key),
                "",
            ]
        )
    else:
        sections.extend(
            [
                f"% Neighbor-version table skipped (no aggregate JSON): {args.neighbor_agg_json}",
                "",
            ]
        )
    sections.extend(
        [
            "% NOTE: D2 currently has no BCB test results; keep the table as commented LaTeX.",
            latex_comment_block(table_bcb(rows)),
            "",
        ]
    )
    content = "\n".join(sections)

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(content, encoding="utf-8")

    print(content)
    print(f"\n% Written to: {args.output_tex}")


if __name__ == "__main__":
    main()
