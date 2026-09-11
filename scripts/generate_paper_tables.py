"""Regenerate the manuscript tables from frozen evaluation records.

Run from the repository root::

    python -m scripts.generate_paper_tables

Reads canonical JSONL, withdrawn-advisory aggregates, neighbor recovery,
and Python-robustness CSVs already in this artifact, writes LaTeX/CSV under
``paper_results/tables/``, and asserts the manuscript headline counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from paths import ANALYSIS_NEIGHBOR, ANALYSIS_ROBUSTNESS, ANALYSIS_RQ2, EVALUATION, PAPER_RESULTS
from plots.model_display import MODEL_LABEL, paper_display_label
from plots.rq2_constraint_validity_table import (
    attach_dependencies,
    accumulate,
    load_clean_ids,
    load_observations,
    render_tex as render_rq2_validity_legacy_tex,
    write_csv as write_rq2_validity_csv,
)
from scripts.stats_so_e2e_dataset_content import (
    compute_content_stats,
    compute_target_package_stats,
    load_records_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = EVALUATION
WITHDRAWN_IMPACT = ANALYSIS_RQ2 / "withdrawn_impact.json"
NEIGHBOR_JSON = ANALYSIS_NEIGHBOR / "recovery.json"
ROBUST_CSV = ANALYSIS_ROBUSTNESS / "python_robustness_all_observations.csv"
PINTRACE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_builder"
    / "so_e2e_balanced"
    / "records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl"
)
CLEAN_IDS = PROJECT_ROOT / "resources" / "revision" / "pintrace_clean_task_ids.jsonl"
TIKTOKEN_CACHE = PROJECT_ROOT / "resources" / "tiktoken"
DEFAULT_OUTPUT = PAPER_RESULTS / "tables"

MODE_MR = "requirements.txt"
MODE_VA = "inline"
TRACK = "d2"
PYTHON_VERSION = "3.12"
RESOLVER_INPUT = frozenset({"resolved", "failed"})
INSTALL_FAIL = frozenset({"environment_build", "infrastructure"})
DISPLAY_ORDER = list(MODEL_LABEL.values())

EXPECTED_RQ1_FORMS = {
    "MR": {"N": 18995, "exact": 4177, "range": 5704, "bare": 9107, "invalid": 7},
    "VA": {"N": 25253, "exact": 19134, "bare": 6119},
}
EXPECTED_RQ1_TASK = {
    "All observations": {"MR Exact": 17.75, "MR Range": 26.15, "MR Bare-only": 43.12, "VA Exact": 81.19},
    "GPT-5.4": {"MR Exact": 2.70, "MR Range": 7.10, "MR Bare-only": 74.10, "VA Exact": 80.80},
    "Claude-Sonnet-4.6": {"MR Exact": 14.10, "MR Range": 10.40, "MR Bare-only": 66.70, "VA Exact": 86.30},
    "Gemini-3.1-Pro": {"MR Exact": 4.80, "MR Range": 7.90, "MR Bare-only": 81.60, "VA Exact": 89.50},
    "DeepSeek-V3.2": {"MR Exact": 36.00, "MR Range": 40.30, "MR Bare-only": 10.70, "VA Exact": 88.50},
    "Kimi-K2.5": {"MR Exact": 12.40, "MR Range": 69.20, "MR Bare-only": 9.70, "VA Exact": 83.80},
    "Qwen3.5-397B": {"MR Exact": 17.80, "MR Range": 34.90, "MR Bare-only": 40.20, "VA Exact": 90.10},
    "Qwen3-235B": {"MR Exact": 26.20, "MR Range": 40.40, "MR Bare-only": 25.30, "VA Exact": 83.00},
    "Qwen3-30B": {"MR Exact": 50.40, "MR Range": 26.70, "MR Bare-only": 1.40, "VA Exact": 45.60},
    "MiniMax-M2.5": {"MR Exact": 4.20, "MR Range": 24.40, "MR Bare-only": 61.00, "VA Exact": 87.00},
    "Llama-4-Scout": {"MR Exact": 8.90, "MR Range": 0.20, "MR Bare-only": 60.50, "VA Exact": 77.30},
}
EXPECTED_RQ2_VALIDITY = {
    ("All observations", "MR Exact"): (1207, 1271),
    ("All observations", "MR Range"): (2187, 2202),
    ("All observations", "VA Exact"): (6389, 6997),
    ("All observations", "MR All"): (3187, 3446),
    ("All observations", "VA All"): (5510, 6997),
}
EXPECTED_RQ2_CVE = {
    "All observations": {
        "MR Exact": (66.53, 9.46),
        "MR Range": (29.40, 7.57),
        "VA Exact": (52.48, 39.49),
    }
}
EXPECTED_RQ3 = {
    "MR": {"n": 10000, "inp": 9235, "lock": 8234, "inst": 7779, "stat": 5044},
    "VA": {"n": 10000, "inp": 8868, "lock": 6889, "inst": 5274, "stat": 3432},
}
EXPECTED_NEIGHBOR = {
    "GPT-5.4": (3, 1, 54),
    "Claude-Sonnet-4.6": (20, 15, 78),
    "Gemini-3.1-Pro": (68, 60, 145),
    "DeepSeek-V3.2": (67, 59, 163),
    "Kimi-K2.5": (49, 41, 133),
    "Qwen3.5-397B": (78, 69, 166),
    "Qwen3-235B": (74, 63, 155),
    "Qwen3-30B": (51, 44, 80),
    "MiniMax-M2.5": (70, 57, 165),
    "Llama-4-Scout": (24, 20, 151),
}
EXPECTED_NEIGHBOR_ALL = (504, 429, 1290)
EXPECTED_ROBUSTNESS = {
    ("PinTrace", "MR", "complete_lock"): (81.21, 83.40, 82.34, 80.49),
    ("PinTrace", "VA", "complete_lock"): (60.93, 72.15, 68.89, 68.36),
    ("PinTrace", "MR", "installation_completed"): (78.99, 81.78, 77.79, 73.21),
    ("PinTrace", "VA", "installation_completed"): (59.67, 70.81, 52.74, 41.84),
    ("PinTrace", "MR", "static_pass"): (53.42, 51.70, 50.44, 47.79),
    ("PinTrace", "VA", "static_pass"): (40.41, 47.66, 34.32, 26.67),
    ("BigCodeBench", "MR", "dynamic_pass"): (45.79, 45.38, 42.47, 41.69),
    ("BigCodeBench", "VA", "dynamic_pass"): (33.90, 32.04, 16.42, 9.04),
}
FROZEN_TEMPORAL = (
    ("2020", 313, 31.3, 177),
    ("2021", 246, 55.9, 153),
    ("2022", 194, 75.3, 133),
    ("2023", 131, 88.4, 101),
    ("2024", 76, 96.0, 74),
    ("2025", 40, 100.0, 49),
)
EXPECTED_DATASET = {
    "n_tasks": 1000,
    "distinct_tpl": 267,
    "ge2": 464,
    "ge3": 131,
    "question_tokens": (430.6, 286.5, 518.3),
    "code_lines": (24.9, 16.0, 39.3),
    "code_tokens": (202.0, 123.5, 298.8),
    "tpl_per_task": (1.6, 1.0, 0.8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _display(model: str) -> str:
    return paper_display_label(model, model)


def _pct(numerator: int, denominator: int, digits: int = 2) -> float:
    return round(100.0 * numerator / denominator, digits) if denominator else 0.0


def _cell(numerator: int, denominator: int) -> str:
    return f"{numerator:,}/{denominator:,} ({_pct(numerator, denominator):.2f}\\%)"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assert_eq(errors: list[str], name: str, got: Any, expected: Any) -> None:
    if got != expected:
        errors.append(f"{name}: got {got!r}, expected {expected!r}")


def _assert_close(errors: list[str], name: str, got: float, expected: float, tol: float = 0.005) -> None:
    if abs(got - expected) > tol:
        errors.append(f"{name}: got {got}, expected {expected}")


def collect_rq1() -> tuple[dict[str, Counter], dict[str, dict[str, int]]]:
    forms: dict[str, Counter] = {"MR": Counter(), "VA": Counter()}
    per_task: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"exact": 0, "range": 0, "bare": 0, "invalid": 0, "n_manifest": 0, "n_exact_ann": 0}
    )
    for row in _iter_jsonl(CANONICAL / "direct_dependencies.jsonl"):
        if row.get("track") != TRACK or not row.get("is_direct"):
            continue
        mode = row.get("mode")
        kind = str(row.get("constraint_kind") or "")
        source = row.get("declaration_source")
        key = (str(mode), str(row.get("model")), str(row.get("task_id")))
        if mode == MODE_MR and source == "manifest":
            forms["MR"][kind] += 1
            per_task[key]["n_manifest"] += 1
            if kind in per_task[key]:
                per_task[key][kind] += 1
        elif mode == MODE_VA and source in {"inline_annotation", "code_import_only"}:
            forms["VA"][kind] += 1
            if kind == "exact":
                per_task[key]["n_exact_ann"] += 1
    models: dict[str, dict[str, int]] = defaultdict(
        lambda: {"MR_exact": 0, "MR_range": 0, "MR_bare_only": 0, "VA_exact": 0, "n_mr": 0, "n_va": 0}
    )
    for row in _iter_jsonl(CANONICAL / "task_outcomes.jsonl"):
        if row.get("track") != TRACK or str(row.get("python_version")) != PYTHON_VERSION:
            continue
        model = _display(str(row["model"]))
        mode = str(row["mode"])
        key = (mode, str(row["model"]), str(row["task_id"]))
        flags = per_task.get(key, {"exact": 0, "range": 0, "bare": 0, "invalid": 0, "n_manifest": 0, "n_exact_ann": 0})
        if mode == MODE_MR:
            models[model]["n_mr"] += 1
            if flags["exact"]:
                models[model]["MR_exact"] += 1
            if flags["range"]:
                models[model]["MR_range"] += 1
            if flags["n_manifest"] > 0 and flags["exact"] == 0 and flags["range"] == 0 and flags["invalid"] == 0 and flags["bare"] > 0:
                models[model]["MR_bare_only"] += 1
        elif mode == MODE_VA:
            models[model]["n_va"] += 1
            if flags["n_exact_ann"]:
                models[model]["VA_exact"] += 1
    return forms, dict(models)


def render_rq1_forms(forms: dict[str, Counter]) -> str:
    mr = forms["MR"]
    va = forms["VA"]
    mr_n = sum(mr.values())
    va_n = sum(va.values())
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\footnotesize",
            r"\centering",
            r"\caption{Aggregate dependency-constraint forms on PinTrace under MR and VA.}",
            r"\label{tab:rq1_constraint_forms}",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Condition & Analysis unit & $N$ & Exact, $n$ (\%) & Range, $n$ (\%) & No version, $n$ (\%) & Invalid, $n$ (\%) \\",
            r"\midrule",
            (
                f"MR & Manifest entries & {mr_n:,} & "
                f"{mr['exact']:,} ({_pct(mr['exact'], mr_n):.2f}) & "
                f"{mr['range']:,} ({_pct(mr['range'], mr_n):.2f}) & "
                f"{mr['bare']:,} ({_pct(mr['bare'], mr_n):.2f}) & "
                f"{mr['invalid']:,} ({_pct(mr['invalid'], mr_n):.2f}) \\\\"
            ),
            (
                f"VA & Code imports & {va_n:,} & "
                f"{va['exact']:,} ({_pct(va['exact'], va_n):.2f}) & "
                f"-- & {va['bare']:,} ({_pct(va['bare'], va_n):.2f}) & -- \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_rq1_models(models: dict[str, dict[str, int]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\small",
        r"\centering",
        r"\caption{Per-model task-level constraint behavior on PinTrace.}",
        r"\label{tab:rq1_model_variation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & MR Exact (\%) & MR Range (\%) & MR Bare-only (\%) & VA Exact (\%) \\",
        r"\midrule",
    ]
    totals = Counter()
    for model in DISPLAY_ORDER:
        row = models[model]
        totals.update(row)
        lines.append(
            f"{model} & {_pct(row['MR_exact'], row['n_mr']):.2f} & "
            f"{_pct(row['MR_range'], row['n_mr']):.2f} & "
            f"{_pct(row['MR_bare_only'], row['n_mr']):.2f} & "
            f"{_pct(row['VA_exact'], row['n_va']):.2f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"\textbf{All observations} & "
                rf"\textbf{{{_pct(totals['MR_exact'], totals['n_mr']):.2f}}} & "
                rf"\textbf{{{_pct(totals['MR_range'], totals['n_mr']):.2f}}} & "
                rf"\textbf{{{_pct(totals['MR_bare_only'], totals['n_mr']):.2f}}} & "
                rf"\textbf{{{_pct(totals['VA_exact'], totals['n_va']):.2f}}} \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def collect_rq3() -> dict[str, dict[str, int]]:
    out = {
        "MR": {"n": 0, "inp": 0, "lock": 0, "inst": 0, "stat": 0},
        "VA": {"n": 0, "inp": 0, "lock": 0, "inst": 0, "stat": 0},
    }
    for row in _iter_jsonl(CANONICAL / "task_outcomes.jsonl"):
        if row.get("track") != TRACK or str(row.get("python_version")) != PYTHON_VERSION:
            continue
        cond = "MR" if row.get("mode") == MODE_MR else "VA"
        status = row.get("resolver_status")
        layer = row.get("compatibility_layer")
        ty = row.get("ty_compatible") is True
        lock = status == "resolved"
        inst = lock and layer not in INSTALL_FAIL
        out[cond]["n"] += 1
        out[cond]["inp"] += int(status in RESOLVER_INPUT)
        out[cond]["lock"] += int(lock)
        out[cond]["inst"] += int(inst)
        out[cond]["stat"] += int(lock and inst and ty)
    return out


def render_rq3(counts: dict[str, dict[str, int]]) -> str:
    mr, va = counts["MR"], counts["VA"]
    rows = [
        ("Resolver input available", "inp", "n"),
        ("Complete lock given input", "lock", "inp"),
        ("Installation completed given lock", "inst", "lock"),
        ("Static validation passed given installation", "stat", "inst"),
        ("End-to-end static pass", "stat", "n"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\small",
        r"\centering",
        r"\caption{Stage-wise retention of the 10,000 PinTrace task--model observations.}",
        r"\label{tab:rq3_environment_outcomes}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Outcome & MR, $n/d$ (\%) & VA, $n/d$ (\%) \\",
        r"\midrule",
    ]
    for label, num_key, den_key in rows:
        lines.append(
            f"{label} & {_cell(mr[num_key], mr[den_key])} & {_cell(va[num_key], va[den_key])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def collect_cve() -> dict[str, dict[str, tuple[int, int, int]]]:
    payload = json.loads(WITHDRAWN_IMPACT.read_text(encoding="utf-8"))
    key_map = {
        ("requirements.txt", "exact"): "MR Exact",
        ("requirements.txt", "range"): "MR Range",
        ("inline", "exact"): "VA Exact",
    }
    per: dict[str, dict[str, tuple[int, int, int]]] = defaultdict(dict)
    for row in payload["per_model"]:
        col = key_map.get((row["mode"], row["kind"]))
        if col is None:
            continue
        per[_display(row["model"])][col] = (int(row["V_after"]), int(row["P"]), int(row["N"]))
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for model_row in per.values():
        for col, (v, p, n) in model_row.items():
            totals[col][0] += v
            totals[col][1] += p
            totals[col][2] += n
    per["All observations"] = {col: (vals[0], vals[1], vals[2]) for col, vals in totals.items()}
    return dict(per)


def _cve_cell(v: int, p: int, n: int) -> str:
    return f"{_pct(v, p):.2f} [{_pct(v, n):.2f}]"


def render_cve(per: dict[str, dict[str, tuple[int, int, int]]]) -> str:
    cols = ("MR Exact", "MR Range", "VA Exact")
    lines = [
        r"\begin{table}[t]",
        r"\small",
        r"\centering",
        r"\caption{Task-level known-vulnerability association after excluding withdrawn advisories.}",
        r"\label{tab:rq2_cve_association}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & MR Exact, $V/P$ [$V/N$] (\%) & MR Range, $V/P$ [$V/N$] (\%) & VA Exact, $V/P$ [$V/N$] (\%) \\",
        r"\midrule",
    ]
    for model in DISPLAY_ORDER:
        row = per[model]
        lines.append(
            f"{model} & " + " & ".join(_cve_cell(*row[col]) for col in cols) + r" \\"
        )
    agg = per["All observations"]
    lines.extend(
        [
            r"\midrule",
            r"\textbf{All observations} & "
            + " & ".join(rf"\textbf{{{_cve_cell(*agg[col])}}}" for col in cols)
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def collect_neighbor() -> dict[str, tuple[int, int, int]]:
    payload = json.loads(
        NEIGHBOR_JSON.read_text(
            encoding="utf-8"
        )
    )
    per: dict[str, tuple[int, int, int]] = {}
    for run in payload["runs"]:
        selected = run["selected_run"]
        summary = run["summary"]
        model = _display(str(selected.get("model_name") or selected.get("run_name")))
        per[model] = (
            int(summary["recover_no_ty_install_error_tasks"]),
            int(summary["recover_ty_tasks"]),
            int(summary["baseline_tasks"]),
        )
    return per


def render_neighbor(per: dict[str, tuple[int, int, int]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Neighboring-version recovery among eligible PinTrace VA baseline failures.}",
        r"\label{tab:neighbor_recovery}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & Installation recovered, $n/N$ (\%) & Static recovered, $n/N$ (\%) \\",
        r"\midrule",
    ]
    inst = stat = n = 0
    for model in DISPLAY_ORDER:
        rec_i, rec_s, den = per[model]
        inst += rec_i
        stat += rec_s
        n += den
        lines.append(f"{model} & {_cell(rec_i, den)} & {_cell(rec_s, den)} \\\\")
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{All observations}} & \textbf{{{_cell(inst, n)}}} & \textbf{{{_cell(stat, n)}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def collect_robustness() -> dict[tuple[str, str, str], dict[str, float]]:
    path = ROBUST_CSV
    by_row: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["request_condition"], row["python_version"])
            by_row[key] = {
                "complete_lock": 100.0 * float(row["e2e_complete_lock_rate"]),
                "installation_completed": 100.0 * float(row["e2e_installation_completed_rate"]),
                "static_pass": 100.0 * float(row["e2e_static_pass_rate"]),
                "dynamic_pass": 100.0 * float(row["e2e_dynamic_pass_rate"] or 0.0)
                if row.get("e2e_dynamic_pass_rate")
                else float("nan"),
            }
    return by_row


def _robust_value(table: dict[tuple[str, str, str], dict[str, float]], dataset: str, cond: str, metric: str, py: str) -> str:
    value = table[(dataset, cond, py)][metric]
    return f"{value:.2f}"


def render_robustness(table: dict[tuple[str, str, str], dict[str, float]]) -> str:
    pys = ("3.8", "3.10", "3.12", "3.14")
    pintrace_rows = (
        ("Complete lock", "complete_lock"),
        ("Installation completed", "installation_completed"),
        ("Static validation passed", "static_pass"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-Python robustness of model-produced dependency information.}",
        r"\label{tab:python_robustness}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Outcome & Condition & Python 3.8 (\%) & Python 3.10 (\%) & Python 3.12 (\%) & Python 3.14 (\%) \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{PinTrace}} \\",
    ]
    for label, metric in pintrace_rows:
        for cond in ("MR", "VA"):
            vals = " & ".join(_robust_value(table, "PinTrace", cond, metric, py) for py in pys)
            lines.append(f"{label} & {cond} & {vals} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Execution tests}} \\")
    for cond in ("MR", "VA"):
        vals = " & ".join(_robust_value(table, "BigCodeBench", cond, "dynamic_pass", py) for py in pys)
        lines.append(f"Dynamic test passed & {cond} & {vals} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def render_dataset_temporal() -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Temporal distribution of the 1,000 PinTrace tasks.}",
        r"\label{tab:dataset_temporal}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Year & \# Tasks & Cumulative (\%) & \# Unique Packages \\",
        r"\midrule",
    ]
    for year, n, cum, pkgs in FROZEN_TEMPORAL:
        lines.append(f"{year} & {n} & {cum:.1f}  & {pkgs} \\\\")
    lines.extend(
        [
            r"\midrule",
            r"Total & 1,000 & 100.0 & 267 \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_dataset_content(content: dict[str, Any], tpl: dict[str, Any]) -> str:
    q = content["question_raw_tokens"]
    loc = content["accepted_answer_selected_code"]["lines"]
    tok = content["accepted_answer_selected_code"]["tokens"]
    per = tpl["tpl_packages_per_task"]
    ge2 = tpl["tasks_with_ge_2_target_packages"]
    ge3 = tpl["tasks_with_ge_3_target_packages"]
    n = EXPECTED_DATASET["n_tasks"]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Content statistics of the PinTrace dataset.}",
            r"\label{tab:dataset_content}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Statistic & Mean & Median & Std. \\",
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Question}} \\",
            rf"\ Token length             & {q['mean']:.1f} & {q['median']:.1f} & {q['stdev']:.1f} \\",
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Accepted answer code block}} \\",
            rf"\ Lines of code            & {loc['mean']:.1f} & {loc['median']:.1f} & {loc['stdev']:.1f} \\",
            rf"\ Token length             & {tok['mean']:.1f} & {tok['median']:.1f} & {tok['stdev']:.1f} \\",
            rf"\ TPL imports per task     & {per['mean']:.1f} & {per['median']:.1f} & {per['stdev']:.1f} \\",
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Library coverage}} \\",
            rf"\ Distinct TPLs            & \multicolumn{{3}}{{r}}{{{tpl['distinct_tpl_lower']}}} \\",
            rf"\ Tasks with $\geq 2$ TPL imports & \multicolumn{{3}}{{r}}{{{ge2} ({100.0 * ge2 / n:.1f}\%)}} \\",
            rf"\ Tasks with $\geq 3$ TPL imports & \multicolumn{{3}}{{r}}{{{ge3} ({100.0 * ge3 / n:.1f}\%)}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_rq2_validity_paper(stats: dict, models: list[str]) -> str:
    from plots.rq2_constraint_validity_table import PSCount, aggregate_condition, COND_MR, COND_VA

    lines = [
        r"\begin{table}[t]",
        r"\small",
        r"\centering",
        r"\caption{Per-model task-level validity of model-produced constraints on the 849 clean PinTrace tasks.}",
        r"\label{tab:rq2_constraint_validity}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & MR Exact, $P/S$ (\%) & MR Range, $P/S$ (\%) & VA Exact, $P/S$ (\%) & MR All, $A/S_{\mathrm{any}}$ (\%) & VA All, $A/S_{\mathrm{any}}$ (\%) \\",
        r"\midrule",
    ]
    for model in models:
        mr = stats[(COND_MR, model)]
        va = stats[(COND_VA, model)]
        lines.append(
            f"{model} & {PSCount(mr.p_exact, mr.s_exact).cell()} & "
            f"{PSCount(mr.p_range, mr.s_range).cell()} & "
            f"{PSCount(va.p_exact, va.s_exact).cell()} & "
            f"{PSCount(mr.a_all, mr.s_any).cell()} & "
            f"{PSCount(va.a_all, va.s_any).cell()} \\\\"
        )
    mr_all = aggregate_condition(stats, models, COND_MR)
    va_all = aggregate_condition(stats, models, COND_VA)
    lines.extend(
        [
            r"\midrule",
            (
                r"\textbf{All observations} & "
                rf"\textbf{{{PSCount(mr_all.p_exact, mr_all.s_exact).cell()}}} & "
                rf"\textbf{{{PSCount(mr_all.p_range, mr_all.s_range).cell()}}} & "
                rf"\textbf{{{PSCount(va_all.p_exact, va_all.s_exact).cell()}}} & "
                rf"\textbf{{{PSCount(mr_all.a_all, mr_all.s_any).cell()}}} & "
                rf"\textbf{{{PSCount(va_all.a_all, va_all.s_any).cell()}}} \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    forms, rq1_models = collect_rq1()
    _write(out / "rq1_constraint_forms.tex", render_rq1_forms(forms))
    _write(out / "rq1_model_variation.tex", render_rq1_models(rq1_models))
    _write_csv(
        out / "rq1_constraint_forms.csv",
        [
            {
                "condition": cond,
                "kind": kind,
                "count": count,
                "N": sum(forms[cond].values()),
            }
            for cond, counter in forms.items()
            for kind, count in sorted(counter.items())
        ],
    )
    _write_csv(
        out / "rq1_model_variation.csv",
        [
            {
                "model": model,
                "mr_exact_pct": _pct(row["MR_exact"], row["n_mr"]),
                "mr_range_pct": _pct(row["MR_range"], row["n_mr"]),
                "mr_bare_only_pct": _pct(row["MR_bare_only"], row["n_mr"]),
                "va_exact_pct": _pct(row["VA_exact"], row["n_va"]),
                **row,
            }
            for model, row in ((m, rq1_models[m]) for m in DISPLAY_ORDER)
        ],
    )
    mr_n = sum(forms["MR"].values())
    va_n = sum(forms["VA"].values())
    _assert_eq(errors, "RQ1 MR N", mr_n, EXPECTED_RQ1_FORMS["MR"]["N"])
    _assert_eq(errors, "RQ1 VA N", va_n, EXPECTED_RQ1_FORMS["VA"]["N"])
    for kind, expected in EXPECTED_RQ1_FORMS["MR"].items():
        if kind != "N":
            _assert_eq(errors, f"RQ1 MR {kind}", forms["MR"][kind], expected)
    _assert_eq(errors, "RQ1 VA exact", forms["VA"]["exact"], EXPECTED_RQ1_FORMS["VA"]["exact"])
    _assert_eq(errors, "RQ1 VA bare", forms["VA"]["bare"], EXPECTED_RQ1_FORMS["VA"]["bare"])
    totals = Counter()
    for model in DISPLAY_ORDER:
        totals.update(rq1_models[model])
        expected = EXPECTED_RQ1_TASK[model]
        got = {
            "MR Exact": _pct(rq1_models[model]["MR_exact"], rq1_models[model]["n_mr"]),
            "MR Range": _pct(rq1_models[model]["MR_range"], rq1_models[model]["n_mr"]),
            "MR Bare-only": _pct(rq1_models[model]["MR_bare_only"], rq1_models[model]["n_mr"]),
            "VA Exact": _pct(rq1_models[model]["VA_exact"], rq1_models[model]["n_va"]),
        }
        for key, value in expected.items():
            _assert_close(errors, f"RQ1 {model} {key}", got[key], value)
    all_expected = EXPECTED_RQ1_TASK["All observations"]
    _assert_close(errors, "RQ1 all MR exact", _pct(totals["MR_exact"], totals["n_mr"]), all_expected["MR Exact"])
    _assert_close(errors, "RQ1 all MR range", _pct(totals["MR_range"], totals["n_mr"]), all_expected["MR Range"])
    _assert_close(errors, "RQ1 all MR bare", _pct(totals["MR_bare_only"], totals["n_mr"]), all_expected["MR Bare-only"])
    _assert_close(errors, "RQ1 all VA exact", _pct(totals["VA_exact"], totals["n_va"]), all_expected["VA Exact"])

    clean_ids = load_clean_ids(CLEAN_IDS)
    observations = load_observations(CANONICAL / "task_outcomes.jsonl", clean_ids)
    attach_dependencies(observations, CANONICAL / "direct_dependencies.jsonl", clean_ids)
    validity_stats = accumulate(observations)
    validity_models = DISPLAY_ORDER
    _write(out / "rq2_constraint_validity.tex", render_rq2_validity_paper(validity_stats, validity_models))
    _write(out / "rq2_constraint_validity_legacy.tex", render_rq2_validity_legacy_tex(validity_stats, validity_models))
    write_rq2_validity_csv(out / "rq2_constraint_validity.csv", validity_stats, validity_models)
    from plots.rq2_constraint_validity_table import aggregate_condition, COND_MR, COND_VA

    mr_all = aggregate_condition(validity_stats, validity_models, COND_MR)
    va_all = aggregate_condition(validity_stats, validity_models, COND_VA)
    _assert_eq(errors, "RQ2 MR exact P/S", (mr_all.p_exact, mr_all.s_exact), EXPECTED_RQ2_VALIDITY[("All observations", "MR Exact")])
    _assert_eq(errors, "RQ2 MR range P/S", (mr_all.p_range, mr_all.s_range), EXPECTED_RQ2_VALIDITY[("All observations", "MR Range")])
    _assert_eq(errors, "RQ2 VA exact P/S", (va_all.p_exact, va_all.s_exact), EXPECTED_RQ2_VALIDITY[("All observations", "VA Exact")])
    _assert_eq(errors, "RQ2 MR all A/S", (mr_all.a_all, mr_all.s_any), EXPECTED_RQ2_VALIDITY[("All observations", "MR All")])
    _assert_eq(errors, "RQ2 VA all A/S", (va_all.a_all, va_all.s_any), EXPECTED_RQ2_VALIDITY[("All observations", "VA All")])

    cve = collect_cve()
    _write(out / "rq2_cve_association.tex", render_cve(cve))
    _write_csv(
        out / "rq2_cve_association.csv",
        [
            {
                "model": model,
                "column": col,
                "V": v,
                "P": p,
                "N": n,
                "V_over_P": _pct(v, p),
                "V_over_N": _pct(v, n),
            }
            for model in [*DISPLAY_ORDER, "All observations"]
            for col, (v, p, n) in cve[model].items()
        ],
    )
    for col, (vp, vn) in EXPECTED_RQ2_CVE["All observations"].items():
        v, p, n = cve["All observations"][col]
        _assert_close(errors, f"RQ2 CVE {col} V/P", _pct(v, p), vp)
        _assert_close(errors, f"RQ2 CVE {col} V/N", _pct(v, n), vn)

    rq3 = collect_rq3()
    _write(out / "rq3_environment_outcomes.tex", render_rq3(rq3))
    _write_csv(
        out / "rq3_environment_outcomes.csv",
        [{"condition": cond, **counts} for cond, counts in rq3.items()],
    )
    for cond, expected in EXPECTED_RQ3.items():
        for key, value in expected.items():
            _assert_eq(errors, f"RQ3 {cond} {key}", rq3[cond][key], value)

    neighbor = collect_neighbor()
    _write(out / "neighbor_recovery.tex", render_neighbor(neighbor))
    _write_csv(
        out / "neighbor_recovery.csv",
        [
            {
                "model": model,
                "install_recovered": neighbor[model][0],
                "static_recovered": neighbor[model][1],
                "eligible": neighbor[model][2],
            }
            for model in DISPLAY_ORDER
        ],
    )
    inst = stat = den = 0
    for model, expected in EXPECTED_NEIGHBOR.items():
        _assert_eq(errors, f"neighbor {model}", neighbor[model], expected)
        inst += neighbor[model][0]
        stat += neighbor[model][1]
        den += neighbor[model][2]
    _assert_eq(errors, "neighbor all", (inst, stat, den), EXPECTED_NEIGHBOR_ALL)

    robust = collect_robustness()
    _write(out / "python_robustness.tex", render_robustness(robust))
    pys = ("3.8", "3.10", "3.12", "3.14")
    for (dataset, cond, metric), expected in EXPECTED_ROBUSTNESS.items():
        got = tuple(round(robust[(dataset, cond, py)][metric], 2) for py in pys)
        _assert_eq(errors, f"robustness {dataset} {cond} {metric}", got, expected)

    records = load_records_jsonl(PINTRACE)
    _assert_eq(errors, "PinTrace n", len(records), EXPECTED_DATASET["n_tasks"])
    tpl = compute_target_package_stats(records)
    _assert_eq(errors, "distinct TPL", tpl["distinct_tpl_lower"], EXPECTED_DATASET["distinct_tpl"])
    _assert_eq(errors, "TPL >=2", tpl["tasks_with_ge_2_target_packages"], EXPECTED_DATASET["ge2"])
    _assert_eq(errors, "TPL >=3", tpl["tasks_with_ge_3_target_packages"], EXPECTED_DATASET["ge3"])
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        content = compute_content_stats(records, enc, python_version="3.12")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"tiktoken is required to regenerate tab:dataset_content: {exc}") from exc
    q = content["question_raw_tokens"]
    loc = content["accepted_answer_selected_code"]["lines"]
    tok = content["accepted_answer_selected_code"]["tokens"]
    _assert_close(errors, "question mean", round(q["mean"], 1), EXPECTED_DATASET["question_tokens"][0], 0.05)
    _assert_close(errors, "question median", round(q["median"], 1), EXPECTED_DATASET["question_tokens"][1], 0.05)
    _assert_close(errors, "loc mean", round(loc["mean"], 1), EXPECTED_DATASET["code_lines"][0], 0.05)
    _assert_close(errors, "code tokens mean", round(tok["mean"], 1), EXPECTED_DATASET["code_tokens"][0], 0.05)
    _write(out / "dataset_content.tex", render_dataset_content(content, tpl))
    per = tpl["tpl_packages_per_task"]
    _assert_close(errors, "tpl mean", round(per["mean"], 1), EXPECTED_DATASET["tpl_per_task"][0], 0.05)
    _write(out / "dataset_temporal.tex", render_dataset_temporal())
    _write_csv(
        out / "dataset_temporal.csv",
        [
            {"year": year, "n_tasks": n, "cumulative_pct": cum, "unique_packages": pkgs}
            for year, n, cum, pkgs in FROZEN_TEMPORAL
        ],
    )

    try:
        shown = out.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out
    report = ["# Paper table regeneration", "", f"Output directory: `{shown}`", ""]
    if errors:
        report.append("## Failed assertions")
        report.extend(f"- {item}" for item in errors)
        _write(out / "assertions.md", "\n".join(report) + "\n")
        sys.stderr.write("\n".join(errors) + "\n")
        raise SystemExit(1)
    report.append("All manuscript headline counts matched.")
    _write(out / "assertions.md", "\n".join(report) + "\n")
    print(f"Wrote paper tables to {shown}")
    print("All manuscript headline counts matched.")


if __name__ == "__main__":
    main()
