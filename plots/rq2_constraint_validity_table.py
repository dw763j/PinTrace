"""RQ2 constraint-validity table with type-level P/S and response-level A/S_any.

Reproduce the frozen per-type validity rates from the corrected canonical
tables, then add all-constraint validity for the same 849-task clean PinTrace
population.

Observation key: (task_id, model, request condition).
Model-written exact/range records follow the canonical funnel filter in
``evaluate.build_canonical_tables``:
declaration_source in {manifest, inline_annotation} and
constraint_kind in {exact, range}.
Input records are not filtered by resolution_origin.

The response-level A/S_any denominator additionally includes model-origin
records with constraint_kind=invalid.  Such records cannot identify a
satisfiable release and therefore make the containing observation fail the
all-constraint criterion.  They remain outside the type-specific exact/range
P/S numerators and denominators.

Usage (from project root)::

    python -m plots.rq2_constraint_validity_table
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from plots.model_display import MODEL_LABEL, paper_display_label
from paths import ANALYSIS_FUNNELS, EVALUATION, display_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = EVALUATION
TABLES_DIR = ANALYSIS_FUNNELS
PLOTS_DIR = Path(__file__).resolve().parent

DEFAULT_DEPS = CANONICAL_DIR / "direct_dependencies.jsonl"
DEFAULT_TASKS = CANONICAL_DIR / "task_outcomes.jsonl"
DEFAULT_CLEAN_IDS = PROJECT_ROOT / "resources" / "revision" / "pintrace_clean_task_ids.jsonl"
DEFAULT_EXACT = TABLES_DIR / "constraint_funnel_exact.csv"
DEFAULT_RANGE = TABLES_DIR / "constraint_funnel_range.csv"
DEFAULT_TEX = PLOTS_DIR / "rq2_constraint_validity_table.tex"
DEFAULT_CSV = PLOTS_DIR / "rq2_constraint_validity_table.csv"
DEFAULT_REPORT = PLOTS_DIR / "rq2_constraint_validity_table_report.md"

TRACK = "d2"
POPULATION = "clean"
MODE_MR = "requirements.txt"
MODE_VA = "inline"
COND_MR = "MR"
COND_VA = "VA"
MODE_TO_COND = {MODE_MR: COND_MR, MODE_VA: COND_VA}
MODEL_SOURCES = frozenset({"manifest", "inline_annotation"})
PARSED_KINDS = frozenset({"exact", "range"})
ALL_CONSTRAINT_KINDS = frozenset({"exact", "range", "invalid"})
EXPECTED_N = 849
EXPECTED_N_MODELS = 10
EXPECTED_N_OBS = EXPECTED_N * EXPECTED_N_MODELS
EXPECTED_CLEAN_AGG = {
    ("MR", "exact"): (1207, 1271),
    ("MR", "range"): (2187, 2202),
    ("VA", "exact"): (6389, 6997),
}
CSV_FIELDS = (
    "model",
    "condition",
    "S_exact",
    "P_exact",
    "S_range",
    "P_range",
    "S_any",
    "A_all",
    "mixed_validity",
    "none_valid",
)


@dataclass(frozen=True)
class PSCount:
    p: int
    s: int

    @property
    def pct(self) -> float | None:
        return (100.0 * self.p / self.s) if self.s else None

    def cell(self) -> str:
        if self.s == 0:
            return "--"
        return f"{_count(self.p)}/{_count(self.s)} ({self.pct:.2f}\\%)"


@dataclass
class ConditionStats:
    n: int = 0
    s_exact: int = 0
    p_exact: int = 0
    s_range: int = 0
    p_range: int = 0
    s_any: int = 0
    a_all: int = 0
    mixed_validity: int = 0
    none_valid: int = 0
    all_valid: int = 0
    both_kinds: int = 0

    def add(self, other: ConditionStats) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class Observation:
    task_id: str
    model: str
    mode: str
    deps: list[dict] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--task-outcomes", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--clean-task-ids", type=Path, default=DEFAULT_CLEAN_IDS)
    parser.add_argument("--exact-csv", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--range-csv", type=Path, default=DEFAULT_RANGE)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _count(value: int) -> str:
    return f"{value:,}"


def _int(value: object) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_funnel(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _display(model: str) -> str:
    return paper_display_label(model, model)


def _model_order(names: set[str]) -> list[str]:
    ordered = [label for label in MODEL_LABEL.values() if label in names]
    leftovers = sorted(names - set(ordered))
    return ordered + leftovers


def is_parsed_model_origin_constraint(dep: dict) -> bool:
    return dep.get("declaration_source") in MODEL_SOURCES and dep.get("constraint_kind") in PARSED_KINDS


def is_any_model_origin_constraint(dep: dict) -> bool:
    return (
        dep.get("declaration_source") in MODEL_SOURCES
        and dep.get("constraint_kind") in ALL_CONSTRAINT_KINDS
    )


def constraint_is_valid(dep: dict) -> bool:
    if dep.get("constraint_kind") == "invalid":
        return False
    value = dep.get("constraint_valid")
    if value is not None:
        return bool(value)
    return bool(dep.get("version_exists"))


def load_clean_ids(path: Path) -> set[str]:
    return {str(row["task_id"]) for row in _read_jsonl(path)}


def load_observations(task_path: Path, clean_ids: set[str]) -> dict[tuple[str, str, str], Observation]:
    observations: dict[tuple[str, str, str], Observation] = {}
    for row in _read_jsonl(task_path):
        if row.get("track") != TRACK or str(row["task_id"]) not in clean_ids:
            continue
        key = (str(row["mode"]), str(row["model"]), str(row["task_id"]))
        if key in observations:
            raise ValueError(f"duplicate task outcome for {key}")
        observations[key] = Observation(
            task_id=str(row["task_id"]),
            model=str(row["model"]),
            mode=str(row["mode"]),
        )
    return observations


def attach_dependencies(
    observations: dict[tuple[str, str, str], Observation],
    dep_path: Path,
    clean_ids: set[str],
) -> dict[str, list[dict]]:
    excluded: dict[str, list[dict]] = {
        "invalid_kind": [],
        "missing_validity_in_scope": [],
        "unmatched_dependency": [],
    }
    for dep in _read_jsonl(dep_path):
        if dep.get("track") != TRACK or str(dep["task_id"]) not in clean_ids:
            continue
        key = (str(dep["mode"]), str(dep["model"]), str(dep["task_id"]))
        if key not in observations:
            excluded["unmatched_dependency"].append(dep)
            continue
        if (
            dep.get("declaration_source") in MODEL_SOURCES
            and dep.get("constraint_kind") == "invalid"
        ):
            excluded["invalid_kind"].append(dep)
        if is_any_model_origin_constraint(dep):
            if is_parsed_model_origin_constraint(dep) and dep.get("constraint_valid") is None:
                excluded["missing_validity_in_scope"].append(dep)
            observations[key].deps.append(dep)
    return excluded


def observation_flags(obs: Observation) -> dict[str, bool | int]:
    exact = [dep for dep in obs.deps if dep.get("constraint_kind") == "exact"]
    ranges = [dep for dep in obs.deps if dep.get("constraint_kind") == "range"]
    exact_present = bool(exact)
    range_present = bool(ranges)
    exact_any_valid = any(constraint_is_valid(dep) for dep in exact)
    range_any_valid = any(constraint_is_valid(dep) for dep in ranges)
    any_present = bool(obs.deps)
    n_valid = sum(constraint_is_valid(dep) for dep in obs.deps)
    n_invalid = len(obs.deps) - n_valid
    all_valid = any_present and n_invalid == 0
    mixed = any_present and n_valid > 0 and n_invalid > 0
    none_valid = any_present and n_valid == 0
    return {
        "exact_present": exact_present,
        "exact_any_valid": exact_any_valid,
        "exact_all_valid": exact_present and all(constraint_is_valid(dep) for dep in exact),
        "range_present": range_present,
        "range_any_valid": range_any_valid,
        "range_all_valid": range_present and all(constraint_is_valid(dep) for dep in ranges),
        "any_constraint_present": any_present,
        "all_constraints_valid": all_valid,
        "all_valid": all_valid,
        "mixed_validity": mixed,
        "none_valid": none_valid,
        "both_kinds": exact_present and range_present,
    }


def accumulate(
    observations: dict[tuple[str, str, str], Observation],
) -> dict[tuple[str, str], ConditionStats]:
    stats: dict[tuple[str, str], ConditionStats] = defaultdict(ConditionStats)
    for obs in observations.values():
        flags = observation_flags(obs)
        row = stats[(MODE_TO_COND[obs.mode], _display(obs.model))]
        row.n += 1
        row.s_exact += int(flags["exact_present"])
        row.p_exact += int(flags["exact_any_valid"])
        row.s_range += int(flags["range_present"])
        row.p_range += int(flags["range_any_valid"])
        row.s_any += int(flags["any_constraint_present"])
        row.a_all += int(flags["all_constraints_valid"])
        row.mixed_validity += int(flags["mixed_validity"])
        row.none_valid += int(flags["none_valid"])
        row.all_valid += int(flags["all_valid"])
        row.both_kinds += int(flags["both_kinds"])
    return stats


def expected_from_funnel(exact_rows: list[dict[str, str]], range_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], tuple[int, int]]:
    expected: dict[tuple[str, str, str], tuple[int, int]] = {}
    for rows, kind in ((exact_rows, "exact"), (range_rows, "range")):
        for row in rows:
            if row["track"] != TRACK or row["population"] != POPULATION:
                continue
            cond = MODE_TO_COND[row["mode"]]
            expected[(cond, _display(row["model"]), kind)] = (_int(row["P"]), _int(row["S"]))
    return expected


def csv_record(model: str, condition: str, stats: ConditionStats) -> dict[str, object]:
    return {
        "model": model,
        "condition": condition,
        "S_exact": stats.s_exact,
        "P_exact": stats.p_exact,
        "S_range": stats.s_range,
        "P_range": stats.p_range,
        "S_any": stats.s_any,
        "A_all": stats.a_all,
        "mixed_validity": stats.mixed_validity,
        "none_valid": stats.none_valid,
    }


def reproduce_or_stop(
    stats: dict[tuple[str, str], ConditionStats],
    expected: dict[tuple[str, str, str], tuple[int, int]],
    models: list[str],
) -> list[str]:
    diffs: list[str] = []
    for model in models:
        for cond, kind, attr_p, attr_s in (
            (COND_MR, "exact", "p_exact", "s_exact"),
            (COND_MR, "range", "p_range", "s_range"),
            (COND_VA, "exact", "p_exact", "s_exact"),
        ):
            got = stats[(cond, model)]
            got_ps = (getattr(got, attr_p), getattr(got, attr_s))
            exp = expected.get((cond, model, kind))
            if exp is None:
                diffs.append(f"{cond} {kind} {model}: missing from frozen funnel CSV")
            elif got_ps != exp:
                diffs.append(
                    f"{cond} {kind} {model}: got {got_ps[0]}/{got_ps[1]} != expected {exp[0]}/{exp[1]}"
                )
        va_range = stats[(COND_VA, model)]
        if (va_range.p_range, va_range.s_range) != (0, 0):
            diffs.append(
                f"VA range {model}: got {va_range.p_range}/{va_range.s_range}, expected 0/0"
            )
    for (cond, kind), (p, s) in EXPECTED_CLEAN_AGG.items():
        total = ConditionStats()
        for model in models:
            total.add(stats[(cond, model)])
        got_p = total.p_exact if kind == "exact" else total.p_range
        got_s = total.s_exact if kind == "exact" else total.s_range
        if (got_p, got_s) != (p, s):
            diffs.append(f"{cond} {kind} aggregate: got {got_p}/{got_s} != expected {p}/{s}")
    return diffs


def sanity_errors(stats: dict[tuple[str, str], ConditionStats], models: list[str]) -> list[str]:
    errors: list[str] = []
    if len(models) != EXPECTED_N_MODELS:
        errors.append(f"expected {EXPECTED_N_MODELS} models, got {len(models)}: {models}")
    expected_labels = list(MODEL_LABEL.values())
    if models != expected_labels:
        errors.append(f"model set/order mismatch: {models} != {expected_labels}")
    for cond in (COND_MR, COND_VA):
        total = ConditionStats()
        for model in models:
            row = stats[(cond, model)]
            total.add(row)
            if row.n != EXPECTED_N:
                errors.append(f"{cond} {model}: N={row.n}, expected {EXPECTED_N}")
            if row.a_all > row.s_any:
                errors.append(f"{cond} {model}: A={row.a_all} > S_any={row.s_any}")
            parts = row.all_valid + row.mixed_validity + row.none_valid
            if parts != row.s_any:
                errors.append(
                    f"{cond} {model}: all_valid+mixed+none={parts} != S_any={row.s_any}"
                )
            if row.all_valid != row.a_all:
                errors.append(f"{cond} {model}: all_valid={row.all_valid} != A={row.a_all}")
        if total.n != EXPECTED_N_OBS:
            errors.append(f"{cond}: {total.n} observations, expected {EXPECTED_N_OBS}")
        if total.a_all > total.s_any:
            errors.append(f"{cond}: A={total.a_all} > S_any={total.s_any}")
        parts = total.all_valid + total.mixed_validity + total.none_valid
        if parts != total.s_any:
            errors.append(f"{cond}: partition {parts} != S_any={total.s_any}")
        if cond == COND_VA and total.s_range != 0:
            errors.append(f"VA range S={total.s_range}, expected 0")
    return errors


def aggregate_condition(stats: dict[tuple[str, str], ConditionStats], models: list[str], cond: str) -> ConditionStats:
    total = ConditionStats()
    for model in models:
        total.add(stats[(cond, model)])
    return total


def render_tex(stats: dict[tuple[str, str], ConditionStats], models: list[str]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\small",
        r"\centering",
        r"\caption{\revised{",
        r"Per-model validity of model-origin constraints on the 849 clean \myso tasks.",
        r"The analysis unit is a task--model observation, and each model contributes 849 observations per request condition.",
        r"$P/S$ is the number of observations containing at least one independently satisfiable constraint of the named type, over those containing that type.",
        r"$A/S_{\mathrm{any}}$ is the number of observations in which every model-origin exact or range constraint is independently satisfiable, over those containing at least one exact, range, or invalid constraint.",
        r"Unparseable invalid constraints enter $S_{\mathrm{any}}$ as failures but remain outside the type-specific $P/S$ columns.",
        r"This all-constraint rate does not measure complete environment resolution.",
        r"The ``All observations'' row sums counts across models before computing the percentage.",
        r"}}",
        r"\label{tab:rq2_constraint_validity}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\revised{\textbf{Model}} &",
        r"\revised{\textbf{MR Exact, $P/S$ (\%)}} &",
        r"\revised{\textbf{MR Range, $P/S$ (\%)}} &",
        r"\revised{\textbf{VA Exact, $P/S$ (\%)}} &",
        r"\revised{\textbf{MR All, $A/S_{\mathrm{any}}$ (\%)}} &",
        r"\revised{\textbf{VA All, $A/S_{\mathrm{any}}$ (\%)}} \\",
        r"\midrule",
    ]
    for model in models:
        mr = stats[(COND_MR, model)]
        va = stats[(COND_VA, model)]
        lines.append(
            f"{model} & "
            f"{PSCount(mr.p_exact, mr.s_exact).cell()} & "
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
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, stats: dict[tuple[str, str], ConditionStats], models: list[str]) -> None:
    records = [csv_record(model, cond, stats[(cond, model)]) for model in models for cond in (COND_MR, COND_VA)]
    for cond in (COND_MR, COND_VA):
        records.append(csv_record("All observations", cond, aggregate_condition(stats, models, cond)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(records)


def _invalid_obs(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for dep in records:
        grouped[(dep["mode"], dep["model"], str(dep["task_id"]))].append(dep)
    rows = []
    for (mode, model, task_id), deps in grouped.items():
        rows.append(
            {
                "condition": MODE_TO_COND[mode],
                "model": _display(model),
                "task_id": task_id,
                "n_records": len(deps),
                "packages": [str(dep.get("pypi_name") or dep.get("raw_requirement")) for dep in deps],
                "raw_requirements": [dep.get("raw_requirement") for dep in deps],
            }
        )
    return rows


def render_report(
    *,
    args: argparse.Namespace,
    models: list[str],
    stats: dict[tuple[str, str], ConditionStats],
    excluded: dict[str, list[dict]],
    n_obs: int,
    reproduced: bool,
    diffs: list[str],
    sanity: list[str],
) -> str:
    mr = aggregate_condition(stats, models, COND_MR)
    va = aggregate_condition(stats, models, COND_VA)
    invalid_obs = _invalid_obs(excluded["invalid_kind"])
    lines = [
        "# RQ2 constraint-validity verification report",
        "",
        "## Data files",
        "",
        f"- task outcomes: `{display_path(args.task_outcomes)}`",
        f"- direct dependencies: `{display_path(args.dependencies)}`",
        f"- clean task ids: `{display_path(args.clean_task_ids)}`",
        f"- frozen exact funnel: `{display_path(args.exact_csv)}`",
        f"- frozen range funnel: `{display_path(args.range_csv)}`",
        f"- population: track=`{TRACK}`, clean PinTrace tasks=`{EXPECTED_N}`, models=`{EXPECTED_N_MODELS}`",
        "",
        "## Observation key",
        "",
        "- `(task_id, model, request condition)`",
        f"- MR = `{MODE_MR}`, VA = `{MODE_VA}`",
        f"- observations loaded from `task_outcomes.jsonl`: {n_obs}",
        f"- expected: {EXPECTED_N_OBS} per condition, {EXPECTED_N_OBS * 2} total",
        "",
        "## Exact/range inclusion filter",
        "",
        "Reuse the canonical funnel filter from `evaluate.build_canonical_tables`:",
        "",
        "- `declaration_source` in `{manifest, inline_annotation}`",
        "- `constraint_kind` in `{exact, range}`",
        "- do **not** filter by `resolution_origin`",
        "- exclude bare/unconstrained declarations, code-import-only records, test-harness dependencies, and transitive dependencies",
        "- type-specific `P/S` includes only `constraint_kind` in `{exact, range}`",
        "- response-level `A/S_any` additionally includes model-origin `constraint_kind=invalid` records as failures",
        "",
        "## Constraint validity field",
        "",
        "- primary: `constraint_valid`",
        "- fallback if `constraint_valid` is missing: `version_exists`",
        "- this matches `_has_valid_constraint` in the canonical generator",
        f"- in-scope exact/range records with missing `constraint_valid`: {len(excluded['missing_validity_in_scope'])}",
        "",
        "## Invalid or unparsed records",
        "",
        f"- `constraint_kind=invalid` records in D2 clean: {len(excluded['invalid_kind'])}",
        f"- observations involved: {len(invalid_obs)}",
    ]
    if invalid_obs:
        for row in invalid_obs:
            lines.append(
                f"- {row['condition']} / {row['model']} / `{row['task_id']}`: "
                f"{row['n_records']} records {row['packages']}"
            )
        lines.extend(
            [
                "- canonical exact/range funnel inclusion: **excluded**, because `constraint_kind` is not `exact` or `range`",
                "- response-level `A/S_any` inclusion: **included as a failure**, because the model-origin declaration cannot identify a satisfiable release",
                "- the affected observation contains only invalid Maven-style coordinates and therefore increases MR `S_any` by one without increasing `A`",
            ]
        )
    else:
        lines.append("- no invalid-kind records in the analysis population")
    if excluded["unmatched_dependency"]:
        lines.append(
            f"- unmatched D2-clean dependency rows without a task outcome: {len(excluded['unmatched_dependency'])}"
        )
    lines.extend(
        [
            "",
            "## Reproduction of frozen P/S",
            "",
            f"- reproduced: {'yes' if reproduced else 'NO'}",
            f"- MR exact All observations: {mr.p_exact}/{mr.s_exact} ({PSCount(mr.p_exact, mr.s_exact).pct:.2f}%)",
            f"- MR range All observations: {mr.p_range}/{mr.s_range} ({PSCount(mr.p_range, mr.s_range).pct:.2f}%)",
            f"- VA exact All observations: {va.p_exact}/{va.s_exact} ({PSCount(va.p_exact, va.s_exact).pct:.2f}%)",
            f"- VA range S: {va.s_range} (expected 0)",
        ]
    )
    if diffs:
        lines.append("- differences:")
        lines.extend(f"  - {item}" for item in diffs)
    else:
        lines.append("- every per-model and aggregate P/S cell matches the frozen funnel tables")
    lines.extend(
        [
            "",
            "## All-constraint validity",
            "",
            f"- MR `S_any`: {mr.s_any} (not `S_exact + S_range` = {mr.s_exact + mr.s_range}; overlap both-kinds = {mr.both_kinds})",
            f"- MR `A/S_any`: {mr.a_all}/{mr.s_any} ({PSCount(mr.a_all, mr.s_any).pct:.2f}%)",
            f"- MR partition all_valid/mixed/none: {mr.all_valid}/{mr.mixed_validity}/{mr.none_valid} (sum {mr.all_valid + mr.mixed_validity + mr.none_valid})",
            f"- VA `S_any`: {va.s_any} (equals `S_exact`={va.s_exact}; VA has no range constraints)",
            f"- VA `A/S_any`: {va.a_all}/{va.s_any} ({PSCount(va.a_all, va.s_any).pct:.2f}%)",
            f"- VA partition all_valid/mixed/none: {va.all_valid}/{va.mixed_validity}/{va.none_valid} (sum {va.all_valid + va.mixed_validity + va.none_valid})",
            "- VA `A` is recomputed from record-level validity and is not assumed equal to `P_exact`",
            "",
            "## Sanity checks",
            "",
        ]
    )
    if sanity:
        lines.append("- FAILED:")
        lines.extend(f"  - {item}" for item in sanity)
    else:
        lines.extend(
            [
                "- 8,490 task--model observations per request condition: pass",
                "- 849 observations per model per condition: pass",
                "- model set is the ten paper models: pass",
                "- VA range `S_range` = 0: pass",
                "- `A <= S_any`: pass",
                "- `all_valid + mixed_validity + none_valid = S_any`: pass",
                "- aggregate percentages sum counts first: pass",
                "- percentages use two decimal places: pass",
                "- frozen P/S table fully reproduced: pass" if reproduced else "- frozen P/S table fully reproduced: FAIL",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    clean_ids = load_clean_ids(args.clean_task_ids)
    if len(clean_ids) != EXPECTED_N:
        raise SystemExit(f"clean task ids: got {len(clean_ids)}, expected {EXPECTED_N}")
    observations = load_observations(args.task_outcomes, clean_ids)
    excluded = attach_dependencies(observations, args.dependencies, clean_ids)
    stats = accumulate(observations)
    models = _model_order({_display(obs.model) for obs in observations.values()})
    expected = expected_from_funnel(_load_funnel(args.exact_csv), _load_funnel(args.range_csv))
    diffs = reproduce_or_stop(stats, expected, models)
    sanity = sanity_errors(stats, models)
    reproduced = not diffs
    report = render_report(
        args=args,
        models=models,
        stats=stats,
        excluded=excluded,
        n_obs=len(observations),
        reproduced=reproduced,
        diffs=diffs,
        sanity=sanity,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")
    if diffs or sanity:
        sys.stderr.write(report)
        sys.stderr.write("\nReproduction or sanity checks failed; final table was not written.\n")
        raise SystemExit(1)
    tex = render_tex(stats, models)
    args.output_tex.write_text(tex, encoding="utf-8")
    write_csv(args.output_csv, stats, models)
    print(tex)
    print(report)
    print(f"Saved: {args.output_tex}")
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.output_report}")


if __name__ == "__main__":
    main()
