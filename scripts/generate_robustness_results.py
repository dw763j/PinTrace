"""Generate auditable cross-Python robustness aggregates and paper tables.

Python 3.8/3.10/3.14 are read from completed robustness ``m5_compat_records.json``
files and classified with ``stages.failure_classifier``. Python 3.12 is read
from the corrected canonical task table. Stage rates use the full task--model
population. This script does not rerun generation, resolution, installation,
static checking, or BigCodeBench tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from plots.model_display import MODEL_LABEL, paper_display_label
from paths import ANALYSIS_ROBUSTNESS, EVALUATION, display_path
from stages.failure_classifier import classify_compatibility_layer, classify_ty_error

PYTHON_VERSIONS = ("3.8", "3.10", "3.12", "3.14")
NON_CANONICAL_PYTHON = {"38": "3.8", "310": "3.10", "314": "3.14"}
TRACK_TO_DATASET = {"d1": "BigCodeBench", "d2": "PinTrace"}
DATASET_TO_TRACK = {"BigCodeBench": "d1", "PinTrace": "d2"}
MODE_TO_CONDITION = {"requirements.txt": "MR", "inline": "VA"}
CONDITIONS = ("MR", "VA")
DATASETS = ("PinTrace", "BigCodeBench")
INSTALL_FAIL_LAYERS = frozenset({"environment_build", "infrastructure"})
RESOLVER_INPUT_STATUSES = frozenset({"resolved", "failed"})
KNOWN_RESOLVER_STATUSES = frozenset({"resolved", "failed", "empty"})
EXPECTED_TASKS = {"BigCodeBench": 813, "PinTrace": 1000}
EXPECTED_CELLS = 2 * 2 * 10 * 4
N_MODELS = 10
AGGREGATE_LABEL = "All observations"

EXPECTED_PY312_PINTRACE = {
    "MR": {
        "n": 10_000,
        "resolver_input": 9_235,
        "complete_lock": 8_234,
        "installation_completed": 7_779,
        "static_pass": 5_044,
    },
    "VA": {
        "n": 10_000,
        "resolver_input": 8_868,
        "complete_lock": 6_889,
        "installation_completed": 5_274,
        "static_pass": 3_432,
    },
}
EXPECTED_PY312_BCB_DYNAMIC = {
    "MR": 3_453,
    "VA": 1_335,
    "n": 8_130,
}

BY_MODEL_FIELDS = (
    "dataset",
    "request_condition",
    "model",
    "python_version",
    "n",
    "resolver_input",
    "complete_lock",
    "installation_completed",
    "static_pass",
    "dynamic_pass",
    "e2e_resolver_input_rate",
    "e2e_complete_lock_rate",
    "e2e_installation_completed_rate",
    "e2e_static_pass_rate",
    "e2e_dynamic_pass_rate",
    "cond_complete_lock_rate",
    "cond_installation_completed_rate",
    "cond_static_pass_rate",
)
ALL_OBS_FIELDS = tuple(name for name in BY_MODEL_FIELDS if name != "model") + ("n_models",)
DISTUTILS_FIELDS = (
    "dataset",
    "request_condition",
    "python_version",
    "model",
    "n",
    "complete_lock_count",
    "installation_failure_count",
    "missing_distutils_count",
    "missing_distutils_over_installation_failure",
    "missing_distutils_over_population",
)


class AssertionError_(Exception):
    """Raised when a regression or completeness assertion fails."""


@dataclass
class Diagnostics:
    unexpected_resolver_status: Counter[str] = field(default_factory=Counter)
    missing_resolver_status: int = 0
    duplicate_task_ids: list[str] = field(default_factory=list)
    missing_distutils_without_install_failure: int = 0
    ty_compatible_without_static_pass: int = 0
    unknown_compatibility_layers: Counter[str] = field(default_factory=Counter)
    model_name_variants: set[str] = field(default_factory=set)

    def merge(self, other: "Diagnostics") -> None:
        self.unexpected_resolver_status.update(other.unexpected_resolver_status)
        self.missing_resolver_status += other.missing_resolver_status
        self.duplicate_task_ids.extend(other.duplicate_task_ids)
        self.missing_distutils_without_install_failure += (
            other.missing_distutils_without_install_failure
        )
        self.ty_compatible_without_static_pass += other.ty_compatible_without_static_pass
        self.unknown_compatibility_layers.update(other.unknown_compatibility_layers)
        self.model_name_variants.update(other.model_name_variants)


@dataclass
class CellCounts:
    n: int = 0
    resolver_input: int = 0
    complete_lock: int = 0
    installation_completed: int = 0
    static_pass: int = 0
    dynamic_pass: int = 0
    installation_failure: int = 0
    missing_distutils: int = 0
    has_dynamic: bool = False

    def add(self, other: "CellCounts") -> None:
        self.n += other.n
        self.resolver_input += other.resolver_input
        self.complete_lock += other.complete_lock
        self.installation_completed += other.installation_completed
        self.static_pass += other.static_pass
        self.dynamic_pass += other.dynamic_pass
        self.installation_failure += other.installation_failure
        self.missing_distutils += other.missing_distutils
        self.has_dynamic = self.has_dynamic or other.has_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=Path("outputs/tosem_revision_20260812_python_1f1644e"),
    )
    parser.add_argument(
        "--py312-task-outcomes",
        type=Path,
        default=EVALUATION / "task_outcomes.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS_ROBUSTNESS,
    )
    parser.add_argument(
        "--old-output",
        type=Path,
        default=None,
        help="Previous robustness output, used only for the difference section.",
    )
    return parser.parse_args()


def _rate(numerator: int | None, denominator: int) -> float | None:
    if numerator is None or denominator == 0:
        return None
    return numerator / denominator


def _pct_text(numerator: int | None, denominator: int) -> str:
    rate = _rate(numerator, denominator)
    if rate is None:
        return "--"
    return f"{100.0 * rate:.2f}"


def _count_text(value: int) -> str:
    return f"{value:,}"


def _display_model(raw_name: str) -> str:
    return paper_display_label(raw_name, raw_name)


def _ordered_models(models: set[str]) -> list[str]:
    present = set(models)
    ordered = [label for label in MODEL_LABEL.values() if label in present]
    rest = sorted(present - set(ordered))
    return ordered + rest


def _parse_robustness_path(path: Path) -> tuple[str, str, str] | None:
    parts = path.parts
    track = next((part for part in parts if part in TRACK_TO_DATASET), None)
    mode = next((part for part in parts if part in MODE_TO_CONDITION), None)
    py_part = next(
        (part for part in parts if part.startswith("py") and part[2:].isdigit()),
        None,
    )
    if track is None or mode is None or py_part is None:
        return None
    python_version = NON_CANONICAL_PYTHON.get(py_part[2:])
    if python_version is None:
        return None
    return track, mode, python_version


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_raw_record(record: dict) -> dict:
    resolver_status = (record.get("resolver") or {}).get("status")
    compat = record.get("compat_results") or {}
    ty_result = compat.get("ty") or {}
    bcb_result = compat.get("bcb_test")
    layer = classify_compatibility_layer(ty_result, bcb_result)
    _, ty_install_category = classify_ty_error(ty_result)
    return {
        "task_id": record.get("task_id"),
        "resolver_status": resolver_status,
        "compatibility_layer": layer,
        "ty_compatible": ty_result.get("is_compatible") is True,
        "ty_install_category": ty_install_category,
        "bcb_status": (bcb_result or {}).get("status") if isinstance(bcb_result, dict) else None,
    }


def _accumulate_observation(
    counts: CellCounts,
    diagnostics: Diagnostics,
    observation: dict,
    *,
    has_dynamic: bool,
    seen_task_ids: set[str],
) -> None:
    task_id = str(observation.get("task_id") or "")
    if not task_id:
        raise AssertionError_("missing task_id in a robustness or canonical record")
    if task_id in seen_task_ids:
        diagnostics.duplicate_task_ids.append(task_id)
        return
    seen_task_ids.add(task_id)

    resolver_status = observation.get("resolver_status")
    layer = str(observation.get("compatibility_layer") or "")
    ty_compatible = observation.get("ty_compatible") is True
    ty_install_category = observation.get("ty_install_category")
    bcb_status = observation.get("bcb_status")

    counts.n += 1
    counts.has_dynamic = has_dynamic
    if resolver_status is None or resolver_status == "":
        diagnostics.missing_resolver_status += 1
    elif resolver_status not in KNOWN_RESOLVER_STATUSES:
        diagnostics.unexpected_resolver_status[str(resolver_status)] += 1
    if layer and layer not in {
        "compatible",
        "environment_build",
        "infrastructure",
        "code_or_syntax",
        "import_or_initialization",
        "api_mismatch_candidate",
        "dynamic_semantic",
        "other",
    }:
        diagnostics.unknown_compatibility_layers[layer] += 1

    resolver_input = resolver_status in RESOLVER_INPUT_STATUSES
    complete_lock = resolver_status == "resolved"
    installation_completed = complete_lock and layer not in INSTALL_FAIL_LAYERS
    static_pass = complete_lock and installation_completed and ty_compatible
    installation_failure = complete_lock and not installation_completed
    missing_distutils = installation_failure and ty_install_category == "missing_distutils"

    if resolver_input:
        counts.resolver_input += 1
    if complete_lock:
        counts.complete_lock += 1
    if installation_completed:
        counts.installation_completed += 1
    if static_pass:
        counts.static_pass += 1
    if has_dynamic and bcb_status == "pass":
        counts.dynamic_pass += 1
    if installation_failure:
        counts.installation_failure += 1
    if missing_distutils:
        counts.missing_distutils += 1
    if ty_install_category == "missing_distutils" and not installation_failure:
        diagnostics.missing_distutils_without_install_failure += 1
    if ty_compatible and not static_pass:
        diagnostics.ty_compatible_without_static_pass += 1


def _summarize_observations(
    observations: list[dict],
    *,
    has_dynamic: bool,
    raw_model: str,
) -> tuple[CellCounts, Diagnostics]:
    counts = CellCounts()
    diagnostics = Diagnostics()
    diagnostics.model_name_variants.add(raw_model)
    seen: set[str] = set()
    for observation in observations:
        _accumulate_observation(
            counts,
            diagnostics,
            observation,
            has_dynamic=has_dynamic,
            seen_task_ids=seen,
        )
    return counts, diagnostics


def _process_robustness_file(path_str: str) -> dict:
    path = Path(path_str)
    parsed = _parse_robustness_path(path)
    if parsed is None:
        raise AssertionError_(f"cannot parse robustness path: {path}")
    track, mode, python_version = parsed
    metrics_path = path.parent / "metrics_summary.json"
    metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
    raw_model = str((metrics or {}).get("model_name") or path.parent.parent.name)
    model = _display_model(raw_model)
    records = _load_json(path)
    if not isinstance(records, list):
        raise AssertionError_(f"{path} is not a JSON list")
    observations = [_classify_raw_record(record) for record in records]
    del records
    counts, cell_diag = _summarize_observations(
        observations,
        has_dynamic=track == "d1",
        raw_model=raw_model,
    )
    return {
        "key": (TRACK_TO_DATASET[track], MODE_TO_CONDITION[mode], model, python_version),
        "path": path_str,
        "counts": asdict(counts),
        "diagnostics": {
            "unexpected_resolver_status": dict(cell_diag.unexpected_resolver_status),
            "missing_resolver_status": cell_diag.missing_resolver_status,
            "duplicate_task_ids": cell_diag.duplicate_task_ids,
            "missing_distutils_without_install_failure": (
                cell_diag.missing_distutils_without_install_failure
            ),
            "ty_compatible_without_static_pass": cell_diag.ty_compatible_without_static_pass,
            "unknown_compatibility_layers": dict(cell_diag.unknown_compatibility_layers),
            "model_name_variants": sorted(cell_diag.model_name_variants),
        },
    }


def _diagnostics_from_dict(payload: dict) -> Diagnostics:
    return Diagnostics(
        unexpected_resolver_status=Counter(payload["unexpected_resolver_status"]),
        missing_resolver_status=payload["missing_resolver_status"],
        duplicate_task_ids=list(payload["duplicate_task_ids"]),
        missing_distutils_without_install_failure=payload[
            "missing_distutils_without_install_failure"
        ],
        ty_compatible_without_static_pass=payload["ty_compatible_without_static_pass"],
        unknown_compatibility_layers=Counter(payload["unknown_compatibility_layers"]),
        model_name_variants=set(payload["model_name_variants"]),
    )


def read_robustness_cells(root: Path) -> tuple[dict[tuple[str, str, str, str], CellCounts], Diagnostics]:
    cells: dict[tuple[str, str, str, str], CellCounts] = {}
    diagnostics = Diagnostics()
    paths = sorted(root.rglob("m5_compat_records.json"))
    path_strs = []
    for path in paths:
        parsed = _parse_robustness_path(path)
        if parsed is None:
            continue
        path_strs.append(str(path))
    if len(path_strs) != 120:
        raise AssertionError_(
            f"expected 120 non-3.12 m5_compat_records.json files under {root}, "
            f"found {len(path_strs)}"
        )
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_process_robustness_file, path): path for path in path_strs}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            key = tuple(result["key"])
            if key in cells:
                raise AssertionError_(f"duplicate robustness cell {key}: {result['path']}")
            cells[key] = CellCounts(**result["counts"])
            diagnostics.merge(_diagnostics_from_dict(result["diagnostics"]))
            done += 1
            print(
                f"[robustness {done}/120] {key[0]} {key[1]} {key[2]} Python {key[3]}",
                file=sys.stderr,
                flush=True,
            )
    return cells, diagnostics


def read_py312_cells(path: Path) -> tuple[dict[tuple[str, str, str, str], CellCounts], Diagnostics]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    diagnostics = Diagnostics()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            track = str(row["track"])
            mode = str(row["mode"])
            if track not in TRACK_TO_DATASET or mode not in MODE_TO_CONDITION:
                continue
            python_version = str(row.get("python_version") or "3.12")
            if python_version != "3.12":
                raise AssertionError_(
                    f"canonical task_outcomes.jsonl contains python_version={python_version!r}"
                )
            raw_model = str(row["model"])
            model = _display_model(raw_model)
            grouped[(TRACK_TO_DATASET[track], MODE_TO_CONDITION[mode], model)].append(
                {
                    "task_id": row.get("task_id"),
                    "resolver_status": row.get("resolver_status"),
                    "compatibility_layer": row.get("compatibility_layer"),
                    "ty_compatible": row.get("ty_compatible") is True,
                    "ty_install_category": row.get("ty_install_category"),
                    "bcb_status": row.get("bcb_status"),
                }
            )
            diagnostics.model_name_variants.add(raw_model)
    cells: dict[tuple[str, str, str, str], CellCounts] = {}
    for (dataset, condition, model), observations in grouped.items():
        counts, cell_diag = _summarize_observations(
            observations,
            has_dynamic=dataset == "BigCodeBench",
            raw_model=model,
        )
        cells[(dataset, condition, model, "3.12")] = counts
        diagnostics.merge(cell_diag)
    return cells, diagnostics


def _cell_row(
    dataset: str,
    condition: str,
    model: str | None,
    python_version: str,
    counts: CellCounts,
) -> dict:
    n = counts.n
    dynamic = counts.dynamic_pass if counts.has_dynamic else None
    row = {
        "dataset": dataset,
        "request_condition": condition,
        "python_version": python_version,
        "n": n,
        "resolver_input": counts.resolver_input,
        "complete_lock": counts.complete_lock,
        "installation_completed": counts.installation_completed,
        "static_pass": counts.static_pass,
        "dynamic_pass": dynamic,
        "e2e_resolver_input_rate": _rate(counts.resolver_input, n),
        "e2e_complete_lock_rate": _rate(counts.complete_lock, n),
        "e2e_installation_completed_rate": _rate(counts.installation_completed, n),
        "e2e_static_pass_rate": _rate(counts.static_pass, n),
        "e2e_dynamic_pass_rate": _rate(dynamic, n),
        "cond_complete_lock_rate": _rate(counts.complete_lock, counts.resolver_input),
        "cond_installation_completed_rate": _rate(
            counts.installation_completed, counts.complete_lock
        ),
        "cond_static_pass_rate": _rate(counts.static_pass, counts.installation_completed),
    }
    if model is not None:
        row["model"] = model
    return row


def _distutils_row(
    dataset: str,
    condition: str,
    python_version: str,
    model: str,
    counts: CellCounts,
) -> dict:
    return {
        "dataset": dataset,
        "request_condition": condition,
        "python_version": python_version,
        "model": model,
        "n": counts.n,
        "complete_lock_count": counts.complete_lock,
        "installation_failure_count": counts.installation_failure,
        "missing_distutils_count": counts.missing_distutils,
        "missing_distutils_over_installation_failure": _rate(
            counts.missing_distutils, counts.installation_failure
        ),
        "missing_distutils_over_population": _rate(counts.missing_distutils, counts.n),
    }


def aggregate_cells(
    cells: dict[tuple[str, str, str, str], CellCounts],
) -> dict[tuple[str, str, str], CellCounts]:
    aggregates: dict[tuple[str, str, str], CellCounts] = {}
    for (dataset, condition, _model, python_version), counts in cells.items():
        key = (dataset, condition, python_version)
        aggregates.setdefault(key, CellCounts()).add(counts)
    return aggregates


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def run_assertions(
    cells: dict[tuple[str, str, str, str], CellCounts],
    aggregates: dict[tuple[str, str, str], CellCounts],
    diagnostics: Diagnostics,
) -> list[str]:
    failures: list[str] = []
    _assert(
        len(cells) == EXPECTED_CELLS,
        f"expected {EXPECTED_CELLS} complete analysis units, found {len(cells)}",
        failures,
    )
    if diagnostics.duplicate_task_ids:
        failures.append(
            "duplicate task_id observations: " + ", ".join(diagnostics.duplicate_task_ids[:8])
        )

    models_by_group: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for dataset, condition, model, python_version in cells:
        models_by_group[(dataset, condition, python_version)].add(model)

    for dataset in DATASETS:
        expected_n = EXPECTED_TASKS[dataset]
        for condition in CONDITIONS:
            for python_version in PYTHON_VERSIONS:
                group = (dataset, condition, python_version)
                models = models_by_group.get(group, set())
                _assert(
                    len(models) == N_MODELS,
                    f"{dataset} {condition} Python {python_version}: "
                    f"expected {N_MODELS} models, found {len(models)} {sorted(models)}",
                    failures,
                )
                for model in _ordered_models(models):
                    key = (dataset, condition, model, python_version)
                    counts = cells.get(key)
                    if counts is None:
                        failures.append(f"missing cell {key}")
                        continue
                    _assert(
                        counts.n == expected_n,
                        f"{dataset} {condition} {model} Python {python_version}: "
                        f"expected {expected_n} tasks, found {counts.n}",
                        failures,
                    )
                    _assert(
                        counts.static_pass
                        <= counts.installation_completed
                        <= counts.complete_lock
                        <= counts.resolver_input
                        <= counts.n,
                        (
                            f"{dataset} {condition} {model} Python {python_version}: "
                            f"stage order violated "
                            f"static_pass={counts.static_pass} "
                            f"installation_completed={counts.installation_completed} "
                            f"complete_lock={counts.complete_lock} "
                            f"resolver_input={counts.resolver_input} n={counts.n}"
                        ),
                        failures,
                    )
                    _assert(
                        counts.missing_distutils <= counts.installation_failure <= counts.complete_lock,
                        (
                            f"{dataset} {condition} {model} Python {python_version}: "
                            f"distutils counts violated "
                            f"missing_distutils={counts.missing_distutils} "
                            f"installation_failure={counts.installation_failure} "
                            f"complete_lock={counts.complete_lock}"
                        ),
                        failures,
                    )

                pooled = CellCounts()
                for model in models:
                    pooled.add(cells[(dataset, condition, model, python_version)])
                aggregate = aggregates[(dataset, condition, python_version)]
                for field_name in (
                    "n",
                    "resolver_input",
                    "complete_lock",
                    "installation_completed",
                    "static_pass",
                    "dynamic_pass",
                    "installation_failure",
                    "missing_distutils",
                ):
                    got = getattr(aggregate, field_name)
                    exp = getattr(pooled, field_name)
                    _assert(
                        got == exp,
                        (
                            f"{dataset} {condition} Python {python_version} aggregate "
                            f"{field_name}={got} != sum of models {exp}"
                        ),
                        failures,
                    )
                _assert(
                    aggregate.static_pass
                    <= aggregate.installation_completed
                    <= aggregate.complete_lock
                    <= aggregate.resolver_input
                    <= aggregate.n,
                    (
                        f"{dataset} {condition} Python {python_version} All observations: "
                        f"stage order violated "
                        f"static_pass={aggregate.static_pass} "
                        f"installation_completed={aggregate.installation_completed} "
                        f"complete_lock={aggregate.complete_lock} "
                        f"resolver_input={aggregate.resolver_input} n={aggregate.n}"
                    ),
                    failures,
                )

    for condition, expected in EXPECTED_PY312_PINTRACE.items():
        got = aggregates[("PinTrace", condition, "3.12")]
        for field_name, exp in expected.items():
            actual = getattr(got, field_name)
            _assert(
                actual == exp,
                (
                    f"Python 3.12 PinTrace {condition} {field_name}: "
                    f"expected {exp}, found {actual}"
                ),
                failures,
            )
    for condition in CONDITIONS:
        got = aggregates[("BigCodeBench", condition, "3.12")]
        _assert(
            got.n == EXPECTED_PY312_BCB_DYNAMIC["n"],
            (
                f"Python 3.12 BigCodeBench {condition} n: "
                f"expected {EXPECTED_PY312_BCB_DYNAMIC['n']}, found {got.n}"
            ),
            failures,
        )
        _assert(
            got.dynamic_pass == EXPECTED_PY312_BCB_DYNAMIC[condition],
            (
                f"Python 3.12 BigCodeBench {condition} dynamic_pass: "
                f"expected {EXPECTED_PY312_BCB_DYNAMIC[condition]}, found {got.dynamic_pass}"
            ),
            failures,
        )
    return failures


def _write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = []
    for row in rows:
        item = {}
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, float):
                item[key] = f"{value:.10f}".rstrip("0").rstrip(".")
            elif value is None:
                item[key] = ""
            else:
                item[key] = value
        serialised.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(serialised)


def write_tex(path: Path, aggregates: dict[tuple[str, str, str], CellCounts]) -> None:
    def cell(dataset: str, condition: str, python_version: str, field_name: str) -> str:
        counts = aggregates[(dataset, condition, python_version)]
        numerator = getattr(counts, field_name)
        return _pct_text(numerator, counts.n)

    pintrace_rows = (
        ("Complete lock", "complete_lock"),
        ("Installation completed", "installation_completed"),
        ("Static validation passed", "static_pass"),
    )
    lines = [
        "% Auto-generated by scripts.generate_robustness_results.",
        "% End-to-end rates over the complete task--model population.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-Python robustness of adopted suggestions. "
        r"Panel~A reports PinTrace end-to-end rates; each request condition and "
        r"Python version uses 10{,}000 task--model observations. "
        r"Panel~B reports BigCodeBench dynamic test pass rates over 8{,}130 "
        r"observations. "
        r"The All observations percentage is computed after summing the ten "
        r"model-level numerators and denominators.}",
        r"\label{tab:python_robustness}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Outcome & Condition & Python 3.8 & Python 3.10 & Python 3.12 & Python 3.14 \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel A: PinTrace}} \\",
    ]
    for outcome, field_name in pintrace_rows:
        for condition in CONDITIONS:
            values = " & ".join(
                cell("PinTrace", condition, version, field_name) for version in PYTHON_VERSIONS
            )
            lines.append(f"{outcome} & {condition} & {values} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Panel B: BigCodeBench}} \\")
    for condition in CONDITIONS:
        values = " & ".join(
            cell("BigCodeBench", condition, version, "dynamic_pass")
            for version in PYTHON_VERSIONS
        )
        lines.append(f"Dynamic test passed & {condition} & {values} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _pp_change(later: CellCounts, baseline: CellCounts, field_name: str) -> float:
    later_rate = getattr(later, field_name) / later.n
    baseline_rate = getattr(baseline, field_name) / baseline.n
    return 100.0 * (later_rate - baseline_rate)


def _sign(value: float, tol: float = 1e-12) -> int:
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


def model_trend_agreement(
    cells: dict[tuple[str, str, str, str], CellCounts],
    aggregates: dict[tuple[str, str, str], CellCounts],
    dataset: str,
    condition: str,
    field_name: str,
    later_version: str,
    baseline_version: str = "3.10",
) -> tuple[int, int, float]:
    aggregate_delta = _pp_change(
        aggregates[(dataset, condition, later_version)],
        aggregates[(dataset, condition, baseline_version)],
        field_name,
    )
    aggregate_sign = _sign(aggregate_delta)
    models = {
        model
        for ds, cond, model, version in cells
        if ds == dataset and cond == condition and version == baseline_version
    }
    agreed = 0
    for model in models:
        delta = _pp_change(
            cells[(dataset, condition, model, later_version)],
            cells[(dataset, condition, model, baseline_version)],
            field_name,
        )
        if _sign(delta) == aggregate_sign:
            agreed += 1
    return agreed, len(models), aggregate_delta


def _load_old_by_model(path: Path | None) -> dict[tuple[str, str, str, str], dict]:
    if path is None:
        return {}
    csv_path = path / "python_robustness_by_model.csv"
    if not csv_path.is_file():
        return {}
    mapping = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = TRACK_TO_DATASET[row["track"]]
            condition = MODE_TO_CONDITION[row["mode"]]
            model = _display_model(row["model"])
            key = (dataset, condition, model, row["python_version"])
            mapping[key] = row
    return mapping


def render_report(
    *,
    args: argparse.Namespace,
    command: str,
    cells: dict[tuple[str, str, str, str], CellCounts],
    aggregates: dict[tuple[str, str, str], CellCounts],
    diagnostics: Diagnostics,
    old_cells: dict[tuple[str, str, str, str], dict],
) -> str:
    lines = [
        "# Cross-Python robustness verification report",
        "",
        "## Data sources",
        "",
        f"- robustness runs (Python 3.8, 3.10, 3.14): `{display_path(args.robustness_root)}`",
        f"- Python 3.12 corrected canonical table: `{display_path(args.py312_task_outcomes)}`",
        f"- output directory: `{display_path(args.output)}`",
        "- classifiers: `stages.failure_classifier.classify_compatibility_layer`, "
        "`stages.failure_classifier.classify_ty_error`",
        "- Python 3.8/3.10/3.14 records are classified from each `m5_compat_records.json`. "
        "Python 3.12 uses the stored `resolver_status`, `compatibility_layer`, "
        "`ty_compatible`, `ty_install_category`, and `bcb_status` fields.",
        "- this script does not rerun model generation, dependency resolution, "
        "installation, static checking, or BigCodeBench tests",
        "",
        "## Denominators and stage definitions",
        "",
        "- request conditions: `requirements.txt` -> MR; `inline` -> VA",
        "- BigCodeBench / D1: 813 tasks per model, condition, and Python version; "
        "All observations uses 8,130 observations",
        "- PinTrace / D2: 1,000 tasks per model, condition, and Python version; "
        "All observations uses 10,000 observations",
        "- each task--model observation contributes to a stage at most once",
        "- `resolver_input`: resolver status is `resolved` or `failed`; `empty` is excluded",
        "- `complete_lock`: resolver status is `resolved`",
        "- `installation_completed`: `complete_lock` and compatibility layer is not "
        "`environment_build` or `infrastructure`",
        "- `static_pass`: `complete_lock`, `installation_completed`, and `ty is_compatible == true`",
        "- `dynamic_pass` (BigCodeBench only): `bcb_test.status == pass` over the full "
        "813-task population, not conditioned on resolution, installation, or static validation",
        "- end-to-end rates use the complete task--model population as the denominator",
        "- stage-conditional rates: `complete_lock / resolver_input`, "
        "`installation_completed / complete_lock`, "
        "`static_pass / installation_completed`",
        "- All observations sums model-level numerators and denominators before computing "
        "the percentage",
        "",
        "## Completeness checks",
        "",
        f"- analysis units: {len(cells)} (expected {EXPECTED_CELLS})",
        f"- models per dataset/condition/Python version: {N_MODELS}",
        f"- raw model-name variants unified to paper labels: "
        f"{', '.join(sorted(diagnostics.model_name_variants))}",
        f"- missing resolver status: {diagnostics.missing_resolver_status}",
        f"- unexpected resolver statuses: "
        f"{dict(diagnostics.unexpected_resolver_status) or 'none'}",
        f"- duplicate task ids: {len(diagnostics.duplicate_task_ids)}",
        f"- unknown compatibility layers: "
        f"{dict(diagnostics.unknown_compatibility_layers) or 'none'}",
        f"- `ty_compatible=true` observations excluded from `static_pass` because they "
        f"lack a complete lock or completed installation: "
        f"{diagnostics.ty_compatible_without_static_pass}",
        f"- `missing_distutils` outside installation-after-lock failures: "
        f"{diagnostics.missing_distutils_without_install_failure}",
        "",
        "## Python 3.12 regression",
        "",
    ]
    for condition, expected in EXPECTED_PY312_PINTRACE.items():
        got = aggregates[("PinTrace", condition, "3.12")]
        lines.append(f"### PinTrace {condition}")
        lines.append("")
        for field_name, exp in expected.items():
            actual = getattr(got, field_name)
            status = "pass" if actual == exp else "FAIL"
            lines.append(
                f"- {field_name}: {_count_text(actual)} "
                f"(expected {_count_text(exp)}; {status})"
            )
        lines.append("")
    lines.append("### BigCodeBench dynamic pass")
    lines.append("")
    for condition in CONDITIONS:
        got = aggregates[("BigCodeBench", condition, "3.12")]
        exp = EXPECTED_PY312_BCB_DYNAMIC[condition]
        status = "pass" if got.dynamic_pass == exp and got.n == EXPECTED_PY312_BCB_DYNAMIC["n"] else "FAIL"
        lines.append(
            f"- {condition}: {_count_text(got.dynamic_pass)}/{_count_text(got.n)} "
            f"(expected {_count_text(exp)}/{_count_text(EXPECTED_PY312_BCB_DYNAMIC['n'])}; {status})"
        )
    lines.extend(["", "## Aggregate counts and end-to-end percentages", ""])

    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        lines.append("")
        for condition in CONDITIONS:
            lines.append(f"#### {condition}")
            lines.append("")
            for version in PYTHON_VERSIONS:
                counts = aggregates[(dataset, condition, version)]
                items = [
                    f"n={_count_text(counts.n)}",
                    (
                        f"resolver_input={_count_text(counts.resolver_input)} "
                        f"({_pct_text(counts.resolver_input, counts.n)}%)"
                    ),
                    (
                        f"complete_lock={_count_text(counts.complete_lock)} "
                        f"({_pct_text(counts.complete_lock, counts.n)}%)"
                    ),
                    (
                        f"installation_completed={_count_text(counts.installation_completed)} "
                        f"({_pct_text(counts.installation_completed, counts.n)}%)"
                    ),
                    (
                        f"static_pass={_count_text(counts.static_pass)} "
                        f"({_pct_text(counts.static_pass, counts.n)}%)"
                    ),
                ]
                if counts.has_dynamic:
                    items.append(
                        f"dynamic_pass={_count_text(counts.dynamic_pass)} "
                        f"({_pct_text(counts.dynamic_pass, counts.n)}%)"
                    )
                items.append(
                    "conditional: "
                    f"lock|input={_pct_text(counts.complete_lock, counts.resolver_input)}%, "
                    f"install|lock={_pct_text(counts.installation_completed, counts.complete_lock)}%, "
                    f"static|install={_pct_text(counts.static_pass, counts.installation_completed)}%"
                )
                lines.append(f"- Python {version}: " + "; ".join(items))
            lines.append("")

    lines.extend(["## Percentage-point changes from Python 3.10", ""])
    trend_fields = {
        "PinTrace": (
            ("complete_lock", "Complete lock"),
            ("installation_completed", "Installation completed"),
            ("static_pass", "Static validation passed"),
        ),
        "BigCodeBench": (
            ("complete_lock", "Complete lock"),
            ("installation_completed", "Installation completed"),
            ("static_pass", "Static validation passed"),
            ("dynamic_pass", "Dynamic test passed"),
        ),
    }
    for dataset, fields in trend_fields.items():
        lines.append(f"### {dataset}")
        lines.append("")
        for condition in CONDITIONS:
            lines.append(f"#### {condition}")
            lines.append("")
            for field_name, label in fields:
                for later in ("3.12", "3.14"):
                    agreed, total, delta = model_trend_agreement(
                        cells, aggregates, dataset, condition, field_name, later
                    )
                    direction = "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
                    lines.append(
                        f"- {label}, 3.10 to {later}: {delta:+.2f} pp ({direction}); "
                        f"{agreed}/{total} models move in the same direction"
                    )
            lines.append("")

    lines.extend(["## Distutils contribution to installation failures", ""])
    lines.append(
        "Only observations that produced a complete lock and then failed installation "
        "enter `installation_failure_count` and `missing_distutils_count`. "
        "`missing_distutils` is the `ty_install_category` returned by "
        "`classify_ty_error`. Removing this error would not imply that the task would "
        "pass static or dynamic validation."
    )
    lines.append("")
    for dataset in DATASETS:
        for condition in CONDITIONS:
            lines.append(f"### {dataset} {condition}")
            lines.append("")
            for version in PYTHON_VERSIONS:
                counts = aggregates[(dataset, condition, version)]
                lines.append(
                    f"- Python {version}: missing_distutils="
                    f"{_count_text(counts.missing_distutils)} / installation_failure="
                    f"{_count_text(counts.installation_failure)} "
                    f"({_pct_text(counts.missing_distutils, counts.installation_failure)}% of "
                    f"installation failures); "
                    f"{_count_text(counts.missing_distutils)}/{_count_text(counts.n)} "
                    f"({_pct_text(counts.missing_distutils, counts.n)}% of the complete "
                    f"population); complete_lock={_count_text(counts.complete_lock)}"
                )
            lines.append("")

    lines.extend(["## Differences from the previous robustness output", ""])
    if not old_cells:
        lines.append(
            f"- previous output not found at `{args.old_output}`; difference table omitted"
        )
    else:
        lines.extend(
            [
                "The previous script inferred installation as "
                "`total - install_error - infra_error` and counted every "
                "`ty_compatible=true` observation as a static pass. "
                "That definition does not require `resolver.status == resolved`, so "
                "empty no-dependency observations entered both the installation and "
                "static-pass numerators. Installation rates could therefore exceed "
                "resolution rates, and the static rates were not comparable to the "
                "corrected RQ3 end-to-end definition.",
                "",
                "All observations previously averaged model-level percentages. This "
                "revision sums counts first. Because every model has the same task "
                "count, the percentages can match, but the pooled denominators are now "
                "auditable.",
                "",
                "The previous `distutils_sensitivity.csv` metric excluded "
                "`missing_distutils` observations from the denominator and still "
                "counted no-dependency static passes. This revision reports only the "
                "contribution of `missing_distutils` to post-lock installation "
                "failures.",
                "",
                "Complete-lock counts should match the previous `resolved` counts. "
                "Installation and static-pass counts are lower once empty resolver "
                "observations are removed.",
                "",
            ]
        )
        mismatches = []
        for key, old in sorted(old_cells.items()):
            dataset, condition, model, version = key
            if key not in cells:
                mismatches.append(f"old cell missing from new output: {key}")
                continue
            new = cells[key]
            old_resolved = int(float(old["resolved"]))
            if new.complete_lock != old_resolved:
                mismatches.append(
                    f"{dataset} {condition} {model} Python {version}: "
                    f"complete_lock {new.complete_lock} != old resolved {old_resolved}"
                )
        if mismatches:
            lines.append("Complete-lock mismatches against the old `resolved` column:")
            lines.append("")
            for item in mismatches[:20]:
                lines.append(f"- {item}")
            if len(mismatches) > 20:
                lines.append(f"- ... {len(mismatches) - 20} more")
        else:
            lines.append(
                "- complete_lock matches the previous `resolved` count in every overlapping cell"
            )
        lines.append("")
        lines.append("All-observations installation and static-pass revisions:")
        lines.append("")
        for dataset in DATASETS:
            for condition in CONDITIONS:
                for version in PYTHON_VERSIONS:
                    new = aggregates[(dataset, condition, version)]
                    old_installed = 0
                    old_static = 0
                    found = 0
                    for (ds, cond, _model, py), old in old_cells.items():
                        if (ds, cond, py) != (dataset, condition, version):
                            continue
                        old_installed += int(float(old["installed"]))
                        old_static += int(float(old["ty_pass"]))
                        found += 1
                    if found != N_MODELS:
                        continue
                    lines.append(
                        f"- {dataset} {condition} Python {version}: "
                        f"installation {_count_text(old_installed)} -> "
                        f"{_count_text(new.installation_completed)} "
                        f"({_pct_text(old_installed, new.n)}% -> "
                        f"{_pct_text(new.installation_completed, new.n)}%); "
                        f"static {_count_text(old_static)} -> "
                        f"{_count_text(new.static_pass)} "
                        f"({_pct_text(old_static, new.n)}% -> "
                        f"{_pct_text(new.static_pass, new.n)}%)"
                    )
        lines.append("")

    lines.extend(
        [
            "## Reproduction command",
            "",
            "```bash",
            command,
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    command = (
        ".venv/bin/python -m scripts.generate_robustness_results "
        f"--robustness-root {args.robustness_root} "
        f"--py312-task-outcomes {args.py312_task_outcomes} "
        f"--output {args.output}"
    )
    robustness_cells, robustness_diag = read_robustness_cells(args.robustness_root)
    py312_cells, py312_diag = read_py312_cells(args.py312_task_outcomes)
    overlap = set(robustness_cells) & set(py312_cells)
    if overlap:
        raise AssertionError_(f"overlapping cells between robustness runs and Python 3.12: {sorted(overlap)[:4]}")
    cells = {**robustness_cells, **py312_cells}
    diagnostics = Diagnostics()
    diagnostics.merge(robustness_diag)
    diagnostics.merge(py312_diag)
    aggregates = aggregate_cells(cells)
    failures = run_assertions(cells, aggregates, diagnostics)
    if failures:
        print("Python robustness assertions failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        raise SystemExit(1)

    models = _ordered_models({model for _ds, _cond, model, _py in cells})
    by_model_rows = [
        _cell_row(dataset, condition, model, version, cells[(dataset, condition, model, version)])
        for dataset in DATASETS
        for condition in CONDITIONS
        for model in models
        for version in PYTHON_VERSIONS
    ]
    all_obs_rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for version in PYTHON_VERSIONS:
                row = _cell_row(dataset, condition, None, version, aggregates[(dataset, condition, version)])
                row["n_models"] = N_MODELS
                all_obs_rows.append(row)
    distutils_rows = [
        _distutils_row(dataset, condition, version, model, cells[(dataset, condition, model, version)])
        for dataset in DATASETS
        for condition in CONDITIONS
        for model in models
        for version in PYTHON_VERSIONS
    ]
    distutils_rows.extend(
        _distutils_row(dataset, condition, version, AGGREGATE_LABEL, aggregates[(dataset, condition, version)])
        for dataset in DATASETS
        for condition in CONDITIONS
        for version in PYTHON_VERSIONS
    )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "python_robustness_by_model.csv", by_model_rows, BY_MODEL_FIELDS)
    _write_csv(args.output / "python_robustness_all_observations.csv", all_obs_rows, ALL_OBS_FIELDS)
    _write_csv(args.output / "distutils_failure_contribution.csv", distutils_rows, DISTUTILS_FIELDS)
    write_tex(args.output / "python_robustness_table.tex", aggregates)
    old_cells = _load_old_by_model(args.old_output)
    report = render_report(
        args=args,
        command=command,
        cells=cells,
        aggregates=aggregates,
        diagnostics=diagnostics,
        old_cells=old_cells,
    )
    (args.output / "python_robustness_report.md").write_text(report, encoding="utf-8")
    config = {
        "command": command,
        "robustness_root": display_path(args.robustness_root),
        "py312_task_outcomes": display_path(args.py312_task_outcomes),
        "output": display_path(args.output),
        "old_output": display_path(args.old_output) if args.old_output else None,
        "python_versions": list(PYTHON_VERSIONS),
        "n_models": N_MODELS,
        "n_cells": len(cells),
        "tasks_per_model": EXPECTED_TASKS,
        "condition_map": MODE_TO_CONDITION,
        "dataset_map": TRACK_TO_DATASET,
        "classifiers": {
            "compatibility_layer": "stages.failure_classifier.classify_compatibility_layer",
            "ty_error": "stages.failure_classifier.classify_ty_error",
        },
        "stage_definitions": {
            "resolver_input": "resolver status in {resolved, failed}",
            "complete_lock": "resolver status == resolved",
            "installation_completed": (
                "resolver status == resolved and compatibility layer "
                "not in {environment_build, infrastructure}"
            ),
            "static_pass": (
                "complete_lock and installation_completed and ty is_compatible == true"
            ),
            "dynamic_pass": (
                "BigCodeBench only; bcb_test.status == pass over the full 813-task population"
            ),
            "end_to_end_denominator": "complete task--model population",
            "all_observations": "sum model-level counts, then compute the percentage",
        },
        "distutils_definition": (
            "missing_distutils from classify_ty_error ty_install_category, "
            "restricted to complete_lock observations that fail installation"
        ),
    }
    (args.output / "python_robustness_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
