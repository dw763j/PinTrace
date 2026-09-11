"""RQ3 dynamic-execution outcomes on the complete 813-task BigCodeBench subset.

The figure reports unrestricted D1 Python 3.12 observations.  It does not
filter on OSV coverage, resolver status, compatibility layer, constraint
presence, or static-validation outcome.  Panel (a) uses all observations as
the denominator.  Conditional dynamic-pass rates among statically compatible
observations are reported in the verification CSV and report only.

Usage (from project root)::

    .venv/bin/python -m plots.rq3_dynamic_outcomes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plots.model_display import MODEL_LABEL, paper_display_label
from paths import EVALUATION, PAPER_RESULTS, display_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = PAPER_RESULTS / "figures"
DEFAULT_INPUT = EVALUATION / "task_outcomes.jsonl"
DEFAULT_OUTPUT_DIR = PLOTS_DIR
DEFAULT_MANUSCRIPT_PDF = MANUSCRIPT_DIR / "rq3_dynamic_outcomes.pdf"

EXPECTED_INPUT_SHA256 = "88750aa52441dbf1db1eda0f208a60eb0aa3290fed6871b849c8958d9998bba7"

TRACK = "d1"
PYTHON_VERSION = "3.12"
MODE_TO_CONDITION = {"requirements.txt": "MR", "inline": "VA"}
CONDITIONS = ("MR", "VA")
AGGREGATE_LABEL = "All observations"
AGGREGATE_RAW = "ALL"
N_MODELS = 10
N_TASKS = 813
N_OBS = N_MODELS * N_TASKS
MIN_LABEL_PCT = 5.0

ALLOWED_BCB = frozenset({"pass", "fail", "error", "runner_error"})
DYNAMIC_PASS = "pass"
STATUS_FIELDS = (
    "n_dynamic_pass",
    "n_test_failure",
    "n_execution_error",
    "n_runner_error",
)
JOINT_KEYS = (
    "static_pass_dynamic_pass",
    "static_pass_dynamic_nonpass",
    "static_fail_dynamic_pass",
    "static_fail_dynamic_nonpass",
)
JOINT_DISPLAY = {
    "static_pass_dynamic_pass": "Static pass + dynamic pass",
    "static_pass_dynamic_nonpass": "Static pass + dynamic non-pass",
    "static_fail_dynamic_pass": "Static fail + dynamic pass",
    "static_fail_dynamic_nonpass": "Static fail + dynamic non-pass",
}
JOINT_COLORS = {
    "static_pass_dynamic_pass": "#117733",
    "static_pass_dynamic_nonpass": "#DDCC77",
    "static_fail_dynamic_pass": "#44AA99",
    "static_fail_dynamic_nonpass": "#CC6677",
}
# Wong/Okabe–Ito colours already used in rq3_failure_layers.py.
CONDITION_COLORS = {"MR": "#0072B2", "VA": "#D55E00"}
DISPLAY_ORDER = tuple(MODEL_LABEL.values())
DISPLAY_TO_CANONICAL_RAW = {label: raw for raw, label in MODEL_LABEL.items()}
REQUIRED_FIELDS = (
    "track",
    "mode",
    "model",
    "task_id",
    "python_version",
    "ty_compatible",
    "bcb_status",
)
CSV_FIELDS = (
    "condition",
    "model_raw",
    "model_display",
    "is_aggregate",
    "n_observations",
    "n_dynamic_pass",
    "n_test_failure",
    "n_execution_error",
    "n_runner_error",
    "dynamic_pass_rate",
    "n_static_pass",
    "n_static_fail",
    "n_static_pass_dynamic_pass",
    "n_static_pass_dynamic_nonpass",
    "n_static_fail_dynamic_pass",
    "n_static_fail_dynamic_nonpass",
    "pct_static_pass_dynamic_pass",
    "pct_static_pass_dynamic_nonpass",
    "pct_static_fail_dynamic_pass",
    "pct_static_fail_dynamic_nonpass",
    "n_static_compatible",
    "n_dynamic_pass_among_static_compatible",
    "conditional_dynamic_pass_rate",
    "n_execution_error_static_incompatible",
    "pct_execution_error_static_incompatible",
)

EXPECTED_DYNAMIC = {
    "MR": {
        "total": 8_130,
        "pass": 3_453,
        "pass_pct": 42.47,
        "fail": 3_872,
        "error": 796,
        "error_pct": 9.79,
        "runner_error": 9,
    },
    "VA": {
        "total": 8_130,
        "pass": 1_335,
        "pass_pct": 16.42,
        "fail": 1_567,
        "error": 5_227,
        "error_pct": 64.29,
        "runner_error": 1,
    },
}
EXPECTED_PASS_DIFF_PP = 26.05
EXPECTED_VA_LOWER = 9
EXPECTED_EXCEPTION = "GPT-5.4"
EXPECTED_GPT = {
    "MR": (401, 813, 49.32),
    "VA": (404, 813, 49.69),
}
EXPECTED_MODEL_RANGE = {
    "MR": (31.61, 51.91),
    "VA": (8.00, 49.69),
}
EXPECTED_JOINT = {
    "MR": {
        "static_pass_dynamic_pass": (3_205, 39.42),
        "static_pass_dynamic_nonpass": (3_570, 43.91),
        "static_fail_dynamic_pass": (248, 3.05),
        "static_fail_dynamic_nonpass": (1_107, 13.62),
    },
    "VA": {
        "static_pass_dynamic_pass": (1_204, 14.81),
        "static_pass_dynamic_nonpass": (1_743, 21.44),
        "static_fail_dynamic_pass": (131, 1.61),
        "static_fail_dynamic_nonpass": (5_052, 62.14),
    },
}
EXPECTED_CONDITIONAL = {
    "MR": (6_775, 3_205, 47.31),
    "VA": (2_947, 1_204, 40.86),
}
EXPECTED_ERROR_STATIC_FAIL = {
    "MR": (796, 737, 92.59),
    "VA": (5_227, 4_831, 92.42),
}

TEXT_MAIN = "#222222"
TEXT_MUTED = "#5A5A5A"
GUIDE_COLOR = "#C8C8C8"
CONNECTOR_COLOR = "#B0B0B0"
SPINE_COLOR = "#8A8A8A"
AGG_EDGE = "#333333"


@dataclass(frozen=True)
class Assertion:
    name: str
    passed: bool
    detail: str


@dataclass
class RowStat:
    condition: str
    model_raw: str
    model_display: str
    is_aggregate: bool
    n_observations: int
    n_dynamic_pass: int
    n_test_failure: int
    n_execution_error: int
    n_runner_error: int
    n_static_pass: int
    n_static_fail: int
    joint: dict[str, int]
    n_execution_error_static_incompatible: int

    @property
    def dynamic_pass_rate(self) -> float:
        if self.n_observations == 0:
            raise ValueError(
                f"zero observations for {self.condition} {self.model_display}"
            )
        return 100.0 * self.n_dynamic_pass / self.n_observations

    def joint_pct(self, key: str) -> float:
        if self.n_observations == 0:
            raise ValueError(
                f"zero observations for {self.condition} {self.model_display}"
            )
        return 100.0 * self.joint[key] / self.n_observations

    @property
    def conditional_dynamic_pass_rate(self) -> float | None:
        if self.n_static_pass == 0:
            return None
        return 100.0 * self.joint["static_pass_dynamic_pass"] / self.n_static_pass

    @property
    def pct_execution_error_static_incompatible(self) -> float | None:
        if self.n_execution_error == 0:
            return None
        return 100.0 * self.n_execution_error_static_incompatible / self.n_execution_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manuscript-pdf", type=Path, default=DEFAULT_MANUSCRIPT_PDF)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_geometry(path: Path) -> dict[str, float | int]:
    payload = path.read_bytes()
    width, height = struct.unpack(">II", payload[16:24])
    info: dict[str, float | int] = {"width_px": width, "height_px": height}
    marker = payload.find(b"pHYs")
    if marker != -1:
        pixels_per_unit_x, pixels_per_unit_y, unit = struct.unpack(
            ">IIB", payload[marker + 4 : marker + 13]
        )
        if unit == 1:
            info["dpi_x"] = round(pixels_per_unit_x * 0.0254, 2)
            info["dpi_y"] = round(pixels_per_unit_y * 0.0254, 2)
            info["width_in"] = round(width / info["dpi_x"], 2)
            info["height_in"] = round(height / info["dpi_y"], 2)
    return info


def pdf_geometry(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    boxes = re.findall(rb"/MediaBox\s*\[([^\]]+)\]", payload)
    fonts = sorted(
        item.decode("ascii", errors="replace")
        for item in set(re.findall(rb"/BaseFont\s*/([^\s/]+)", payload))
    )
    subtypes = sorted(
        item.decode("ascii", errors="replace")
        for item in set(
            re.findall(
                rb"/Subtype\s*/(Type1|TrueType|Type0|CIDFontType0|CIDFontType2)",
                payload,
            )
        )
    )
    info: dict[str, object] = {"fonts": fonts, "subtypes": subtypes}
    if boxes:
        numbers = [float(part) for part in boxes[0].split()]
        width_pt = numbers[2] - numbers[0]
        height_pt = numbers[3] - numbers[1]
        info["width_pt"] = width_pt
        info["height_pt"] = height_pt
        info["width_in"] = round(width_pt / 72.0, 2)
        info["height_in"] = round(height_pt / 72.0, 2)
    return info


def round_pct(value: float, digits: int = 2) -> float:
    return round(value, digits)


def display_for(model_raw: str) -> str:
    label = paper_display_label(model_raw, model_raw)
    if label not in MODEL_LABEL.values():
        raise ValueError(
            f"unknown model {model_raw!r} mapped to {label!r}; "
            f"expected one of {list(MODEL_LABEL.values())}"
        )
    return label


def missing_fields(row: dict, index: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"record {index} is missing fields {missing}")


def as_bool(value: object, field: str, index: int) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"record {index}: {field} has unexpected value {value!r}; expected a JSON boolean"
    )


def as_bcb_status(value: object, index: int) -> str:
    if value is None or value == "":
        raise ValueError(f"record {index}: bcb_status is missing")
    status = str(value)
    if status not in ALLOWED_BCB:
        raise ValueError(
            f"record {index}: unexpected bcb_status {value!r}; "
            f"expected one of {sorted(ALLOWED_BCB)}"
        )
    return status


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    return rows


def joint_key(static_pass: bool, dynamic_pass: bool) -> str:
    static_part = "static_pass" if static_pass else "static_fail"
    dynamic_part = "dynamic_pass" if dynamic_pass else "dynamic_nonpass"
    return f"{static_part}_{dynamic_part}"


def empty_counts() -> dict[str, int]:
    counts = {
        "n_observations": 0,
        "n_dynamic_pass": 0,
        "n_test_failure": 0,
        "n_execution_error": 0,
        "n_runner_error": 0,
        "n_static_pass": 0,
        "n_static_fail": 0,
        "n_execution_error_static_incompatible": 0,
    }
    for key in JOINT_KEYS:
        counts[key] = 0
    return counts


def accumulate(counts: dict[str, int], static_pass: bool, status: str) -> None:
    dynamic_pass = status == DYNAMIC_PASS
    counts["n_observations"] += 1
    if status == "pass":
        counts["n_dynamic_pass"] += 1
    elif status == "fail":
        counts["n_test_failure"] += 1
    elif status == "error":
        counts["n_execution_error"] += 1
        if not static_pass:
            counts["n_execution_error_static_incompatible"] += 1
    elif status == "runner_error":
        counts["n_runner_error"] += 1
    if static_pass:
        counts["n_static_pass"] += 1
    else:
        counts["n_static_fail"] += 1
    counts[joint_key(static_pass, dynamic_pass)] += 1


def select_observations(rows: list[dict]) -> list[dict]:
    seen: set[tuple[object, object, object, object]] = set()
    selected: list[dict] = []
    unknown_modes: list[str] = []

    for index, row in enumerate(rows):
        missing_fields(row, index)
        key = (row["track"], row["mode"], row["model"], row["task_id"])
        if key in seen:
            raise ValueError(f"duplicate observation key {key}")
        seen.add(key)
        if row["track"] != TRACK:
            continue
        if str(row["python_version"]) != PYTHON_VERSION:
            continue
        mode = row["mode"]
        if mode not in MODE_TO_CONDITION:
            unknown_modes.append(
                f"mode={mode!r} model={row['model']!r} task_id={row['task_id']!r}"
            )
            continue
        as_bool(row["ty_compatible"], "ty_compatible", index)
        as_bcb_status(row["bcb_status"], index)
        selected.append(row)

    if unknown_modes:
        preview = "\n  ".join(unknown_modes[:20])
        raise ValueError(f"unknown request modes:\n  {preview}")
    if not selected:
        raise ValueError("D1 Python 3.12 filter selected zero observations")
    return selected


def build_stats(selected: list[dict]) -> dict[str, list[RowStat]]:
    raw_by_display: dict[str, str] = {}
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(empty_counts)
    tasks: dict[str, set[str]] = {condition: set() for condition in CONDITIONS}

    for row in selected:
        model_raw = str(row["model"])
        display = display_for(model_raw)
        canonical_raw = DISPLAY_TO_CANONICAL_RAW.get(display)
        if canonical_raw is None:
            raise ValueError(f"no canonical raw name for display {display!r} from {model_raw!r}")
        previous = raw_by_display.get(display)
        if previous is not None and previous != canonical_raw:
            raise ValueError(
                f"display name {display!r} maps from both {previous!r} and {canonical_raw!r}"
            )
        raw_by_display[display] = canonical_raw
        condition = MODE_TO_CONDITION[str(row["mode"])]
        tasks[condition].add(str(row["task_id"]))
        accumulate(
            counts[(condition, display)],
            static_pass=bool(row["ty_compatible"]),
            status=str(row["bcb_status"]),
        )

    missing_models = [name for name in DISPLAY_ORDER if name not in raw_by_display]
    extra_models = sorted(set(raw_by_display) - set(DISPLAY_ORDER))
    if missing_models or extra_models:
        raise ValueError(
            f"model set mismatch; missing={missing_models} extra={extra_models}"
        )
    for condition in CONDITIONS:
        n_tasks = len(tasks[condition])
        if n_tasks != N_TASKS:
            raise ValueError(f"{condition} has {n_tasks} tasks; expected {N_TASKS}")

    by_condition: dict[str, list[RowStat]] = {}
    for condition in CONDITIONS:
        model_rows: list[RowStat] = []
        aggregate = empty_counts()
        for display in DISPLAY_ORDER:
            item = counts[(condition, display)]
            for key, value in item.items():
                aggregate[key] += value
            model_rows.append(stat_from_counts(condition, raw_by_display[display], display, False, item))
        model_rows.append(
            stat_from_counts(condition, AGGREGATE_RAW, AGGREGATE_LABEL, True, aggregate)
        )
        by_condition[condition] = model_rows
    return by_condition


def stat_from_counts(
    condition: str,
    model_raw: str,
    model_display: str,
    is_aggregate: bool,
    counts: dict[str, int],
) -> RowStat:
    status_sum = sum(counts[field] for field in STATUS_FIELDS)
    if status_sum != counts["n_observations"]:
        raise ValueError(
            f"{condition} {model_display}: status counts {status_sum} "
            f"!= n={counts['n_observations']}"
        )
    joint = {key: counts[key] for key in JOINT_KEYS}
    if sum(joint.values()) != counts["n_observations"]:
        raise ValueError(
            f"{condition} {model_display}: joint counts {sum(joint.values())} "
            f"!= n={counts['n_observations']}"
        )
    if counts["n_static_pass"] + counts["n_static_fail"] != counts["n_observations"]:
        raise ValueError(f"{condition} {model_display}: static split does not cover n")
    return RowStat(
        condition=condition,
        model_raw=model_raw,
        model_display=model_display,
        is_aggregate=is_aggregate,
        n_observations=counts["n_observations"],
        n_dynamic_pass=counts["n_dynamic_pass"],
        n_test_failure=counts["n_test_failure"],
        n_execution_error=counts["n_execution_error"],
        n_runner_error=counts["n_runner_error"],
        n_static_pass=counts["n_static_pass"],
        n_static_fail=counts["n_static_fail"],
        joint=joint,
        n_execution_error_static_incompatible=counts[
            "n_execution_error_static_incompatible"
        ],
    )


def csv_records(by_condition: dict[str, list[RowStat]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for stat in by_condition[condition]:
            cond_rate = stat.conditional_dynamic_pass_rate
            err_share = stat.pct_execution_error_static_incompatible
            records.append(
                {
                    "condition": condition,
                    "model_raw": stat.model_raw,
                    "model_display": stat.model_display,
                    "is_aggregate": "true" if stat.is_aggregate else "false",
                    "n_observations": stat.n_observations,
                    "n_dynamic_pass": stat.n_dynamic_pass,
                    "n_test_failure": stat.n_test_failure,
                    "n_execution_error": stat.n_execution_error,
                    "n_runner_error": stat.n_runner_error,
                    "dynamic_pass_rate": f"{stat.dynamic_pass_rate:.2f}",
                    "n_static_pass": stat.n_static_pass,
                    "n_static_fail": stat.n_static_fail,
                    "n_static_pass_dynamic_pass": stat.joint["static_pass_dynamic_pass"],
                    "n_static_pass_dynamic_nonpass": stat.joint[
                        "static_pass_dynamic_nonpass"
                    ],
                    "n_static_fail_dynamic_pass": stat.joint["static_fail_dynamic_pass"],
                    "n_static_fail_dynamic_nonpass": stat.joint[
                        "static_fail_dynamic_nonpass"
                    ],
                    "pct_static_pass_dynamic_pass": f"{stat.joint_pct('static_pass_dynamic_pass'):.2f}",
                    "pct_static_pass_dynamic_nonpass": f"{stat.joint_pct('static_pass_dynamic_nonpass'):.2f}",
                    "pct_static_fail_dynamic_pass": f"{stat.joint_pct('static_fail_dynamic_pass'):.2f}",
                    "pct_static_fail_dynamic_nonpass": f"{stat.joint_pct('static_fail_dynamic_nonpass'):.2f}",
                    "n_static_compatible": stat.n_static_pass,
                    "n_dynamic_pass_among_static_compatible": stat.joint[
                        "static_pass_dynamic_pass"
                    ],
                    "conditional_dynamic_pass_rate": (
                        "" if cond_rate is None else f"{cond_rate:.2f}"
                    ),
                    "n_execution_error_static_incompatible": (
                        stat.n_execution_error_static_incompatible
                    ),
                    "pct_execution_error_static_incompatible": (
                        "" if err_share is None else f"{err_share:.2f}"
                    ),
                }
            )
    return records


def run_assertions(
    selected: list[dict],
    by_condition: dict[str, list[RowStat]],
    input_sha: str,
) -> list[Assertion]:
    assertions: list[Assertion] = []
    assertions.append(
        Assertion(
            f"input SHA-256 equals {EXPECTED_INPUT_SHA256}",
            input_sha == EXPECTED_INPUT_SHA256,
            f"sha256={input_sha}",
        )
    )

    python_versions = {str(row["python_version"]) for row in selected}
    assertions.append(
        Assertion(
            "filtered records are Python 3.12",
            python_versions == {PYTHON_VERSION},
            f"python_version values={sorted(python_versions)}",
        )
    )
    tracks = {str(row["track"]) for row in selected}
    assertions.append(
        Assertion(
            'filtered records have track == "d1"',
            tracks == {TRACK},
            f"track values={sorted(tracks)}",
        )
    )

    keys = [(row["track"], row["mode"], row["model"], row["task_id"]) for row in selected]
    assertions.append(
        Assertion(
            "(track, mode, model, task_id) unique in the filtered set",
            len(keys) == len(set(keys)),
            f"n={len(keys)} unique={len(set(keys))}",
        )
    )

    statuses = {str(row["bcb_status"]) for row in selected}
    assertions.append(
        Assertion(
            "bcb_status values are pass, fail, error, or runner_error",
            statuses <= ALLOWED_BCB,
            f"bcb_status values={sorted(statuses)}",
        )
    )
    ty_values = {row["ty_compatible"] for row in selected}
    assertions.append(
        Assertion(
            "ty_compatible values are JSON booleans",
            ty_values <= {True, False},
            f"ty_compatible values={sorted(ty_values, key=str)}",
        )
    )

    va_lower: list[str] = []
    va_higher_or_equal: list[str] = []
    for condition in CONDITIONS:
        rows = [stat for stat in by_condition[condition] if not stat.is_aggregate]
        aggregate = by_condition[condition][-1]
        assertions.append(
            Assertion(
                f"{condition} contains exactly {N_MODELS} models",
                len(rows) == N_MODELS,
                f"n_models={len(rows)}",
            )
        )
        assertions.append(
            Assertion(
                f"{condition} contains exactly {N_OBS:,} observations",
                aggregate.n_observations == N_OBS,
                f"n={aggregate.n_observations:,}",
            )
        )
        for stat in rows:
            assertions.append(
                Assertion(
                    f"{condition} {stat.model_display} contains exactly {N_TASKS} observations",
                    stat.n_observations == N_TASKS,
                    f"n={stat.n_observations}",
                )
            )
        expected = EXPECTED_DYNAMIC[condition]
        assertions.append(
            Assertion(
                f"{condition} total equals {expected['total']:,}",
                aggregate.n_observations == expected["total"],
                f"total={aggregate.n_observations:,}",
            )
        )
        assertions.append(
            Assertion(
                f"{condition} dynamic pass equals {expected['pass']:,} ({expected['pass_pct']:.2f}%)",
                aggregate.n_dynamic_pass == expected["pass"]
                and round_pct(aggregate.dynamic_pass_rate) == expected["pass_pct"],
                f"pass={aggregate.n_dynamic_pass:,} ({aggregate.dynamic_pass_rate:.2f}%)",
            )
        )
        assertions.append(
            Assertion(
                f"{condition} test failure (bcb_status=fail) equals {expected['fail']:,}",
                aggregate.n_test_failure == expected["fail"],
                f"fail={aggregate.n_test_failure:,}",
            )
        )
        assertions.append(
            Assertion(
                f"{condition} execution error equals {expected['error']:,} ({expected['error_pct']:.2f}%)",
                aggregate.n_execution_error == expected["error"]
                and round_pct(100.0 * aggregate.n_execution_error / aggregate.n_observations)
                == expected["error_pct"],
                f"error={aggregate.n_execution_error:,} "
                f"({100.0 * aggregate.n_execution_error / aggregate.n_observations:.2f}%)",
            )
        )
        assertions.append(
            Assertion(
                f"{condition} runner_error equals {expected['runner_error']:,}",
                aggregate.n_runner_error == expected["runner_error"],
                f"runner_error={aggregate.n_runner_error:,}",
            )
        )
        joint_sum = sum(aggregate.joint.values())
        assertions.append(
            Assertion(
                f"{condition} four joint counts sum to {N_OBS:,}",
                joint_sum == N_OBS,
                f"sum={joint_sum:,}",
            )
        )
        for key in JOINT_KEYS:
            got_n = aggregate.joint[key]
            got_pct = round_pct(aggregate.joint_pct(key))
            exp_n, exp_pct = EXPECTED_JOINT[condition][key]
            assertions.append(
                Assertion(
                    f"{condition} {JOINT_DISPLAY[key]} equals {exp_n:,} ({exp_pct:.2f}%)",
                    got_n == exp_n and got_pct == exp_pct,
                    f"count={got_n:,} ({got_pct:.2f}%)",
                )
            )
        static_n, pass_n, cond_pct = EXPECTED_CONDITIONAL[condition]
        got_cond = aggregate.conditional_dynamic_pass_rate
        assertions.append(
            Assertion(
                f"{condition} statically compatible equals {static_n:,}",
                aggregate.n_static_pass == static_n,
                f"static_pass={aggregate.n_static_pass:,}",
            )
        )
        assertions.append(
            Assertion(
                f"{condition} dynamic pass among statically compatible equals {pass_n:,} ({cond_pct:.2f}%)",
                aggregate.joint["static_pass_dynamic_pass"] == pass_n
                and got_cond is not None
                and round_pct(got_cond) == cond_pct,
                f"conditional={aggregate.joint['static_pass_dynamic_pass']:,}/"
                f"{aggregate.n_static_pass:,} "
                f"({0.0 if got_cond is None else got_cond:.2f}%)",
            )
        )
        err_total, err_static, err_pct = EXPECTED_ERROR_STATIC_FAIL[condition]
        got_err_share = aggregate.pct_execution_error_static_incompatible
        assertions.append(
            Assertion(
                f"{condition} execution errors among statically incompatible observations "
                f"equal {err_static:,}/{err_total:,} ({err_pct:.2f}%)",
                aggregate.n_execution_error == err_total
                and aggregate.n_execution_error_static_incompatible == err_static
                and got_err_share is not None
                and round_pct(got_err_share) == err_pct,
                f"share={aggregate.n_execution_error_static_incompatible:,}/"
                f"{aggregate.n_execution_error:,} "
                f"({0.0 if got_err_share is None else got_err_share:.2f}%)",
            )
        )
        rates = [round_pct(stat.dynamic_pass_rate) for stat in rows]
        exp_min, exp_max = EXPECTED_MODEL_RANGE[condition]
        assertions.append(
            Assertion(
                f"{condition} model pass-rate range equals {exp_min:.2f}% to {exp_max:.2f}%",
                min(rates) == exp_min and max(rates) == exp_max,
                f"range={min(rates):.2f}% to {max(rates):.2f}%",
            )
        )

    mr_by_name = {
        stat.model_display: stat
        for stat in by_condition["MR"]
        if not stat.is_aggregate
    }
    va_by_name = {
        stat.model_display: stat
        for stat in by_condition["VA"]
        if not stat.is_aggregate
    }
    for name in DISPLAY_ORDER:
        mr_rate = mr_by_name[name].dynamic_pass_rate
        va_rate = va_by_name[name].dynamic_pass_rate
        if va_rate < mr_rate:
            va_lower.append(name)
        else:
            va_higher_or_equal.append(name)
    assertions.append(
        Assertion(
            "VA pass rate is lower for 9 of 10 models",
            len(va_lower) == EXPECTED_VA_LOWER,
            "VA lower=" + ", ".join(va_lower) + f" ({len(va_lower)})",
        )
    )
    assertions.append(
        Assertion(
            f"{EXPECTED_EXCEPTION} is the sole exception",
            va_higher_or_equal == [EXPECTED_EXCEPTION],
            "exception=" + ", ".join(va_higher_or_equal) if va_higher_or_equal else "none",
        )
    )
    for condition in CONDITIONS:
        pass_n, n_obs, pct = EXPECTED_GPT[condition]
        stat = mr_by_name[EXPECTED_EXCEPTION] if condition == "MR" else va_by_name[EXPECTED_EXCEPTION]
        assertions.append(
            Assertion(
                f"{EXPECTED_EXCEPTION} {condition} pass equals {pass_n}/{n_obs} ({pct:.2f}%)",
                stat.n_dynamic_pass == pass_n
                and stat.n_observations == n_obs
                and round_pct(stat.dynamic_pass_rate) == pct,
                f"pass={stat.n_dynamic_pass}/{stat.n_observations} ({stat.dynamic_pass_rate:.2f}%)",
            )
        )

    mr_agg = by_condition["MR"][-1]
    va_agg = by_condition["VA"][-1]
    diff = round_pct(mr_agg.dynamic_pass_rate) - round_pct(va_agg.dynamic_pass_rate)
    assertions.append(
        Assertion(
            f"aggregate pass-rate difference equals {EXPECTED_PASS_DIFF_PP:.2f} percentage points",
            round_pct(diff) == EXPECTED_PASS_DIFF_PP,
            f"difference={diff:.2f} pp",
        )
    )
    return assertions


def label_color(hex_color: str) -> str:
    red = int(hex_color[1:3], 16) / 255.0
    green = int(hex_color[3:5], 16) / 255.0
    blue = int(hex_color[5:7], 16) / 255.0

    def linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * linearize(red)
        + 0.7152 * linearize(green)
        + 0.0722 * linearize(blue)
    )
    contrast_white = 1.05 / (luminance + 0.05)
    contrast_black = (luminance + 0.05) / 0.05
    return "#FFFFFF" if contrast_white >= contrast_black else TEXT_MAIN


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 9,
        }
    )


def style_axes(ax: plt.Axes, x_max: float) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_bounds(0, x_max)
    ax.tick_params(axis="x", length=3.2, color=SPINE_COLOR, labelsize=8.0, pad=1.6, colors=TEXT_MAIN)
    ax.tick_params(axis="y", length=0, pad=3.6, labelsize=8.6, colors=TEXT_MAIN)


def _rate_label(ax: plt.Axes, x: float, y: float, value: float, color: str, **kwargs) -> None:
    ax.annotate(
        f"{value:.1f}%",
        xy=(x, y),
        textcoords="offset points",
        fontsize=6.3,
        color=color,
        clip_on=False,
        zorder=6,
        **kwargs,
    )


def plot_pass_rates(ax: plt.Axes, by_condition: dict[str, list[RowStat]], y: np.ndarray) -> None:
    mr_rows = by_condition["MR"]
    va_rows = by_condition["VA"]
    mr_rates = np.array([stat.dynamic_pass_rate for stat in mr_rows], dtype=float)
    va_rates = np.array([stat.dynamic_pass_rate for stat in va_rows], dtype=float)

    for yi, mr, va in zip(y, mr_rates, va_rates):
        ax.plot(
            [mr, va],
            [yi, yi],
            color=CONNECTOR_COLOR,
            linewidth=0.9,
            zorder=2,
            solid_capstyle="round",
        )
    ax.scatter(
        mr_rates,
        y,
        s=36,
        color=CONDITION_COLORS["MR"],
        edgecolors="#FFFFFF",
        linewidths=0.55,
        zorder=4,
        clip_on=False,
    )
    ax.scatter(
        va_rates,
        y,
        s=36,
        color=CONDITION_COLORS["VA"],
        edgecolors="#FFFFFF",
        linewidths=0.55,
        zorder=4,
        clip_on=False,
    )
    ax.scatter(
        [mr_rates[-1]],
        [y[-1]],
        s=46,
        facecolors=CONDITION_COLORS["MR"],
        edgecolors=AGG_EDGE,
        linewidths=0.9,
        zorder=5,
        clip_on=False,
    )
    ax.scatter(
        [va_rates[-1]],
        [y[-1]],
        s=46,
        facecolors=CONDITION_COLORS["VA"],
        edgecolors=AGG_EDGE,
        linewidths=0.9,
        zorder=5,
        clip_on=False,
    )

    for yi, mr, va, is_agg in zip(y, mr_rates, va_rates, [stat.is_aggregate for stat in mr_rows]):
        weight = "bold" if is_agg else "normal"
        left_x, left_val, left_color = (va, va, CONDITION_COLORS["VA"])
        right_x, right_val, right_color = (mr, mr, CONDITION_COLORS["MR"])
        if mr < va:
            left_x, left_val, left_color = (mr, mr, CONDITION_COLORS["MR"])
            right_x, right_val, right_color = (va, va, CONDITION_COLORS["VA"])
        # Place labels on the outer sides. Near-overlapping points (GPT-5.4)
        # need a larger offset so the two percentages do not collide.
        dx = 11.0 if abs(mr - va) < 3.0 else 5.5
        _rate_label(
            ax,
            left_x,
            yi,
            left_val,
            left_color,
            xytext=(-dx, 0.0),
            ha="right",
            va="center",
            fontweight=weight,
        )
        _rate_label(
            ax,
            right_x,
            yi,
            right_val,
            right_color,
            xytext=(dx, 0.0),
            ha="left",
            va="center",
            fontweight=weight,
        )

    separator_y = (y[-2] + y[-1]) / 2.0
    ax.axhline(separator_y, color=GUIDE_COLOR, linewidth=0.85, zorder=1)
    ax.set_xlim(0, 62)
    ax.set_xticks((0, 15, 30, 45, 60))
    ax.set_xticklabels(("0%", "15%", "30%", "45%", "60%"))
    ax.set_xlabel("Dynamic pass rate", fontsize=8.6, color=TEXT_MAIN, labelpad=3)
    ax.set_title("(a) Dynamic pass rate", loc="center", fontsize=10.2, pad=6, color=TEXT_MAIN)
    style_axes(ax, 60)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_bounds(y[0] - 0.32, y[-1] + 0.32)


def plot_joint(ax: plt.Axes, by_condition: dict[str, list[RowStat]]) -> None:
    stats = [by_condition[condition][-1] for condition in CONDITIONS]
    y = np.array([1.05, 0.0])
    bar_height = 0.72
    left = np.zeros(2, dtype=float)
    small_labels: list[tuple[float, float, float, str, str]] = []

    for key in JOINT_KEYS:
        widths = np.array([stat.joint_pct(key) for stat in stats], dtype=float)
        ax.barh(
            y,
            widths,
            left=left,
            height=bar_height,
            color=JOINT_COLORS[key],
            edgecolor="#FFFFFF",
            linewidth=0.55,
            zorder=3,
        )
        for yi, width, start in zip(y, widths, left):
            if width >= MIN_LABEL_PCT:
                ax.text(
                    start + width / 2.0,
                    yi,
                    f"{width:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color=label_color(JOINT_COLORS[key]),
                    clip_on=True,
                    zorder=4,
                )
            elif key == "static_fail_dynamic_pass" and width > 0:
                small_labels.append(
                    (
                        start + width / 2.0,
                        yi,
                        yi + bar_height / 2.0,
                        f"{width:.1f}%",
                        JOINT_COLORS[key],
                    )
                )
        left = left + widths
    if not np.allclose(left, 100.0, atol=1e-9):
        raise SystemExit(f"joint stacked shares are not 100%: {left.tolist()}")

    for x, yi, y_top, text, color in small_labels:
        ax.plot(
            [x, x],
            [y_top, y_top + 0.07],
            color=color,
            linewidth=0.7,
            zorder=5,
            clip_on=False,
        )
        ax.text(
            x,
            y_top + 0.09,
            text,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=color,
            zorder=5,
        )

    ax.set_xlim(0, 100)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"))
    ax.set_xlabel("Share of observations", fontsize=8.6, color=TEXT_MAIN, labelpad=3)
    ax.set_yticks(y)
    ax.set_yticklabels(list(CONDITIONS), fontsize=9.2)
    ax.set_ylim(-0.52, 2.12)
    ax.set_title(
        "(b) Joint static and dynamic outcomes",
        loc="center",
        fontsize=10.2,
        pad=6,
        color=TEXT_MAIN,
    )
    style_axes(ax, 100)


def plot_figure(
    by_condition: dict[str, list[RowStat]],
    pdf_path: Path,
    png_path: Path,
    dpi: int,
) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    apply_style()
    row_labels = [stat.model_display for stat in by_condition["MR"]]
    n_rows = len(row_labels)
    y = np.arange(n_rows, dtype=float)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.72),
        gridspec_kw={"width_ratios": [1.22, 1.0], "wspace": 0.26},
    )
    fig.patch.set_facecolor("#FFFFFF")

    plot_pass_rates(ax_a, by_condition, y)
    pad = 0.32
    ax_a.set_ylim(-pad, float(n_rows - 1) + pad)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(row_labels, fontsize=8.6)
    ax_a.invert_yaxis()
    for tick, name in zip(ax_a.get_yticklabels(), row_labels):
        if name == AGGREGATE_LABEL:
            tick.set_fontweight("bold")
            tick.set_color(TEXT_MAIN)

    plot_joint(ax_b, by_condition)

    handles_a = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=6.2,
            markerfacecolor=CONDITION_COLORS[condition],
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.5,
            label=condition,
        )
        for condition in CONDITIONS
    ]
    ax_a.legend(
        handles=handles_a,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#E4E4E4",
        facecolor="#FFFFFF",
        framealpha=0.96,
        fontsize=7.4,
        handletextpad=0.35,
        handlelength=1.15,
        labelspacing=0.28,
        borderpad=0.40,
        borderaxespad=0.25,
    )
    handles_b = [
        Patch(
            facecolor=JOINT_COLORS[key],
            edgecolor="#FFFFFF",
            linewidth=0.5,
            label=JOINT_DISPLAY[key],
        )
        for key in JOINT_KEYS
    ]
    ax_b.legend(
        handles=handles_b,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#E4E4E4",
        facecolor="#FFFFFF",
        framealpha=0.96,
        fontsize=6.4,
        ncol=1,
        handlelength=1.05,
        handleheight=0.55,
        handletextpad=0.40,
        labelspacing=0.28,
        borderpad=0.40,
        borderaxespad=0.15,
    )

    fig.subplots_adjust(left=0.168, right=0.985, top=0.90, bottom=0.11, wspace=0.26)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def copy_manuscript_pdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def model_mapping_lines() -> list[str]:
    lines = []
    for raw, display in MODEL_LABEL.items():
        lines.append(f"- `{raw}` -> `{display}`")
    lines.append(
        "- `meta-llama/llama-4-scout` is accepted through `paper_display_label` and displayed as `Llama-4-Scout`"
    )
    lines.append(
        "- D1 records spell DeepSeek as `DeepSeek-V3.2` under MR and `deepseek-v3.2` under VA; both map to `DeepSeek-V3.2`"
    )
    lines.append(
        "- D1 records spell Llama as `meta-llama_llama-4-scout` or `meta-llama/llama-4-scout`; both map to `Llama-4-Scout`"
    )
    return lines


def render_report(
    input_path: Path,
    input_sha: str,
    selected: list[dict],
    by_condition: dict[str, list[RowStat]],
    assertions: list[Assertion],
    output_paths: dict[str, Path],
    output_sha: dict[str, str],
    reproduce_cmd: str,
    png_meta: dict[str, float | int],
    pdf_meta: dict[str, object],
) -> str:
    lines = [
        "# RQ3 dynamic-execution outcomes verification report",
        "",
        "## Input",
        "",
        f"- path: `{display_path(input_path)}`",
        f"- SHA-256: `{input_sha}`",
        f"- expected SHA-256: `{EXPECTED_INPUT_SHA256}`",
        f"- loaded records: {len(selected):,} after filters; full file is hashed above",
        "",
        "## Filters",
        "",
        "- source: `task_outcomes.jsonl` from the corrected canonical tables",
        "- do not use the obsolete 724-task OSV subset",
        "- do not use any former BigCodeBench summary table",
        "- `track == \"d1\"`",
        '- `python_version == "3.12"`',
        "- no filter on OSV coverage, resolver status, compatibility layer, constraint presence, or static-validation outcome",
        "- analysis unit: one task--model observation",
        "- uniqueness key: `(track, mode, model, task_id)`",
        "",
        "## Status definitions",
        "",
        "- dynamic pass: `bcb_status == \"pass\"`",
        "- dynamic non-pass: `bcb_status in {\"fail\", \"error\", \"runner_error\"}`",
        '- `bcb_status == "fail"` is reported as test failure; the runner combines unittest failures and unittest errors under this status',
        "- `bcb_status == \"error\"` is an execution error",
        "- `bcb_status == \"runner_error\"` is a runner error",
        "- static pass: `ty_compatible is true`",
        "- static fail: `ty_compatible is false`",
        "- missing or unexpected `ty_compatible` or `bcb_status` values abort the script",
        "",
        "## Mode mapping",
        "",
        "- `requirements.txt` -> `MR`",
        "- `inline` -> `VA`",
        "",
        "## Model-name mapping",
        "",
        "Display names and order come from `plots.model_display.MODEL_LABEL` via `paper_display_label`.",
        "The order matches `plots/rq3_failure_layers.py`.",
        "No new aliases are introduced.",
        "",
    ]
    lines.extend(model_mapping_lines())
    lines.extend(
        [
            "",
            f"- figure order: {', '.join(DISPLAY_ORDER)}, then `{AGGREGATE_LABEL}`",
            f"- `{AGGREGATE_LABEL}` sums raw counts across models before computing rates",
            "",
            "## Per-model dynamic outcomes",
            "",
        ]
    )
    for condition in CONDITIONS:
        lines.append(f"### {condition}")
        lines.append("")
        for stat in by_condition[condition]:
            lines.append(
                f"- {stat.model_display} (`{stat.model_raw}`): "
                f"n={stat.n_observations:,}; "
                f"pass={stat.n_dynamic_pass:,} ({stat.dynamic_pass_rate:.2f}%); "
                f"test failure={stat.n_test_failure:,}; "
                f"execution error={stat.n_execution_error:,}; "
                f"runner error={stat.n_runner_error:,}"
            )
        lines.append("")

    lines.extend(["## Aggregate dynamic outcomes", ""])
    for condition in CONDITIONS:
        aggregate = by_condition[condition][-1]
        error_pct = 100.0 * aggregate.n_execution_error / aggregate.n_observations
        lines.append(f"### {condition}")
        lines.append("")
        lines.append(f"- total: {aggregate.n_observations:,}")
        lines.append(
            f"- pass: {aggregate.n_dynamic_pass:,} ({aggregate.dynamic_pass_rate:.2f}%)"
        )
        lines.append(f"- test failure (`fail`): {aggregate.n_test_failure:,}")
        lines.append(
            f"- execution error (`error`): {aggregate.n_execution_error:,} ({error_pct:.2f}%)"
        )
        lines.append(f"- runner error (`runner_error`): {aggregate.n_runner_error:,}")
        lines.append("")

    mr_rate = by_condition["MR"][-1].dynamic_pass_rate
    va_rate = by_condition["VA"][-1].dynamic_pass_rate
    lines.extend(
        [
            f"- aggregate pass-rate difference: {mr_rate - va_rate:.2f} percentage points",
            "",
            "## Joint static and dynamic outcomes",
            "",
        ]
    )
    for condition in CONDITIONS:
        aggregate = by_condition[condition][-1]
        lines.append(f"### {condition}")
        lines.append("")
        for key in JOINT_KEYS:
            lines.append(
                f"- {JOINT_DISPLAY[key]}: {aggregate.joint[key]:,} "
                f"({aggregate.joint_pct(key):.2f}%)"
            )
        lines.append(
            f"- four-category sum: {sum(aggregate.joint.values()):,}"
        )
        lines.append("")

    lines.extend(["## Conditional dynamic success among statically compatible observations", ""])
    lines.append("These rates are reported but are not the denominator in Panel (a).")
    lines.append("")
    for condition in CONDITIONS:
        aggregate = by_condition[condition][-1]
        cond_rate = aggregate.conditional_dynamic_pass_rate
        lines.append(
            f"- {condition}: statically compatible={aggregate.n_static_pass:,}; "
            f"dynamic pass among them={aggregate.joint['static_pass_dynamic_pass']:,}; "
            f"conditional rate={0.0 if cond_rate is None else cond_rate:.2f}%"
        )
    lines.extend(
        [
            "",
            "## Execution errors and static incompatibility",
            "",
        ]
    )
    for condition in CONDITIONS:
        aggregate = by_condition[condition][-1]
        share = aggregate.pct_execution_error_static_incompatible
        lines.append(
            f"- {condition}: execution errors={aggregate.n_execution_error:,}; "
            f"statically incompatible execution errors="
            f"{aggregate.n_execution_error_static_incompatible:,}; "
            f"share={0.0 if share is None else share:.2f}%"
        )

    passed = sum(item.passed for item in assertions)
    lines.extend(
        [
            "",
            "## Assertions",
            "",
            f"- {passed}/{len(assertions)} passed",
            "",
        ]
    )
    for item in assertions:
        status = "pass" if item.passed else "FAIL"
        lines.append(f"- {item.name}: {status} ({item.detail})")

    lines.extend(
        [
            "",
            "## Software versions",
            "",
            f"- Python: `{sys.version.replace(chr(10), ' ')}`",
            f"- pandas: `{pd.__version__}`",
            f"- matplotlib: `{matplotlib.__version__}`",
            "",
            "## Reproduction command",
            "",
            "From the repository root:",
            "",
            f"`{reproduce_cmd}`",
            "",
            "## Output geometry and fonts",
            "",
            f"- PNG: {png_meta.get('width_px')} x {png_meta.get('height_px')} px, "
            f"{png_meta.get('dpi_x')} dpi, {png_meta.get('width_in')} in x {png_meta.get('height_in')} in",
            f"- PDF MediaBox: {pdf_meta.get('width_in')} in x {pdf_meta.get('height_in')} in "
            f"({pdf_meta.get('width_pt')} pt x {pdf_meta.get('height_pt')} pt)",
            f"- embedded fonts: {', '.join(pdf_meta.get('fonts') or [])}",
            f"- font subtypes: {', '.join(pdf_meta.get('subtypes') or [])}",
            "- `pdfinfo` and `pdffonts` were not installed on this host; page size and embedded fonts were read directly from the PDF",
            "",
            "## Outputs",
            "",
        ]
    )
    for label, path in output_paths.items():
        lines.append(f"- {label}: `{display_path(path)}`")
        if label in output_sha:
            lines.append(f"  - SHA-256: `{output_sha[label]}`")
        else:
            lines.append("  - SHA-256 is computed after this file is written and printed to stdout")
    lines.extend(
        [
            "",
            "## Manuscript files",
            "",
            "Manuscript source files were not modified.  This script writes the analysis CSV, PDF, PNG, and this report under `plots/`, and copies the PDF to `LLM_TPL_20260507_for_revision/`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "rq3_dynamic_outcomes.csv"
    pdf_path = output_dir / "rq3_dynamic_outcomes.pdf"
    png_path = output_dir / "rq3_dynamic_outcomes.png"
    report_path = output_dir / "rq3_dynamic_outcomes_report.md"
    manuscript_pdf = args.manuscript_pdf.resolve()
    reproduce_cmd = ".venv/bin/python -m plots.rq3_dynamic_outcomes"

    input_sha = sha256_file(input_path)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            f"input SHA-256 {input_sha} does not match {EXPECTED_INPUT_SHA256}"
        )

    rows = load_jsonl(input_path)
    selected = select_observations(rows)
    by_condition = build_stats(selected)
    assertions = run_assertions(selected, by_condition, input_sha)
    failed = [item for item in assertions if not item.passed]
    if failed:
        details = "\n".join(f"- FAIL {item.name}: {item.detail}" for item in failed)
        raise SystemExit(
            "Assertions failed; not writing a figure from a disputed population.\n"
            + details
        )

    records = csv_records(by_condition)
    expected_csv_rows = len(CONDITIONS) * (N_MODELS + 1)
    if len(records) != expected_csv_rows:
        raise SystemExit(f"CSV rows={len(records)} expected {expected_csv_rows}")

    plot_figure(by_condition, pdf_path, png_path, args.dpi)
    write_csv(csv_path, records)
    copy_manuscript_pdf(pdf_path, manuscript_pdf)
    hashed_outputs = {
        "csv": csv_path,
        "pdf": pdf_path,
        "png": png_path,
        "manuscript_pdf": manuscript_pdf,
    }
    output_paths = {**hashed_outputs, "report": report_path}
    for path in hashed_outputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"output missing or empty: {path}")

    png_meta = png_geometry(png_path)
    pdf_meta = pdf_geometry(pdf_path)
    report = render_report(
        input_path=input_path,
        input_sha=input_sha,
        selected=selected,
        by_condition=by_condition,
        assertions=assertions,
        output_paths=output_paths,
        output_sha={label: sha256_file(path) for label, path in hashed_outputs.items()},
        reproduce_cmd=reproduce_cmd,
        png_meta=png_meta,
        pdf_meta=pdf_meta,
    )
    report_path.write_text(report, encoding="utf-8")
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise SystemExit(f"output missing or empty: {report_path}")

    print(report)
    print(f"Saved: {csv_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {manuscript_pdf}")
    print(f"Report SHA-256: {sha256_file(report_path)}")


if __name__ == "__main__":
    main()
