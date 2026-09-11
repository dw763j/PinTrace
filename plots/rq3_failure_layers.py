"""RQ3 failure-layer composition among PinTrace observations that supply resolver input but fail static validation.

The figure is failure-only.  It excludes compatible observations, empty-resolver
(no-dependency) observations, BigCodeBench dynamic results, and CVE / version-age
results.  Stage pass rates belong to the preceding RQ3 subsubsection.

Population:
    track == "d2"
    resolver_status != "empty"
    compatibility_layer != "compatible"

Usage (from project root)::

    .venv/bin/python -m plots.rq3_failure_layers
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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from plots.model_display import MODEL_LABEL, paper_display_label
from paths import EVALUATION, display_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EVALUATION / "task_outcomes.jsonl"
DEFAULT_OUTPUT_DIR = PLOTS_DIR

TRACK = "d2"
PYTHON_VERSION = "3.12"
MODE_TO_CONDITION = {"requirements.txt": "MR", "inline": "VA"}
CONDITIONS = ("MR", "VA")
AGGREGATE_LABEL = "All observations"
AGGREGATE_RAW = "ALL"
N_MODELS = 10
MIN_LABEL_PCT = 8.0

LAYER_KEYS = (
    "environment_build",
    "api_mismatch_candidate",
    "import_or_initialization",
    "code_or_syntax",
)
LAYER_DISPLAY = {
    "environment_build": "Environment construction",
    "api_mismatch_candidate": "API-mismatch candidate",
    "import_or_initialization": "Import/initialization",
    "code_or_syntax": "Code/syntax",
}
LAYER_COLORS = {
    "environment_build": "#D55E00",
    "api_mismatch_candidate": "#E69F00",
    "import_or_initialization": "#CC79A7",
    "code_or_syntax": "#0072B2",
}
ALLOWED_LAYERS = frozenset(LAYER_KEYS)
DISPLAY_ORDER = tuple(MODEL_LABEL.values())
REQUIRED_FIELDS = (
    "track",
    "mode",
    "model",
    "task_id",
    "resolver_status",
    "compatibility_layer",
    "python_version",
)
CSV_FIELDS = (
    "condition",
    "model_raw",
    "model_display",
    "is_aggregate",
    "failure_denominator",
    "layer",
    "numerator",
    "percentage",
)

EXPECTED_TOTALS = {"MR": 4_191, "VA": 5_436}
EXPECTED_LAYER_COUNTS = {
    "MR": {
        "environment_build": 1_456,
        "api_mismatch_candidate": 1_558,
        "import_or_initialization": 645,
        "code_or_syntax": 532,
    },
    "VA": {
        "environment_build": 3_594,
        "api_mismatch_candidate": 1_068,
        "import_or_initialization": 385,
        "code_or_syntax": 389,
    },
}
EXPECTED_LAYER_PCTS = {
    "MR": {
        "environment_build": 34.74,
        "api_mismatch_candidate": 37.17,
        "import_or_initialization": 15.39,
        "code_or_syntax": 12.69,
    },
    "VA": {
        "environment_build": 66.11,
        "api_mismatch_candidate": 19.65,
        "import_or_initialization": 7.08,
        "code_or_syntax": 7.16,
    },
}
EXPECTED_VA_ENV_MIN = ("GPT-5.4", 42.86)
EXPECTED_VA_ENV_MAX = ("Qwen3.5-397B", 78.66)
EXPECTED_MR_API_DOMINANT = 7
EXPECTED_MR_ENV_DOMINANT = 3

TEXT_MAIN = "#222222"
TEXT_MUTED = "#5A5A5A"
GUIDE_COLOR = "#C8C8C8"
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
    counts: dict[str, int]

    @property
    def failure_denominator(self) -> int:
        return sum(self.counts[layer] for layer in LAYER_KEYS)

    def numerator(self, layer: str) -> int:
        return self.counts[layer]

    def percentage(self, layer: str) -> float:
        total = self.failure_denominator
        if total == 0:
            raise ValueError(
                f"zero failure denominator for {self.condition} {self.model_display}"
            )
        return 100.0 * self.counts[layer] / total

    def dominant_layer(self) -> str:
        ranked = sorted(
            LAYER_KEYS,
            key=lambda layer: (-self.counts[layer], LAYER_KEYS.index(layer)),
        )
        top = ranked[0]
        tied = [layer for layer in LAYER_KEYS if self.counts[layer] == self.counts[top]]
        if len(tied) > 1:
            raise ValueError(
                f"{self.condition} {self.model_display} has a tie for the largest "
                f"failure layer among {[LAYER_DISPLAY[layer] for layer in tied]} "
                f"with count {self.counts[top]}"
            )
        return top


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        pixels_per_unit_x, pixels_per_unit_y, unit = struct.unpack(">IIB", payload[marker + 4 : marker + 13])
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
            re.findall(rb"/Subtype\s*/(Type1|TrueType|Type0|CIDFontType0|CIDFontType2)", payload)
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


def format_unexpected(rows: list[dict]) -> str:
    lines = [
        "Unexpected compatibility_layer values after the failure-only filter "
        f"({len(rows)} records).  Refusing to drop them silently."
    ]
    preview = rows[:30]
    for row in preview:
        lines.append(
            "  mode={mode!r} model={model!r} task_id={task_id!r} "
            "resolver_status={resolver_status!r} compatibility_layer={compatibility_layer!r} "
            "python_version={python_version!r}".format(
                mode=row.get("mode"),
                model=row.get("model"),
                task_id=row.get("task_id"),
                resolver_status=row.get("resolver_status"),
                compatibility_layer=row.get("compatibility_layer"),
                python_version=row.get("python_version"),
            )
        )
    if len(rows) > len(preview):
        lines.append(f"  ... {len(rows) - len(preview)} more records omitted")
    return "\n".join(lines)


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


def select_failure_observations(rows: list[dict]) -> list[dict]:
    seen: set[tuple[object, object, object, object]] = set()
    selected: list[dict] = []
    unexpected: list[dict] = []
    unknown_modes: list[str] = []

    for index, row in enumerate(rows):
        missing_fields(row, index)
        key = (row["track"], row["mode"], row["model"], row["task_id"])
        if key in seen:
            raise ValueError(f"duplicate observation key {key}")
        seen.add(key)
        if row["track"] != TRACK:
            continue
        mode = row["mode"]
        if mode not in MODE_TO_CONDITION:
            unknown_modes.append(
                f"mode={mode!r} model={row['model']!r} task_id={row['task_id']!r}"
            )
            continue
        if row["resolver_status"] == "empty":
            continue
        layer = row["compatibility_layer"]
        if layer == "compatible":
            continue
        if layer not in ALLOWED_LAYERS:
            unexpected.append(row)
            continue
        selected.append(row)

    if unknown_modes:
        preview = "\n  ".join(unknown_modes[:20])
        raise ValueError(f"unknown request modes:\n  {preview}")
    if unexpected:
        raise SystemExit(format_unexpected(unexpected))
    if not selected:
        raise ValueError("failure-only filter selected zero observations")
    return selected


def build_stats(selected: list[dict]) -> dict[str, list[RowStat]]:
    raw_by_display: dict[str, str] = {}
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for row in selected:
        model_raw = str(row["model"])
        display = display_for(model_raw)
        previous = raw_by_display.get(display)
        if previous is not None and previous != model_raw:
            raise ValueError(
                f"display name {display!r} maps from both {previous!r} and {model_raw!r}"
            )
        raw_by_display[display] = model_raw
        condition = MODE_TO_CONDITION[str(row["mode"])]
        counts[(condition, display)][str(row["compatibility_layer"])] += 1

    missing_models = [name for name in DISPLAY_ORDER if name not in raw_by_display]
    extra_models = sorted(set(raw_by_display) - set(DISPLAY_ORDER))
    if missing_models or extra_models:
        raise ValueError(
            f"model set mismatch; missing={missing_models} extra={extra_models}"
        )

    by_condition: dict[str, list[RowStat]] = {}
    for condition in CONDITIONS:
        model_rows: list[RowStat] = []
        aggregate_counts = {layer: 0 for layer in LAYER_KEYS}
        for display in DISPLAY_ORDER:
            layer_counts = {layer: int(counts[(condition, display)][layer]) for layer in LAYER_KEYS}
            for layer, value in layer_counts.items():
                aggregate_counts[layer] += value
            model_rows.append(
                RowStat(
                    condition=condition,
                    model_raw=raw_by_display[display],
                    model_display=display,
                    is_aggregate=False,
                    counts=layer_counts,
                )
            )
        model_rows.append(
            RowStat(
                condition=condition,
                model_raw=AGGREGATE_RAW,
                model_display=AGGREGATE_LABEL,
                is_aggregate=True,
                counts=aggregate_counts,
            )
        )
        by_condition[condition] = model_rows
    return by_condition


def csv_records(by_condition: dict[str, list[RowStat]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for stat in by_condition[condition]:
            total = stat.failure_denominator
            for layer in LAYER_KEYS:
                records.append(
                    {
                        "condition": condition,
                        "model_raw": stat.model_raw,
                        "model_display": stat.model_display,
                        "is_aggregate": "true" if stat.is_aggregate else "false",
                        "failure_denominator": total,
                        "layer": LAYER_DISPLAY[layer],
                        "numerator": stat.numerator(layer),
                        "percentage": f"{stat.percentage(layer):.2f}",
                    }
                )
    return records


def run_assertions(
    selected: list[dict],
    by_condition: dict[str, list[RowStat]],
) -> list[Assertion]:
    assertions: list[Assertion] = []

    python_versions = {str(row["python_version"]) for row in selected}
    assertions.append(
        Assertion(
            "filtered records are Python 3.12",
            python_versions == {PYTHON_VERSION},
            f"python_version values={sorted(python_versions)}",
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

    for condition in CONDITIONS:
        rows = [stat for stat in by_condition[condition] if not stat.is_aggregate]
        aggregate = by_condition[condition][-1]
        n_models = len(rows)
        assertions.append(
            Assertion(
                f"{condition} contains exactly {N_MODELS} models",
                n_models == N_MODELS,
                f"n_models={n_models}",
            )
        )
        total = aggregate.failure_denominator
        assertions.append(
            Assertion(
                f"{condition} failure total equals {EXPECTED_TOTALS[condition]:,}",
                total == EXPECTED_TOTALS[condition],
                f"total={total:,}",
            )
        )
        layer_sum = sum(aggregate.counts[layer] for layer in LAYER_KEYS)
        assertions.append(
            Assertion(
                f"{condition} four-layer sum equals {EXPECTED_TOTALS[condition]:,}",
                layer_sum == EXPECTED_TOTALS[condition],
                f"sum={layer_sum:,}",
            )
        )
        for layer in LAYER_KEYS:
            got = aggregate.counts[layer]
            expected = EXPECTED_LAYER_COUNTS[condition][layer]
            assertions.append(
                Assertion(
                    f"{condition} {LAYER_DISPLAY[layer]} count equals {expected:,}",
                    got == expected,
                    f"count={got:,}",
                )
            )
            got_pct = round_pct(aggregate.percentage(layer))
            expected_pct = EXPECTED_LAYER_PCTS[condition][layer]
            assertions.append(
                Assertion(
                    f"{condition} {LAYER_DISPLAY[layer]} share equals {expected_pct:.2f}%",
                    got_pct == expected_pct,
                    f"share={got_pct:.2f}%",
                )
            )
        for stat in by_condition[condition]:
            layer_sum = sum(stat.counts[layer] for layer in LAYER_KEYS)
            assertions.append(
                Assertion(
                    f"{condition} {stat.model_display} four-layer sum equals the model failure count",
                    layer_sum == stat.failure_denominator,
                    f"sum={layer_sum:,} n={stat.failure_denominator:,}",
                )
            )
            pct_sum = sum(stat.percentage(layer) for layer in LAYER_KEYS)
            assertions.append(
                Assertion(
                    f"{condition} {stat.model_display} unrounded shares sum to 100%",
                    abs(pct_sum - 100.0) <= 1e-9,
                    f"sum={pct_sum:.12f}%",
                )
            )

    va_models = [stat for stat in by_condition["VA"] if not stat.is_aggregate]
    va_env_dominant = [
        stat for stat in va_models if stat.dominant_layer() == "environment_build"
    ]
    assertions.append(
        Assertion(
            "VA: Environment construction is the largest layer in all 10 models",
            len(va_env_dominant) == N_MODELS,
            "dominant="
            + ", ".join(
                f"{stat.model_display}={LAYER_DISPLAY[stat.dominant_layer()]} "
                f"{stat.percentage(stat.dominant_layer()):.2f}%"
                for stat in va_models
            ),
        )
    )
    va_env_pcts = [(stat.model_display, round_pct(stat.percentage("environment_build"))) for stat in va_models]
    va_min_name, va_min_pct = min(va_env_pcts, key=lambda item: item[1])
    va_max_name, va_max_pct = max(va_env_pcts, key=lambda item: item[1])
    assertions.append(
        Assertion(
            f"VA Environment construction minimum is {EXPECTED_VA_ENV_MIN[0]} at {EXPECTED_VA_ENV_MIN[1]:.2f}%",
            (va_min_name, va_min_pct) == EXPECTED_VA_ENV_MIN,
            f"min={va_min_name} {va_min_pct:.2f}%",
        )
    )
    assertions.append(
        Assertion(
            f"VA Environment construction maximum is {EXPECTED_VA_ENV_MAX[0]} at {EXPECTED_VA_ENV_MAX[1]:.2f}%",
            (va_max_name, va_max_pct) == EXPECTED_VA_ENV_MAX,
            f"max={va_max_name} {va_max_pct:.2f}%",
        )
    )

    mr_models = [stat for stat in by_condition["MR"] if not stat.is_aggregate]
    mr_api = [stat for stat in mr_models if stat.dominant_layer() == "api_mismatch_candidate"]
    mr_env = [stat for stat in mr_models if stat.dominant_layer() == "environment_build"]
    assertions.append(
        Assertion(
            "MR: API-mismatch candidate is the largest layer in seven models",
            len(mr_api) == EXPECTED_MR_API_DOMINANT,
            "API-dominant=" + ", ".join(stat.model_display for stat in mr_api)
            + "; env-dominant="
            + ", ".join(stat.model_display for stat in mr_env)
            + "; other="
            + ", ".join(
                f"{stat.model_display}:{LAYER_DISPLAY[stat.dominant_layer()]}"
                for stat in mr_models
                if stat not in mr_api and stat not in mr_env
            ),
        )
    )
    assertions.append(
        Assertion(
            "MR: Environment construction is the largest layer in the remaining three models",
            len(mr_env) == EXPECTED_MR_ENV_DOMINANT
            and len(mr_api) + len(mr_env) == N_MODELS,
            "env-dominant=" + ", ".join(stat.model_display for stat in mr_env),
        )
    )
    return assertions


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.tick_params(axis="x", length=3.6, color=SPINE_COLOR, labelsize=11.0, pad=2.2, colors=TEXT_MAIN)
    ax.tick_params(axis="y", length=0, pad=5.0, labelsize=12.0, colors=TEXT_MAIN)


def plot_panel(
    ax: plt.Axes,
    stats: list[RowStat],
    title: str,
    y: np.ndarray,
    bar_height: float,
) -> None:
    left = np.zeros(len(stats), dtype=float)
    for layer in LAYER_KEYS:
        widths = np.array([stat.percentage(layer) for stat in stats], dtype=float)
        ax.barh(
            y,
            widths,
            left=left,
            height=bar_height,
            color=LAYER_COLORS[layer],
            edgecolor="#FFFFFF",
            linewidth=0.5,
            zorder=3,
        )
        for yi, width, start in zip(y, widths, left):
            if width < MIN_LABEL_PCT:
                continue
            ax.text(
                start + width / 2.0,
                yi,
                f"{width:.1f}%",
                ha="center",
                va="center",
                fontsize=10.0,
                color="#FFFFFF",
                clip_on=True,
                zorder=4,
            )
        left = left + widths
    if not np.allclose(left, 100.0, atol=1e-9):
        raise SystemExit(f"{title}: stacked shares are not 100%: {left.tolist()}")

    for yi, stat in zip(y, stats):
        ax.text(
            102.0,
            yi,
            f"n={stat.failure_denominator:,}",
            ha="left",
            va="center",
            fontsize=10.0,
            color=TEXT_MUTED,
            clip_on=False,
            fontweight="bold" if stat.is_aggregate else "normal",
            zorder=5,
        )

    agg_index = next(index for index, stat in enumerate(stats) if stat.is_aggregate)
    ax.add_patch(
        Rectangle(
            (0.0, y[agg_index] - bar_height / 2.0),
            100.0,
            bar_height,
            fill=False,
            edgecolor=AGG_EDGE,
            linewidth=1.15,
            zorder=6,
            clip_on=False,
        )
    )
    separator_y = (y[agg_index - 1] + y[agg_index]) / 2.0
    ax.axhline(separator_y, color=GUIDE_COLOR, linewidth=0.85, zorder=1)
    ax.axvline(50.0, color=GUIDE_COLOR, linewidth=0.7, zorder=1)

    ax.set_xlim(0, 126)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"))
    ax.set_xlabel("Share of failure observations", fontsize=12.0, color=TEXT_MAIN, labelpad=6)
    ax.set_title(title, loc="center", fontsize=14.0, pad=10, color=TEXT_MAIN)
    style_axes(ax)


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
    bar_height = 0.70

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 7.6),
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    fig.patch.set_facecolor("#FFFFFF")

    for ax, condition, title in zip(axes, CONDITIONS, ("(a) MR", "(b) VA")):
        plot_panel(ax, by_condition[condition], title, y, bar_height)

    pad = bar_height / 2.0 + 0.20
    axes[0].set_ylim(-pad, float(n_rows - 1) + pad)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(row_labels, fontsize=12.0)
    axes[0].invert_yaxis()
    for tick, name in zip(axes[0].get_yticklabels(), row_labels):
        if name == AGGREGATE_LABEL:
            tick.set_fontweight("bold")
            tick.set_color(TEXT_MAIN)
    axes[1].tick_params(axis="y", labelleft=False)

    handles = [
        Patch(
            facecolor=LAYER_COLORS[layer],
            edgecolor="#FFFFFF",
            linewidth=0.5,
            label=LAYER_DISPLAY[layer],
        )
        for layer in LAYER_KEYS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=12.0,
        handlelength=1.5,
        handleheight=0.85,
        handletextpad=0.45,
        columnspacing=1.7,
        borderaxespad=0.0,
        bbox_to_anchor=(0.545, 0.995),
        bbox_transform=fig.transFigure,
    )
    fig.subplots_adjust(left=0.168, right=0.972, top=0.875, bottom=0.10, wspace=0.18)
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


def model_mapping_lines() -> list[str]:
    lines = []
    for raw, display in MODEL_LABEL.items():
        lines.append(f"- `{raw}` -> `{display}`")
    lines.append(
        "- `meta-llama/llama-4-scout` is accepted through `paper_display_label` and displayed as `Llama-4-Scout`"
    )
    return lines


def render_report(
    input_path: Path,
    input_sha: str,
    by_condition: dict[str, list[RowStat]],
    assertions: list[Assertion],
    output_paths: dict[str, Path],
    output_sha: dict[str, str],
    reproduce_cmd: str,
    png_meta: dict[str, float | int],
    pdf_meta: dict[str, object],
) -> str:
    lines = [
        "# RQ3 failure-layer distribution verification report",
        "",
        "## Input",
        "",
        f"- path: `{display_path(input_path)}`",
        f"- SHA-256: `{input_sha}`",
        "",
        "## Filter",
        "",
        "- source: `task_outcomes.jsonl` from the corrected canonical tables",
        "- do not use `compatibility_layers.csv`",
        "- `track == \"d2\"`",
        "- `resolver_status != \"empty\"` (drop no-dependency observations)",
        "- `compatibility_layer != \"compatible\"` (failure-only)",
        "- retain only the four mutually exclusive static failure layers listed below",
        "- stop if `infrastructure`, `other`, a missing layer, or any other unexpected layer is present",
        "- the one MR `api_mismatch_candidate` observation with `resolver_status=empty` is excluded by the empty-resolver filter; the MR API-mismatch candidate count must therefore be 1,558 rather than 1,559",
        "",
        "## Mode mapping",
        "",
        "- `requirements.txt` -> `MR`",
        "- `inline` -> `VA`",
        "",
        "## Model-name mapping",
        "",
        "Display names and order come from `plots.model_display.MODEL_LABEL` via `paper_display_label`.",
        "No new aliases are introduced.",
        "",
    ]
    lines.extend(model_mapping_lines())
    lines.extend(
        [
            "",
            f"- figure order: {', '.join(DISPLAY_ORDER)}, then `{AGGREGATE_LABEL}`",
            f"- `{AGGREGATE_LABEL}` sums raw counts across models before computing percentages",
            "",
            "## Overall counts and shares",
            "",
        ]
    )
    for condition in CONDITIONS:
        aggregate = by_condition[condition][-1]
        total = aggregate.failure_denominator
        lines.append(f"### {condition}")
        lines.append("")
        lines.append(f"- failure observations: {total:,}")
        for layer in LAYER_KEYS:
            count = aggregate.counts[layer]
            lines.append(
                f"- {LAYER_DISPLAY[layer]}: {count:,} ({aggregate.percentage(layer):.2f}%)"
            )
        lines.append("")

    lines.extend(["## Per-model failure denominators and layer counts", ""])
    for condition in CONDITIONS:
        lines.append(f"### {condition}")
        lines.append("")
        for stat in by_condition[condition]:
            parts = [
                f"{LAYER_DISPLAY[layer]}={stat.counts[layer]:,} ({stat.percentage(layer):.2f}%)"
                for layer in LAYER_KEYS
            ]
            lines.append(
                f"- {stat.model_display} (`{stat.model_raw}`): n={stat.failure_denominator:,}; "
                + "; ".join(parts)
            )
        lines.append("")

    passed = sum(item.passed for item in assertions)
    lines.extend(
        [
            "## Assertions",
            "",
            f"- {passed}/{len(assertions)} passed",
            "",
        ]
    )
    for item in assertions:
        status = "pass" if item.passed else "FAIL"
        lines.append(f"- {item.name}: {status} ({item.detail})")

    va_models = [stat for stat in by_condition["VA"] if not stat.is_aggregate]
    mr_models = [stat for stat in by_condition["MR"] if not stat.is_aggregate]
    mr_api = [stat.model_display for stat in mr_models if stat.dominant_layer() == "api_mismatch_candidate"]
    mr_env = [stat.model_display for stat in mr_models if stat.dominant_layer() == "environment_build"]
    lines.extend(
        [
            "",
            "## Cross-model pattern checks",
            "",
            "- VA: Environment construction is the largest failure layer in all 10 models:",
        ]
    )
    for stat in va_models:
        lines.append(
            f"  - {stat.model_display}: {stat.percentage('environment_build'):.2f}% "
            f"({stat.counts['environment_build']:,}/{stat.failure_denominator:,})"
        )
    va_env_pcts = [stat.percentage("environment_build") for stat in va_models]
    min_stat = min(va_models, key=lambda stat: stat.percentage("environment_build"))
    max_stat = max(va_models, key=lambda stat: stat.percentage("environment_build"))
    lines.extend(
        [
            f"- VA Environment construction range: {min(va_env_pcts):.2f}% ({min_stat.model_display}) to "
            f"{max(va_env_pcts):.2f}% ({max_stat.model_display})",
            "- MR models whose largest failure layer is API-mismatch candidate:",
            f"  - {', '.join(mr_api)} ({len(mr_api)})",
            "- MR models whose largest failure layer is Environment construction:",
            f"  - {', '.join(mr_env)} ({len(mr_env)})",
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
            "Manuscript files were not modified.  This script writes only the analysis CSV, PDF, PNG, and this report under `plots/`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "rq3_failure_layers.csv"
    pdf_path = output_dir / "rq3_failure_layers.pdf"
    png_path = output_dir / "rq3_failure_layers.png"
    report_path = output_dir / "rq3_failure_layers_report.md"
    reproduce_cmd = ".venv/bin/python -m plots.rq3_failure_layers"

    rows = load_jsonl(input_path)
    selected = select_failure_observations(rows)
    by_condition = build_stats(selected)
    assertions = run_assertions(selected, by_condition)
    failed = [item for item in assertions if not item.passed]
    if failed:
        details = "\n".join(f"- FAIL {item.name}: {item.detail}" for item in failed)
        raise SystemExit(
            "Assertions failed; not writing a figure from a disputed population.\n"
            + details
        )

    records = csv_records(by_condition)
    expected_csv_rows = len(CONDITIONS) * (N_MODELS + 1) * len(LAYER_KEYS)
    if len(records) != expected_csv_rows:
        raise SystemExit(f"CSV rows={len(records)} expected {expected_csv_rows}")

    plot_figure(by_condition, pdf_path, png_path, args.dpi)
    write_csv(csv_path, records)
    hashed_outputs = {
        "csv": csv_path,
        "pdf": pdf_path,
        "png": png_path,
    }
    output_paths = {**hashed_outputs, "report": report_path}
    for path in hashed_outputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"output missing or empty: {path}")

    png_meta = png_geometry(png_path)
    pdf_meta = pdf_geometry(pdf_path)
    report = render_report(
        input_path=input_path,
        input_sha=sha256_file(input_path),
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
    print(f"Report SHA-256: {sha256_file(report_path)}")


if __name__ == "__main__":
    main()
