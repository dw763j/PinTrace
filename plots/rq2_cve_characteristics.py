"""RQ2 CVE characteristics of valid exact VA advice on clean PinTrace.

Panel (a) shows the within-model distribution of maximum catalogued severity
across CVE-associated dependency occurrences.  UNKNOWN and NONE are merged
into Unscored.  Panel (b) shows the share of each model's unique associated
advisories published on or before its documented knowledge cutoff.  Intervals
for self-reported cutoffs are sensitivity bounds, not confidence intervals.

Population encoded in the corrected inputs:
    D2 clean, mode=inline (VA), resolution_origin=model_exact,
    constraint_valid=true, is_cve_associated=true.

Usage (from project root)::

    python -m plots.rq2_cve_characteristics
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plots.model_display import MODEL_LABEL, paper_display_label
from paths import ANALYSIS_RQ2, PAPER_RESULTS, display_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = PAPER_RESULTS / "figures"

DEFAULT_SEVERITY = ANALYSIS_RQ2 / "vulnerability_severity_exact.csv"
DEFAULT_DISCLOSURE = ANALYSIS_RQ2 / "cve_disclosure_exact.csv"
DEFAULT_CUTOFFS = PROJECT_ROOT / "resources" / "revision" / "model_knowledge_cutoff.json"
DEFAULT_PDF = PLOTS_DIR / "rq2_cve_characteristics.pdf"
DEFAULT_CSV = PLOTS_DIR / "rq2_cve_characteristics.csv"
DEFAULT_REPORT = PLOTS_DIR / "rq2_cve_characteristics_report.md"
DEFAULT_MANUSCRIPT_PDF = MANUSCRIPT_DIR / "rq2_cve_characteristics.pdf"

SEVERITY_ORDER = ("Critical", "High", "Medium", "Low", "Unscored")
SOURCE_TO_PLOT = {
    "CRITICAL": "Critical",
    "HIGH": "High",
    "MEDIUM": "Medium",
    "LOW": "Low",
    "UNKNOWN": "Unscored",
    "NONE": "Unscored",
}
SEVERITY_COLORS = {
    "Critical": "#8B1E3F",
    "High": "#D95F59",
    "Medium": "#E3A857",
    "Low": "#5F9E6E",
    "Unscored": "#A8ADB4",
}
CUTOFF_OFFICIAL = "#D55E00"
CUTOFF_SELF = "#7A5195"
CUTOFF_SENS3 = "#9D79AD"
CUTOFF_SENS6 = "#D8C7DF"
GRID_COLOR = "#D8D8D0"
GUIDE_COLOR = "#E6E6DE"
SPINE_COLOR = "#8A8A8A"
TEXT_MAIN = "#222222"
TEXT_MUTED = "#5A5A5A"

EXPECTED_TOTAL = 5_301
EXPECTED_SEVERITY = {
    "Critical": 894,
    "High": 2_193,
    "Medium": 1_767,
    "Low": 245,
    "Unscored": 202,
}
EXPECTED_UNKNOWN = 194
EXPECTED_NONE = 8
EXPECTED_N_MODELS = 10
EXPECTED_RATE_MIN = 26.28
EXPECTED_RATE_MAX = 86.55
EXPECTED_N_RATES_GE_50 = 9
CSV_FIELDS = (
    "model",
    "display_name",
    "occurrences",
    "critical",
    "high",
    "medium",
    "low",
    "unscored",
    "unscored_unknown",
    "unscored_none",
    "critical_pct",
    "high_pct",
    "medium_pct",
    "low_pct",
    "unscored_pct",
    "unique_advisories",
    "cutoff_source",
    "cutoff_date",
    "pre_cutoff",
    "central_rate_pct",
    "rate_minus_3_pct",
    "rate_plus_3_pct",
    "rate_minus_6_pct",
    "rate_plus_6_pct",
)


@dataclass
class ModelRow:
    model: str
    display: str
    occurrences: int = 0
    counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in SEVERITY_ORDER})
    unknown: int = 0
    none: int = 0
    unique_advisories: int = 0
    cutoff_source: str = ""
    cutoff_date: str = ""
    rates: dict[int, float] = field(default_factory=dict)
    pre_cutoff: dict[int, int] = field(default_factory=dict)

    @property
    def central_rate(self) -> float:
        return self.rates[0]

    def pct(self, label: str) -> float:
        return 100.0 * self.counts[label] / self.occurrences if self.occurrences else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--severity-csv", type=Path, default=DEFAULT_SEVERITY)
    parser.add_argument("--disclosure-csv", type=Path, default=DEFAULT_DISCLOSURE)
    parser.add_argument("--cutoff-json", type=Path, default=DEFAULT_CUTOFFS)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manuscript-pdf", type=Path, default=DEFAULT_MANUSCRIPT_PDF)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_cutoffs(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key).strip().lower(): value for key, value in raw.items()}


def _cutoff_for(model: str, cutoffs: dict[str, dict]) -> dict:
    key = model.strip().lower()
    for candidate in (key, key.replace("/", "_"), key.replace("_", "/")):
        if candidate in cutoffs:
            return cutoffs[candidate]
    raise KeyError(f"no knowledge cutoff for {model!r}")


def _ordered_models(names: list[str]) -> list[str]:
    labelled = [(paper_display_label(name, name), name) for name in names]
    rank = {label: index for index, label in enumerate(MODEL_LABEL.values())}
    labelled.sort(key=lambda item: (rank.get(item[0], 10_000), item[0]))
    return [name for _, name in labelled]


def _pct(value: float) -> float:
    return round(100.0 * value, 2)


def load_rows(
    severity_csv: Path,
    disclosure_csv: Path,
    cutoff_json: Path,
) -> list[ModelRow]:
    cutoffs = _load_cutoffs(cutoff_json)
    by_model: dict[str, ModelRow] = {}

    for rec in _read_csv(severity_csv):
        model = str(rec["model"])
        row = by_model.setdefault(
            model,
            ModelRow(model=model, display=paper_display_label(model, model)),
        )
        source = str(rec["severity"]).strip().upper()
        if source not in SOURCE_TO_PLOT:
            raise ValueError(f"unexpected severity {source!r} for {model}")
        count = int(rec["occurrences"])
        unit = str(rec.get("unit") or "")
        if unit and unit != "dependency_occurrence":
            raise ValueError(f"unexpected unit {unit!r} for {model}")
        row.counts[SOURCE_TO_PLOT[source]] += count
        row.occurrences += count
        if source == "UNKNOWN":
            row.unknown += count
        elif source == "NONE":
            row.none += count

    for rec in _read_csv(disclosure_csv):
        model = str(rec["model"])
        row = by_model.setdefault(
            model,
            ModelRow(model=model, display=paper_display_label(model, model)),
        )
        offset = int(rec["offset_months"])
        source = str(rec["source"]).strip().lower()
        unique = int(rec["unique_advisories"])
        if row.unique_advisories and row.unique_advisories != unique:
            raise ValueError(f"unique_advisories changed for {model}")
        if row.cutoff_source and row.cutoff_source != source:
            raise ValueError(f"cutoff source changed for {model}")
        row.unique_advisories = unique
        row.cutoff_source = source
        row.rates[offset] = float(rec["pre_cutoff_rate"])
        row.pre_cutoff[offset] = int(rec["pre_cutoff"])

    missing = [name for name, row in by_model.items() if 0 not in row.rates]
    if missing:
        raise ValueError(f"missing central disclosure rate for: {missing}")

    for model, row in by_model.items():
        cutoff = _cutoff_for(model, cutoffs)
        json_source = str(cutoff["source"]).strip().lower()
        if json_source != row.cutoff_source:
            raise ValueError(
                f"cutoff source mismatch for {model}: csv={row.cutoff_source} json={json_source}"
            )
        row.cutoff_date = str(cutoff["date"])
        if row.cutoff_source == "self_reported":
            needed = {-6, -3, 0, 3, 6}
            if set(row.rates) != needed:
                raise ValueError(f"{model} self-reported offsets {sorted(row.rates)} != {sorted(needed)}")
        elif row.cutoff_source == "official":
            if set(row.rates) != {0}:
                raise ValueError(f"{model} official offsets {sorted(row.rates)} != [0]")
        else:
            raise ValueError(f"unexpected cutoff source {row.cutoff_source!r} for {model}")

    return [by_model[name] for name in _ordered_models(list(by_model))]


def acceptance_errors(rows: list[ModelRow]) -> list[str]:
    errors: list[str] = []
    if len(rows) != EXPECTED_N_MODELS:
        errors.append(f"n_models={len(rows)} expected {EXPECTED_N_MODELS}")

    expected_labels = list(MODEL_LABEL.values())
    got_labels = [row.display for row in rows]
    if got_labels != expected_labels:
        errors.append(f"model order {got_labels} != {expected_labels}")

    unknown = sum(row.unknown for row in rows)
    none = sum(row.none for row in rows)
    totals = {label: sum(row.counts[label] for row in rows) for label in SEVERITY_ORDER}
    total = sum(totals.values())
    if unknown != EXPECTED_UNKNOWN:
        errors.append(f"UNKNOWN={unknown} expected {EXPECTED_UNKNOWN}")
    if none != EXPECTED_NONE:
        errors.append(f"NONE={none} expected {EXPECTED_NONE}")
    if unknown + none != totals["Unscored"]:
        errors.append("Unscored is not UNKNOWN + NONE")
    for label, expected in EXPECTED_SEVERITY.items():
        if totals[label] != expected:
            errors.append(f"{label}={totals[label]} expected {expected}")
    if total != EXPECTED_TOTAL:
        errors.append(f"total occurrences={total} expected {EXPECTED_TOTAL}")

    rates = [_pct(row.central_rate) for row in rows]
    if min(rates) != EXPECTED_RATE_MIN or max(rates) != EXPECTED_RATE_MAX:
        errors.append(
            f"central rates {min(rates):.2f}--{max(rates):.2f} "
            f"expected {EXPECTED_RATE_MIN:.2f}--{EXPECTED_RATE_MAX:.2f}"
        )
    n_ge_50 = sum(rate >= 50.0 for rate in rates)
    if n_ge_50 != EXPECTED_N_RATES_GE_50:
        errors.append(f"central rates >= 50% : {n_ge_50} expected {EXPECTED_N_RATES_GE_50}")
    return errors


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 8,
        }
    )


def _style_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, color=GRID_COLOR, alpha=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.tick_params(axis="x", length=3.0, color=SPINE_COLOR, labelsize=7.2, pad=1.2)
    ax.tick_params(axis="y", length=0, pad=3.0, labelsize=7.6)


def _horizontal_guides(ax, n_rows: int) -> None:
    for index in range(n_rows - 1):
        ax.axhline(index + 0.5, color=GUIDE_COLOR, linewidth=0.6, zorder=0)


def plot_figure(rows: list[ModelRow], output: Path, dpi: int) -> None:
    _apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.62),
        sharey=True,
        gridspec_kw={"width_ratios": [0.56, 0.44], "wspace": 0.12},
    )
    fig.patch.set_facecolor("#FFFFFF")
    y = np.arange(len(rows), dtype=float)
    bar_height = 0.58

    _horizontal_guides(ax_a, len(rows))
    left = np.zeros(len(rows))
    for label in SEVERITY_ORDER:
        widths = np.array([row.pct(label) for row in rows], dtype=float)
        ax_a.barh(
            y,
            widths,
            left=left,
            height=bar_height,
            color=SEVERITY_COLORS[label],
            edgecolor="#FFFFFF",
            linewidth=0.45,
            label=label,
            zorder=3,
        )
        left = left + widths
    if not np.allclose(left, 100.0, atol=1e-9):
        raise SystemExit(f"stacked shares are not 100%: {left.tolist()}")

    for yi, row in zip(y, rows):
        ax_a.text(
            101.6,
            yi,
            f"n={row.occurrences:,}",
            ha="left",
            va="center",
            fontsize=6.2,
            color=TEXT_MUTED,
            clip_on=False,
        )

    ax_a.set_xlim(0, 117)
    ax_a.set_xticks((0, 25, 50, 75, 100))
    ax_a.set_xticklabels(("0%", "25%", "50%", "75%", "100%"))
    ax_a.set_xlabel("Share of CVE-associated occurrences", fontsize=8.0, color=TEXT_MAIN)
    ax_a.set_title("(a) Maximum catalogued severity", loc="left", fontsize=9.0, pad=4, color=TEXT_MAIN)
    _style_axes(ax_a)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([row.display for row in rows], fontsize=7.6)
    ax_a.set_ylim(-0.42, len(rows) - 0.58)
    ax_a.invert_yaxis()

    _horizontal_guides(ax_b, len(rows))
    for yi, row in zip(y, rows):
        if row.cutoff_source == "self_reported":
            outer_left = 100.0 * row.rates[-6]
            outer_width = 100.0 * (row.rates[6] - row.rates[-6])
            inner_left = 100.0 * row.rates[-3]
            inner_width = 100.0 * (row.rates[3] - row.rates[-3])
            ax_b.barh(
                yi,
                outer_width,
                left=outer_left,
                height=0.16,
                color=CUTOFF_SENS6,
                edgecolor="none",
                zorder=2,
            )
            ax_b.barh(
                yi,
                inner_width,
                left=inner_left,
                height=0.28,
                color=CUTOFF_SENS3,
                edgecolor="none",
                zorder=3,
            )
            ax_b.plot(
                100.0 * row.central_rate,
                yi,
                linestyle="None",
                marker="D",
                markersize=4.8,
                markerfacecolor="#FFFFFF",
                markeredgecolor=CUTOFF_SELF,
                markeredgewidth=1.15,
                zorder=4,
            )
        else:
            ax_b.plot(
                100.0 * row.central_rate,
                yi,
                linestyle="None",
                marker="o",
                markersize=4.8,
                markerfacecolor=CUTOFF_OFFICIAL,
                markeredgecolor=CUTOFF_OFFICIAL,
                markeredgewidth=0.6,
                zorder=4,
            )
        ax_b.text(
            101.6,
            yi,
            f"{_pct(row.central_rate):.1f}%",
            ha="left",
            va="center",
            fontsize=6.2,
            color=TEXT_MUTED,
            clip_on=False,
        )

    ax_b.set_xlim(0, 117)
    ax_b.set_xticks((0, 25, 50, 75, 100))
    ax_b.set_xticklabels(("0%", "25%", "50%", "75%", "100%"))
    ax_b.set_xlabel("Share of unique advisories published by cutoff", fontsize=8.0, color=TEXT_MAIN)
    ax_b.set_title("(b) Advisories published by cutoff", loc="left", fontsize=9.0, pad=4, color=TEXT_MAIN)
    _style_axes(ax_b)
    ax_b.tick_params(axis="y", labelleft=False)

    sev_handles = [
        Patch(facecolor=SEVERITY_COLORS[label], edgecolor="#FFFFFF", linewidth=0.5, label=label)
        for label in SEVERITY_ORDER
    ]
    cutoff_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=5.6,
            markerfacecolor=CUTOFF_OFFICIAL,
            markeredgecolor=CUTOFF_OFFICIAL,
            label="Provider-published",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markersize=5.6,
            markerfacecolor="#FFFFFF",
            markeredgecolor=CUTOFF_SELF,
            markeredgewidth=1.15,
            label="Self-reported",
        ),
        Patch(facecolor=CUTOFF_SENS3, edgecolor="none", label="\u00b13-month sensitivity"),
        Patch(facecolor=CUTOFF_SENS6, edgecolor="none", label="\u00b16-month sensitivity"),
    ]
    leg_a = fig.legend(
        handles=sev_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.0,
        handlelength=1.15,
        handleheight=0.58,
        handletextpad=0.35,
        columnspacing=1.05,
        borderaxespad=0.0,
        bbox_to_anchor=(0.55, 0.038),
        bbox_transform=fig.transFigure,
    )
    fig.add_artist(leg_a)
    fig.legend(
        handles=cutoff_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.0,
        handlelength=1.15,
        handleheight=0.58,
        handletextpad=0.35,
        columnspacing=1.05,
        borderaxespad=0.0,
        bbox_to_anchor=(0.55, -0.018),
        bbox_transform=fig.transFigure,
    )
    fig.subplots_adjust(left=0.168, right=0.985, top=0.90, bottom=0.24, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _pct_field(value: float | None) -> str:
    if value is None:
        return ""
    return f"{_pct(value):.2f}"


def csv_record(row: ModelRow) -> dict[str, object]:
    official = row.cutoff_source == "official"
    return {
        "model": row.model,
        "display_name": row.display,
        "occurrences": row.occurrences,
        "critical": row.counts["Critical"],
        "high": row.counts["High"],
        "medium": row.counts["Medium"],
        "low": row.counts["Low"],
        "unscored": row.counts["Unscored"],
        "unscored_unknown": row.unknown,
        "unscored_none": row.none,
        "critical_pct": f"{row.pct('Critical'):.2f}",
        "high_pct": f"{row.pct('High'):.2f}",
        "medium_pct": f"{row.pct('Medium'):.2f}",
        "low_pct": f"{row.pct('Low'):.2f}",
        "unscored_pct": f"{row.pct('Unscored'):.2f}",
        "unique_advisories": row.unique_advisories,
        "cutoff_source": row.cutoff_source,
        "cutoff_date": row.cutoff_date,
        "pre_cutoff": row.pre_cutoff[0],
        "central_rate_pct": _pct_field(row.central_rate),
        "rate_minus_3_pct": "" if official else _pct_field(row.rates[-3]),
        "rate_plus_3_pct": "" if official else _pct_field(row.rates[3]),
        "rate_minus_6_pct": "" if official else _pct_field(row.rates[-6]),
        "rate_plus_6_pct": "" if official else _pct_field(row.rates[6]),
    }


def write_csv(path: Path, rows: list[ModelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_record(row) for row in rows)


def render_report(args: argparse.Namespace, rows: list[ModelRow], errors: list[str]) -> str:
    totals = {label: sum(row.counts[label] for row in rows) for label in SEVERITY_ORDER}
    total = sum(totals.values())
    unknown = sum(row.unknown for row in rows)
    none = sum(row.none for row in rows)
    rates = [_pct(row.central_rate) for row in rows]
    n_ge_50 = sum(rate >= 50.0 for rate in rates)
    official = [row for row in rows if row.cutoff_source == "official"]
    self_reported = [row for row in rows if row.cutoff_source == "self_reported"]
    lines = [
        "# RQ2 CVE-characteristics verification report",
        "",
        "## Reproduction command",
        "",
        "From the repository root:",
        "",
        "`.venv/bin/python -m plots.rq2_cve_characteristics`",
        "",
        "## Inputs and population",
        "",
        f"- severity: `{display_path(args.severity_csv)}`",
        f"- disclosure: `{display_path(args.disclosure_csv)}`",
        f"- knowledge cutoffs: `{display_path(args.cutoff_json)}`",
        "- population encoded in the corrected CSVs: D2 clean, mode=`inline` (VA), `resolution_origin=model_exact`, `constraint_valid=true`, `is_cve_associated=true`",
        f"- models: {len(rows)}",
        "- model order: same as Table tab:rq2_cve_association / `plots.model_display.MODEL_LABEL`",
        "",
        "## Panel (a): maximum catalogued severity",
        "",
        "- analysis unit: CVE-associated exact dependency occurrence",
        "- 100% horizontal stacked bars",
        f"- UNKNOWN={unknown} and NONE={none} merged into Unscored={totals['Unscored']}",
        f"- Critical={totals['Critical']:,}, High={totals['High']:,}, Medium={totals['Medium']:,}, Low={totals['Low']:,}, Unscored={totals['Unscored']:,}",
        f"- total occurrences={total:,} (asserted {EXPECTED_TOTAL:,} before plotting)",
        "",
    ]
    for row in rows:
        parts = ", ".join(f"{label}={row.counts[label]}" for label in SEVERITY_ORDER)
        lines.append(f"- {row.display}: n={row.occurrences:,}; {parts}")
    lines.extend(
        [
            "",
            "## Panel (b): unique advisories published by cutoff",
            "",
            "- analysis unit: unique associated advisory per model",
            "- repeated dependency occurrences do not increase weight",
            f"- provider-published cutoffs: {len(official)} models, filled circles, no sensitivity interval",
            f"- self-reported cutoffs: {len(self_reported)} models, hollow diamonds, inner \u00b13-month and outer \u00b16-month sensitivity bounds",
            "- these intervals are cutoff-date sensitivity bounds, not confidence intervals",
            f"- central rates range {min(rates):.2f}%--{max(rates):.2f}%",
            f"- models with central rate >= 50%: {n_ge_50}/{len(rows)}",
            "",
        ]
    )
    for row in rows:
        source = "provider-published" if row.cutoff_source == "official" else "self-reported"
        extra = ""
        if row.cutoff_source == "self_reported":
            extra = (
                f"; \u00b13 mo {_pct(row.rates[-3]):.2f}--{_pct(row.rates[3]):.2f}%; "
                f"\u00b16 mo {_pct(row.rates[-6]):.2f}--{_pct(row.rates[6]):.2f}%"
            )
        lines.append(
            f"- {row.display}: {row.pre_cutoff[0]}/{row.unique_advisories} "
            f"({_pct(row.central_rate):.2f}%), {source}, cutoff={row.cutoff_date}{extra}"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{display_path(args.output_pdf)}`",
            f"- `{display_path(args.output_csv)}`",
            f"- `{display_path(args.output_report)}`",
            f"- `{display_path(args.manuscript_pdf)}`",
            "",
            "## Acceptance checks",
            "",
        ]
    )
    if errors:
        lines.append("- FAILED:")
        lines.extend(f"  - {item}" for item in errors)
    else:
        lines.extend(
            [
                f"- severity totals Critical={EXPECTED_SEVERITY['Critical']:,}, High={EXPECTED_SEVERITY['High']:,}, Medium={EXPECTED_SEVERITY['Medium']:,}, Low={EXPECTED_SEVERITY['Low']:,}, Unscored={EXPECTED_SEVERITY['Unscored']:,}: pass",
                f"- UNKNOWN={EXPECTED_UNKNOWN} and NONE={EXPECTED_NONE} merge into Unscored: pass",
                f"- total occurrences={EXPECTED_TOTAL:,}: pass",
                f"- central disclosure rates {EXPECTED_RATE_MIN:.2f}%--{EXPECTED_RATE_MAX:.2f}%: pass",
                f"- {EXPECTED_N_RATES_GE_50} of {EXPECTED_N_MODELS} central rates >= 50%: pass",
                "- cutoff JSON source matches the disclosure CSV for every model: pass",
                "- model order matches Table tab:rq2_cve_association: pass",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def copy_manuscript_pdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def main() -> None:
    args = parse_args()
    rows = load_rows(args.severity_csv, args.disclosure_csv, args.cutoff_json)
    total = sum(row.occurrences for row in rows)
    if total != EXPECTED_TOTAL:
        raise SystemExit(f"asserted {EXPECTED_TOTAL:,} occurrences before plotting; got {total:,}")
    errors = acceptance_errors(rows)
    plot_figure(rows, args.output_pdf, args.dpi)
    write_csv(args.output_csv, rows)
    copy_manuscript_pdf(args.output_pdf, args.manuscript_pdf)
    report = render_report(args, rows, errors)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")
    print(report)
    if errors:
        raise SystemExit("Acceptance checks failed; see the report.")
    print(f"Saved: {args.output_pdf}")
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.output_report}")
    print(f"Saved: {args.manuscript_pdf}")


if __name__ == "__main__":
    main()
