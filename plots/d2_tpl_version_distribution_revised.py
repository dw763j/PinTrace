"""RQ1 version-concentration figure (revised exact-declaration population).

Compare the concentration of model-written exact version strings between MR
and VA for the ten packages with the most exact declarations across both
conditions.

Input:
    outputs/evaluation/direct_dependencies.jsonl

Filters:
    track == "d2", is_direct, constraint_kind == "exact"
    Do not filter by constraint_valid or resolution_origin.

Usage (from project root)::

    python -m plots.d2_tpl_version_distribution_revised
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paths import EVALUATION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EVALUATION / "direct_dependencies.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "d2_tpl_version_distribution_revised.pdf"
DEFAULT_JSON = Path(__file__).resolve().parent / "d2_tpl_version_distribution_revised.json"

MODE_MR = "requirements.txt"
MODE_VA = "inline"
COND_MR = "MR"
COND_VA = "VA"
CONDITIONS = (COND_MR, COND_VA)
TOP_N = 10
MIN_LABEL_SHARE = 0.12

RANK_COLORS = ("#2F5597", "#3E6CB8", "#B4C7E7", "#D9D9D9")
RANK_TEXT = ("#FFFFFF", "#FFFFFF", "#1B2A4A", "#4A4A4A")
RANK_LEGEND = ("Most frequent", "Second", "Third", "Other")

EXPECTED_MR = 4177
EXPECTED_VA = 19134
EXPECTED_TOP10 = (
    "numpy",
    "pandas",
    "django",
    "selenium",
    "requests",
    "flask",
    "matplotlib",
    "pillow",
    "torch",
    "djangorestframework",
)
EXPECTED_SAME_MODAL = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detail-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _normalize_package(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _condition(mode: object) -> str | None:
    if mode == MODE_MR:
        return COND_MR
    if mode == MODE_VA:
        return COND_VA
    return None


def collect_counts(path: Path) -> tuple[Counter[str], dict[tuple[str, str], Counter[str]]]:
    totals: Counter[str] = Counter()
    by_cond_pkg: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("track") != "d2" or not row.get("is_direct"):
                continue
            if row.get("constraint_kind") != "exact":
                continue
            cond = _condition(row.get("mode"))
            package = _normalize_package(row.get("pypi_name"))
            if cond is None or not package:
                continue
            version = str(row.get("specified_version") or "").strip()
            totals[cond] += 1
            by_cond_pkg[(cond, package)][version] += 1
    return totals, dict(by_cond_pkg)


def combined_totals(by_cond_pkg: dict[tuple[str, str], Counter[str]]) -> Counter[str]:
    combined: Counter[str] = Counter()
    for (cond, package), counter in by_cond_pkg.items():
        if cond in CONDITIONS:
            combined[package] += sum(counter.values())
    return combined


def rank_packages(by_cond_pkg: dict[tuple[str, str], Counter[str]], top_n: int) -> list[str]:
    combined = combined_totals(by_cond_pkg)
    return [name for name, _ in combined.most_common(top_n)]


def package_total(by_cond_pkg: dict[tuple[str, str], Counter[str]], cond: str, package: str) -> int:
    return sum(by_cond_pkg.get((cond, package), Counter()).values())


def segments_for(counter: Counter[str]) -> list[tuple[str, int, int]]:
    """Return (label, count, rank) with rank 0–2 for top versions and 3 for Other."""
    ranked = [(version, count) for version, count in counter.most_common() if version]
    ranked.extend((version, count) for version, count in counter.most_common() if not version)
    top = ranked[:3]
    kept = {version for version, _ in top}
    other = sum(count for version, count in ranked if version not in kept)
    segments = [(version, count, rank) for rank, (version, count) in enumerate(top)]
    if other:
        segments.append(("Other", other, 3))
    return segments


def modal_version(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def sanity_report(
    totals: Counter[str],
    by_cond_pkg: dict[tuple[str, str], Counter[str]],
    packages: list[str],
) -> dict:
    same_modal = 0
    package_rows = []
    for package in packages:
        mr_ctr = by_cond_pkg.get((COND_MR, package), Counter())
        va_ctr = by_cond_pkg.get((COND_VA, package), Counter())
        mr_modal = modal_version(mr_ctr)
        va_modal = modal_version(va_ctr)
        same = bool(mr_modal) and mr_modal == va_modal
        same_modal += int(same)
        package_rows.append({
            "package": package,
            "combined": package_total(by_cond_pkg, COND_MR, package)
            + package_total(by_cond_pkg, COND_VA, package),
            "mr_n": package_total(by_cond_pkg, COND_MR, package),
            "va_n": package_total(by_cond_pkg, COND_VA, package),
            "mr_modal": mr_modal,
            "va_modal": va_modal,
            "same_modal": same,
            "mr_segments": [
                {"label": label, "count": count, "rank": rank}
                for label, count, rank in segments_for(mr_ctr)
            ],
            "va_segments": [
                {"label": label, "count": count, "rank": rank}
                for label, count, rank in segments_for(va_ctr)
            ],
        })
    report = {
        "mr_exact_declarations": totals[COND_MR],
        "va_exact_declarations": totals[COND_VA],
        "top_packages": packages,
        "same_modal_count": same_modal,
        "packages": package_rows,
        "checks": {
            "mr_exact": totals[COND_MR] == EXPECTED_MR,
            "va_exact": totals[COND_VA] == EXPECTED_VA,
            "top10": tuple(packages) == EXPECTED_TOP10,
            "same_modal": same_modal == EXPECTED_SAME_MODAL,
        },
    }
    failed = [name for name, ok in report["checks"].items() if not ok]
    if failed:
        raise SystemExit(
            "Sanity checks failed: "
            + ", ".join(failed)
            + f"\n  MR={totals[COND_MR]} (expected {EXPECTED_MR})"
            + f"\n  VA={totals[COND_VA]} (expected {EXPECTED_VA})"
            + f"\n  top10={packages}"
            + f"\n  same_modal={same_modal} (expected {EXPECTED_SAME_MODAL})"
        )
    return report


def _segment_label(version: str, share: float) -> str:
    pct = f"{share * 100:.0f}%"
    if share >= 0.32:
        return f"{version}  ({pct})"
    return f"{version}\n{pct}"


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif", "Times"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": 8,
    })


def plot_figure(
    by_cond_pkg: dict[tuple[str, str], Counter[str]],
    packages: list[str],
    output: Path,
    dpi: int,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.22), sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    row_step = 0.76
    y = np.arange(len(packages)) * row_step
    bar_height = 0.64

    for ax, cond, panel in zip(axes, CONDITIONS, ("(a) MR", "(b) VA")):
        ax.set_facecolor("#FFFFFF")
        n_values = [package_total(by_cond_pkg, cond, package) for package in packages]
        left = np.zeros(len(packages), dtype=float)
        stacked = {0: np.zeros(len(packages)), 1: np.zeros(len(packages)),
                   2: np.zeros(len(packages)), 3: np.zeros(len(packages))}
        labels = {0: [""] * len(packages), 1: [""] * len(packages),
                  2: [""] * len(packages), 3: [""] * len(packages)}

        for i, package in enumerate(packages):
            counter = by_cond_pkg.get((cond, package), Counter())
            total = n_values[i] or 1
            for version, count, rank in segments_for(counter):
                share = count / total
                stacked[rank][i] = share * 100.0
                labels[rank][i] = version

        for rank in (0, 1, 2, 3):
            widths = stacked[rank]
            ax.barh(
                y,
                widths,
                left=left,
                height=bar_height,
                color=RANK_COLORS[rank],
                edgecolor="#FFFFFF",
                linewidth=0.55,
                label=RANK_LEGEND[rank] if ax is axes[0] else None,
            )
            for i, (start, width, version, n) in enumerate(zip(left, widths, labels[rank], n_values)):
                if width <= 0 or n <= 0:
                    continue
                share = width / 100.0
                is_modal = rank == 0
                if not is_modal and (rank == 3 or share < MIN_LABEL_SHARE):
                    continue
                ax.text(
                    start + width / 2.0,
                    y[i],
                    _segment_label(version, share),
                    ha="center",
                    va="center",
                    fontsize=5.6 if share < 0.22 else (6.0 if share < 0.32 else 6.3),
                    color=RANK_TEXT[rank],
                    fontweight="bold" if rank <= 1 else "normal",
                    linespacing=0.92,
                    clip_on=True,
                    zorder=3,
                )
            left = left + widths

        for yi, n in zip(y, n_values):
            ax.text(
                101.4,
                yi,
                f"n={n:,}",
                ha="left",
                va="center",
                fontsize=6.2,
                color="#5A5A5A",
                clip_on=False,
            )

        ax.set_xlim(0, 117)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=7.2)
        ax.set_title("")
        ax.text(
            50,
            1.012,
            panel,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#222222",
            clip_on=False,
        )
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#D8D8D0", linewidth=0.5, linestyle="--", alpha=0.85)
        ax.tick_params(axis="x", length=3.0, color="#8A8A8A", labelsize=7.2, pad=1.2)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#8A8A8A")
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["bottom"].set_bounds(0, 100)
        pad = bar_height / 2.0 + 0.05
        ax.set_ylim(-pad, float(y[-1]) + pad)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(packages, fontsize=7.6)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)

    handles = [
        Patch(facecolor=color, edgecolor="#FFFFFF", linewidth=0.6, label=label)
        for color, label in zip(RANK_COLORS, RANK_LEGEND)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7.6,
        handlelength=1.25,
        handleheight=0.62,
        columnspacing=1.4,
        borderaxespad=0.0,
        bbox_to_anchor=(0.55, 0.058),
        bbox_transform=fig.transFigure,
    )
    fig.subplots_adjust(left=0.178, right=0.968, bottom=0.128, top=0.94, wspace=0.26)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    args = parse_args()
    totals, by_cond_pkg = collect_counts(args.input)
    packages = rank_packages(by_cond_pkg, args.top_n)
    report = sanity_report(totals, by_cond_pkg, packages)
    print(
        f"MR exact declarations: {totals[COND_MR]:,}; "
        f"VA exact declarations: {totals[COND_VA]:,}"
    )
    print("Top packages:", ", ".join(packages))
    print(f"Same modal version under MR and VA: {report['same_modal_count']}/{len(packages)}")
    args.detail_json.parent.mkdir(parents=True, exist_ok=True)
    args.detail_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.detail_json}")
    plot_figure(by_cond_pkg, packages, args.output, args.dpi)


if __name__ == "__main__":
    main()
