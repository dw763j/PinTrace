import os
"""Compact release-age plot with row bands spanning all numeric columns."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from summarize_mr_constraint_profiles import DISPLAY_NAMES


def build_release_age(analysis, output):
    source = Path(analysis) / "version_age_summary.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    indexed = {row["model"]: row for row in raw}
    assert set(indexed) == set(DISPLAY_NAMES)
    data = []
    for model in DISPLAY_NAMES:
        row = indexed[model]
        point = dict(model=model, n=int(row["n"]))
        point.update({key: float(row[key]) for key in ["median_months", "q25_months", "q75_months", "q05_months", "q95_months"]})
        assert point["q05_months"] <= point["q25_months"] <= point["median_months"] <= point["q75_months"] <= point["q95_months"]
        data.append(point)
    style = {"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": "#20364A",
             "axes.labelcolor": "#20364A", "xtick.color": "#627487", "ytick.color": "#20364A",
             "axes.edgecolor": "#E2E8EE", "axes.spines.top": False,
             "axes.spines.right": False, "axes.spines.left": False,
             "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
             "pdf.fonttype": 42, "svg.fonttype": "none"}
    band_end, median_x, n_x = 1.48, 1.13, 1.43
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.3, 4.3))
        fig.subplots_adjust(left=.30, right=.75, bottom=.17, top=.86)
        y = np.arange(len(data))
        for position in y:
            if position % 2 == 0:
                ax.axhspan(position-.45, position+.45, xmin=0, xmax=band_end,
                           clip_on=False, color="#F4F7FA", linewidth=0, zorder=-3)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E2E8EE", linewidth=.7)
        ax.hlines(y, [r["q05_months"] for r in data], [r["q95_months"] for r in data], color="#91ADBA", linewidth=1.7)
        ax.hlines(y, [r["q25_months"] for r in data], [r["q75_months"] for r in data], color="#147D75", linewidth=7)
        ax.scatter([r["median_months"] for r in data], y, color="white", edgecolor="#147D75", s=28, linewidth=1.3, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([DISPLAY_NAMES[r["model"]] for r in data], fontsize=9.2)
        ax.tick_params(axis="y", pad=5)
        ax.set_ylim(len(data)-.5, -.65)
        ax.set_xlim(0, 72)
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.tick_params(axis="x", labelsize=8.7, pad=3)
        ax.set_xlabel("Age at 1 Feb 2026 (months)", fontsize=9.2, labelpad=4)
        for position, row in enumerate(data):
            ax.text(median_x, position, f"{row['median_months']:.1f}", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=9.7, fontweight="bold", color="#147D75")
            ax.text(n_x, position, f"{row['n']:,}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=9.0, color="#627487")
        ax.text(median_x, 1.035, "Median", transform=ax.transAxes, ha="center", fontsize=9, color="#627487")
        ax.text(n_x, 1.035, "n", transform=ax.transAxes, ha="right", fontsize=9, color="#627487")
        total = sum(r["n"] for r in data)
        fig.text(.02, .98, f"Clean PinTrace · satisfiable exact VA advice · {total:,} occurrences",
                 fontsize=8.5, color="#627487", va="top")
        handles = [Line2D([0], [0], color="#91ADBA", linewidth=1.7, label="5th–95th percentiles"),
                   Line2D([0], [0], color="#147D75", linewidth=7, label="25th–75th percentiles"),
                   Line2D([0], [0], marker="o", color="#147D75", markerfacecolor="white", linewidth=0, label="Median")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.3,
                   handlelength=1.8, columnspacing=1.5, borderaxespad=.3)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(output / f"fig3_release_age.{extension}", dpi=240, bbox_inches="tight", pad_inches=.05, facecolor="white")
        plt.close(fig)
    manifest = dict(source=source.name, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    figsize_inches=[6.3, 4.3], shaded_row_end_axes=band_end,
                    median_column_axes=median_x, n_column_axes=n_x, records=data)
    (output / "fig3_release_age_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    root = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
    result = build_release_age(root / "analysis", root / "figures")
    print(f"Rendered {len(result['records'])} models; row shading spans both numeric columns.")
