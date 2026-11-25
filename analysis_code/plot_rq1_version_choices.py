import os
"""Compact paired distributions of original, model-written exact version strings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def build_version_choices(analysis, output):
    source = Path(analysis) / "rq1_concentration_checked.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    packages = list(dict.fromkeys(row["package"] for row in rows))
    indexed = {(row["package"], row["condition"]): row for row in rows}
    assert len(packages) == 10 and len(indexed) == len(rows) == 20
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = ["#326AA8", "#7FA8CF", "#BBCFE3", "#E5EAF0"]
    styles = {"font.family": "DejaVu Sans", "font.size": 9.5,
              "text.color": "#20364A", "axes.labelcolor": "#627487",
              "xtick.color": "#627487", "ytick.color": "#20364A",
              "axes.edgecolor": "#E2E8EE", "axes.spines.top": False,
              "axes.spines.right": False, "axes.spines.left": False,
              "axes.spines.bottom": False, "xtick.major.size": 0,
              "ytick.major.size": 0, "pdf.fonttype": 42, "svg.fonttype": "none"}
    plotted = []
    with plt.rc_context(styles):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
        fig.subplots_adjust(left=.22, right=.995, top=.88, bottom=.11, wspace=.13)
        for ax, condition in zip(axes, ["MR", "VA"]):
            ax.set_title(condition, fontsize=10.5, fontweight="bold", pad=7)
            for y, package in enumerate(packages):
                row = indexed[(package, condition)]
                n = int(row["N"])
                counts = [int(row[key]) for key in ["rank1", "rank2", "rank3", "other"]]
                assert sum(counts) == n
                assert counts[0] == int(row["modal_count"])
                left = 0
                for count, color in zip(counts, colors):
                    width = 100 * count / n
                    ax.barh(y, width, left=left, height=.34, color=color,
                            edgecolor="white", linewidth=.45, zorder=2)
                    left += width
                modal_share = 100 * counts[0] / n
                top_share = 100 * sum(counts[:3]) / n
                ax.text(modal_share / 2, y, f"{modal_share:.0f}%", ha="center",
                        va="center", fontsize=8.5, color="white", fontweight="bold")
                metadata = (f"mode {row['modal_version']}   "
                            f"n={n:,}   " + r"$\chi_{p,q}^{(3)}$=" + f"{top_share:.1f}%")
                ax.text(0, y + .39, metadata, ha="left", va="center", fontsize=8.2,
                        color="#627487")
                plotted.append(dict(package=package, condition=condition, count=n,
                                    segment_counts=counts, modal_string=row["modal_version"],
                                    modal_pct=modal_share, top_three_pct=top_share))
            ax.set_yticks(np.arange(10))
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.set_ylim(9.75, -.5)
            ax.tick_params(axis="x", labelsize=8.5, pad=3)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E2E8EE", linewidth=.55, zorder=0)
        axes[0].set_yticklabels(packages, fontsize=9.2)
        axes[0].tick_params(axis="y", pad=7)
        axes[1].tick_params(axis="y", labelleft=False)
        fig.legend(handles=[Patch(facecolor=c, label=label) for c, label in
                            zip(colors, ["Most frequent", "Second", "Third", "Other"])],
                   loc="upper center", bbox_to_anchor=(.60, .99), ncol=4,
                   frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.2)
        fig.text(.60, .018, "Exact-advice occurrences (%)", ha="center", fontsize=9.2,
                 color="#627487")
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(output / f"fig2_version_concentration.{extension}", dpi=240,
                        bbox_inches="tight", pad_inches=.04, facecolor="white")
        plt.close(fig)
    manifest = dict(source=source.name, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    layout="Ten shared package rows and paired MR/VA panels",
                    figsize_inches=[7.2, 4.8], records=plotted)
    (output / "fig2_version_concentration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    root = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=root / "analysis")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args()
    result = build_version_choices(args.analysis, args.output)
    print(f"Generated Figure 2 from {len(result['records'])} package-condition distributions.")
