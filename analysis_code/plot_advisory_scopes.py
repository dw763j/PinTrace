import os
"""Compact response-level comparison of nested advisory-publication scopes."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def build_advisory_scopes(analysis, output):
    source = Path(analysis) / "security_funnels.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        raw = [r for r in csv.DictReader(handle) if r["model"] == "All observations"]
    records = []
    for mode, kind, condition, form, origin in [
        ("requirements.txt", "exact", "MR", "exact", "model-selected"),
        ("requirements.txt", "range", "MR", "ranged", "resolver-selected"),
        ("inline", "exact", "VA", "exact", "model-selected"),
    ]:
        rows = {r["security_scope"]: r for r in raw if r["mode"] == mode and r["kind"] == kind}
        early, full = rows["published_by_cutoff"], rows["associated"]
        n, cutoff, snapshot = int(early["N"]), int(early["V"]), int(full["V"])
        assert n == int(full["N"]) == 9270 and 0 <= cutoff <= snapshot <= n
        records.append(dict(condition=condition, form=form, release_origin=origin, N=n,
                            cutoff_count=cutoff, later_only_count=snapshot-cutoff, snapshot_count=snapshot,
                            cutoff_pct=100*cutoff/n, later_only_pct=100*(snapshot-cutoff)/n,
                            snapshot_pct=100*snapshot/n))
    style = {"font.family": "DejaVu Sans", "font.size": 9, "text.color": "#20364A",
             "axes.labelcolor": "#20364A", "xtick.color": "#627487", "ytick.color": "#20364A",
             "axes.edgecolor": "#E2E8EE", "axes.spines.top": False,
             "axes.spines.right": False, "axes.spines.left": False,
             "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
             "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.3, 2.7))
        fig.subplots_adjust(left=.23, right=.72, bottom=.20, top=.73)
        y = np.arange(len(records))
        for position in y:
            if position % 2 == 0:
                ax.axhspan(position-.44, position+.44, xmin=0, xmax=1.52, clip_on=False,
                           color="#F4F7FA", linewidth=0, zorder=-3)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E2E8EE", linewidth=.7)
        ax.barh(y, [r["cutoff_pct"] for r in records], height=.42, color="#147D75", zorder=2)
        ax.barh(y, [r["later_only_pct"] for r in records], left=[r["cutoff_pct"] for r in records],
                height=.42, color="#A56120", zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['condition']} {r['form']}\n{r['release_origin']}" for r in records], fontsize=9)
        ax.tick_params(axis="y", pad=7)
        ax.set_ylim(2.5, -.6)
        ax.set_xlim(0, 45)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_xlabel("Responses with an advisory association (%)", fontsize=9, labelpad=4)
        for position, row in enumerate(records):
            ax.text(1.10, position, f"{row['cutoff_pct']:.2f}%", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=9, color="#147D75")
            ax.text(1.40, position-.12, f"{row['snapshot_pct']:.2f}%", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=9.3, fontweight="bold")
            ax.text(1.40, position+.19, f"n={row['snapshot_count']:,}", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=8, color="#627487")
        ax.text(1.10, 1.08, "Cutoff", transform=ax.transAxes, ha="center", fontsize=8.5, color="#627487")
        ax.text(1.40, 1.08, "Snapshot", transform=ax.transAxes, ha="center", fontsize=8.5, color="#627487")
        fig.text(.02, .99, "Clean PinTrace · 9,270 responses per request condition", fontsize=8.5,
                 color="#627487", va="top")
        fig.legend(handles=[Patch(facecolor="#147D75", label="Published by cutoff"),
                            Patch(facecolor="#A56120", label="Added by later publications")],
                   loc="upper left", bbox_to_anchor=(.01, .92), ncol=2, frameon=False,
                   fontsize=8.5, handlelength=1.5, columnspacing=1.8)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(output / f"fig4_temporal_associations.{extension}", dpi=240,
                        bbox_inches="tight", pad_inches=.05, facecolor="white")
        plt.close(fig)
    manifest = dict(source=source.name, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    figsize_inches=[6.3, 2.7], unit="response", rate_denominator=9270,
                    segment_rule="snapshot count minus cutoff-publication count", records=records)
    (output / "fig4_temporal_associations_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    root = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
    result = build_advisory_scopes(root / "analysis", root / "figures")
    print(f"Rendered {len(result['records'])} nested-scope comparisons at response level.")
