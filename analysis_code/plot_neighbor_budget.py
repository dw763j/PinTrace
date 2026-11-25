import os
"""Compact cumulative and marginal recovery in the recorded candidate order."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
INK, MUTED, GRID = "#20364A", "#627487", "#E2E8EE"
BLUE, TEAL = "#326AA8", "#147D75"


def build_neighbor_budget(analysis, output):
    source = analysis / "neighbor_budget_prefix.csv"
    with source.open(encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["cohort"] == "rematched_eligible"]
    assert len(rows) == 5 and {int(r["N"]) for r in rows} == {815}
    budget = np.array([int(r["prefix_budget"]) for r in rows])
    installed = np.array([int(r["installation_recovered"]) for r in rows])
    static = np.array([int(r["static_recovered"]) for r in rows])
    n = int(rows[0]["N"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "xtick.major.size": 0, "ytick.major.size": 0,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 240,
    })
    fig, (upper, lower) = plt.subplots(2, 1, figsize=(6.9, 3.5),
        gridspec_kw={"height_ratios": [2.25, 1]})
    fig.subplots_adjust(left=0.12, right=0.985, top=0.895, bottom=0.16, hspace=0.26)
    upper.axvspan(30, 40, color="#F0F3F6")
    upper.fill_between(budget, 0, static / n * 100, color="#ECF2F9")
    upper.fill_between(budget, static / n * 100, installed / n * 100, color="#EAF5F2")
    for values, color, marker, label in (
        (installed, TEAL, "o", "Installation recovered"),
        (static, BLUE, "s", "Static pass recovered"),
    ):
        upper.plot(budget, values / n * 100, marker=marker, color=color, lw=2, ms=5, label=label)
    upper.text(39.5, 71, f"{installed[-1]} / {n}  |  {installed[-1] / n * 100:.2f}%",
               ha="right", fontsize=10, color=TEAL, weight="bold")
    upper.text(39.5, 43, f"{static[-1]} / {n}  |  {static[-1] / n * 100:.2f}%",
               ha="right", fontsize=10, color=BLUE, weight="bold")
    tail_label = "No additional\nrecovery" if installed[-1] == installed[-2] and static[-1] == static[-2] else "Later search\nyield"
    upper.text(35, 13, tail_label, ha="center", fontsize=9, color=MUTED)
    upper.set(xlim=(4, 41), ylim=(0, 78), xticks=budget, yticks=[0, 20, 40, 60],
              ylabel="Recovered responses (%)")
    upper.set_axisbelow(True)
    upper.grid(axis="y", color=GRID, lw=0.7)
    positions = np.arange(len(budget))
    windows = [f"1-{budget[0]}"] + [f"{a + 1}-{b}" for a, b in zip(budget[:-1], budget[1:])]
    marginal_installed = np.diff(np.r_[0, installed])
    marginal_static = np.diff(np.r_[0, static])
    for offset, values, color in ((-0.18, marginal_installed, TEAL), (0.18, marginal_static, BLUE)):
        bars = lower.bar(positions + offset, values, 0.32, color=color)
        for bar, value in zip(bars, values):
            lower.annotate(str(int(value)), xy=(bar.get_x() + bar.get_width() / 2, value),
                           xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                           fontsize=9, color=color)
    lower.set_xticks(positions)
    lower.set_xticklabels(windows)
    lower.set_xlabel("Candidate positions in the recorded order")
    lower.set_ylabel("Additional\nrecoveries", fontsize=9)
    lower.set_ylim(0, max(marginal_installed) * 1.3)
    lower.set_yticks([0, 100, 200])
    lower.set_axisbelow(True)
    lower.grid(axis="y", color=GRID, lw=0.7)
    fig.text(0.02, 0.97, f"n = {n}", fontsize=9, color=MUTED, va="top")
    fig.legend(*upper.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.56, 0.995),
               ncol=2, frameon=False, fontsize=9, borderaxespad=0.1)
    output.mkdir(parents=True, exist_ok=True)
    exports = []
    for extension in ("pdf", "svg", "png"):
        path = output / ("fig7_neighbor_budget." + extension)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.055, facecolor="white")
        exports.append(str(path))
    plt.close(fig)
    manifest = {"inputs_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
                "size_inches": [6.9, 3.5], "population": "815 eligible installation-failure responses",
                "prefixes": rows, "outputs": exports,
                "semantics": "Cumulative recovery and additions along the same recorded candidate lists"}
    (output / "fig7_neighbor_budget_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=ROOT / "analysis")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--sync-latex", action="store_true")
    args = parser.parse_args()
    result = build_neighbor_budget(args.analysis, args.output)
    if args.sync_latex:
        for name in result["outputs"]:
            path = Path(name)
            shutil.copyfile(path, ROOT / "latex/figures" / path.name)
    print(json.dumps({"figure": "fig7_neighbor_budget", "size_inches": result["size_inches"],
                      "latex_synchronized": args.sync_latex}))
