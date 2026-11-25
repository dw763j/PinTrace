import os
"""Compact RQ3 displays with explicit populations and stage denominators."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
NAMES = {
    "gpt-5.4": "GPT-5.4", "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "deepseek-v3.2": "DeepSeek V3.2",
    "kimi-k2.5": "Kimi K2.5", "qwen3.5-397b-a17b": "Qwen3.5 397B",
    "qwen3-235b-a22b-instruct-2507": "Qwen3 235B",
    "qwen3-30b-a3b-instruct-2507": "Qwen3 30B", "minimax-m2.5": "MiniMax M2.5",
    "meta-llama/llama-4-scout": "Llama 4 Scout", "All observations": "All observations",
}
INK, MUTED, GRID = "#20364A", "#627487", "#E2E8EE"
BLUE, AMBER, CORAL = "#326AA8", "#A56120", "#BD6257"
COLORS = {"MR": BLUE, "VA": AMBER}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 240,
    })


def load(analysis, name):
    with (analysis / name).open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def row_y(index, count, bottom, height):
    return bottom + height * (1 - (index + 0.5) / count)


def row_band(fig, index, count, bottom, height, color):
    y = bottom + height * (1 - (index + 1) / count)
    fig.add_artist(Rectangle((0.01, y), 0.98, height / count,
                             transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=-10))


def save(fig, output, name, inputs, metadata):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output / (name + "." + extension)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.055, facecolor="white")
        paths.append(str(path))
    report = {"inputs_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "size_inches": list(fig.get_size_inches()), "outputs": paths, **metadata}
    plt.close(fig)
    (output / (name + "_manifest.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_construction_stages(analysis, output):
    style()
    records = load(analysis, "rq3_stage_outcomes.csv")
    records = [r for r in records if r["model"] == "All observations"]
    stages = ["not_entered", "resolution_failure", "installation_failure", "static_diagnostics", "static_pass"]
    labels = ["Did not enter\njoint resolution", "Joint resolution failed", "Installation failed",
              "Static diagnostics", "Static validation\npassed"]
    indexed = {(r["condition"], r["stage"]): r for r in records}
    assert len(indexed) == 10
    for condition in COLORS:
        assert sum(int(indexed[condition, stage]["count"]) for stage in stages) == 10000
    fig = plt.figure(figsize=(7.6, 3.35))
    bottom, height = 0.15, 0.65
    for i in range(5):
        color = "#FCF3E7" if i == 2 else "#ECF2F9" if i == 4 else "#F4F7FA" if i % 2 == 0 else "white"
        row_band(fig, i, 5, bottom, height, color)
    ax = fig.add_axes([0.265, bottom, 0.335, height], facecolor="none")
    y = np.arange(5)
    for condition, offset in (("MR", -0.17), ("VA", 0.17)):
        values = [float(indexed[condition, stage]["population_pct"]) for stage in stages]
        ax.barh(y + offset, values, height=0.27, color=COLORS[condition], zorder=3)
    ax.set(xlim=(0, 60), ylim=(4.5, -0.5), xticks=[0, 20, 40, 60], yticks=[])
    ax.set_xlabel("All responses (%)", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.7)
    fig.text(0.265, 0.94, "(a) Recorded stage", fontsize=11, weight="bold")
    fig.text(0.635, 0.94, "(b) Stage-specific outcomes", fontsize=11, weight="bold")
    fig.text(0.635, 0.885, "n / reached-stage N; percentage", fontsize=9, color=MUTED)
    fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=6, label="MR"),
                        Line2D([0], [0], color=AMBER, lw=6, label="VA")],
               loc="center", bbox_to_anchor=(0.425, 0.865), ncol=2, frameon=False, fontsize=10)
    for condition, x in (("MR", 0.72), ("VA", 0.90)):
        fig.text(x, 0.822, condition, ha="center", color=COLORS[condition], fontsize=10, weight="bold")
    for i, label in enumerate(labels):
        yy = row_y(i, 5, bottom, height)
        weight = "bold" if i in (2, 4) else "normal"
        fig.text(0.018, yy, label, va="center", fontsize=10.2, weight=weight)
        for condition, x in (("MR", 0.72), ("VA", 0.90)):
            row = indexed[condition, stages[i]]
            fig.text(x, yy + 0.017, f"{int(row['count']):,} / {int(row['reached_stage_N']):,}",
                     ha="center", va="center", fontsize=9.1, color=COLORS[condition])
            fig.text(x, yy - 0.031, f"{float(row['conditional_pct']):.2f}%", ha="center", va="center",
                     fontsize=10, color=COLORS[condition], weight=weight)
    return save(fig, output, "fig5_failure_stages", [analysis / "rq3_stage_outcomes.csv"], {
        "population": "Complete PinTrace, 10000 saved responses per condition",
        "panel_a_denominator": "All responses",
        "panel_b_denominators": dict(zip(stages, ["All responses", "Entered joint resolution", "Complete lock", "Installed", "Installed"])),
        "nonentry_is_not_necessarily_failure": True,
    })


def build_execution_outcomes(analysis, output):
    style()
    records = load(analysis, "rq3_execution_outcomes.csv")
    indexed = {(r["model"], r["condition"]): r for r in records}
    models = list(NAMES)
    assert len(indexed) == 22
    fig = plt.figure(figsize=(7.8, 4.4))
    bottom, height = 0.18, 0.64
    for i in range(11):
        row_band(fig, i, 11, bottom, height, "#ECF2F9" if i == 10 else "#F4F7FA" if i % 2 == 0 else "white")
    left = fig.add_axes([0.385, bottom, 0.285, height], facecolor="none")
    right = fig.add_axes([0.745, bottom, 0.235, height], facecolor="none", sharey=left)
    y = np.arange(11)
    for condition, offset, marker in (("MR", -0.17, "o"), ("VA", 0.17, "D")):
        rows = [indexed[model, condition] for model in models]
        nonpass = np.array([float(r["no_environment_among_nonpasses_pct"]) for r in rows])
        overall = np.array([float(r["test_pass_pct"]) for r in rows])
        low = np.array([float(r["test_pass_ci_low"]) for r in rows])
        high = np.array([float(r["test_pass_ci_high"]) for r in rows])
        left.scatter(nonpass, y + offset, color=COLORS[condition], marker=marker, s=23, zorder=3)
        right.errorbar(overall, y + offset, xerr=[overall - low, high - overall], fmt=marker,
                       color=COLORS[condition], ecolor=COLORS[condition], markersize=4.1,
                       capsize=1.9, linewidth=1.05, zorder=3)
    left.set(xlim=(-2, 102), xticks=[0, 25, 50, 75, 100], ylim=(10.5, -0.5), yticks=[])
    right.set(xlim=(0, 60), xticks=[0, 20, 40, 60], yticks=[])
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.tick_params(axis="x", labelsize=9)
        ax.axhline(9.5, color="#CAD4DF", linewidth=0.8)
    left.axvline(50, color=CORAL, lw=1, linestyle=(0, (3, 3)))
    left.text(50, 1.018, "50%", ha="center", transform=left.get_xaxis_transform(), fontsize=8.5, color=CORAL)
    left.set_xlabel("No installed environment (%)", fontsize=9.7)
    right.set_xlabel("Passing task tests (%)", fontsize=9.7)
    fig.text(0.385, 0.95, "(a) Among test nonpasses", fontsize=10.8, weight="bold")
    fig.text(0.385, 0.895, "No installed environment", fontsize=9.3, color=MUTED)
    fig.text(0.745, 0.95, "(b) All responses", fontsize=10.8, weight="bold")
    fig.text(0.745, 0.895, "Test pass, 95% intervals", fontsize=9.3, color=MUTED)
    fig.text(0.018, 0.86, "Model", fontsize=10.4, weight="bold")
    fig.text(0.29, 0.93, "Test nonpasses n", ha="center", fontsize=9.5, weight="bold")
    for condition, x in (("MR", 0.255), ("VA", 0.327)):
        fig.text(x, 0.86, condition, ha="center", fontsize=10, color=COLORS[condition], weight="bold")
    for i, model in enumerate(models):
        yy = row_y(i, 11, bottom, height)
        weight = "bold" if model == "All observations" else "normal"
        fig.text(0.018, yy, NAMES[model], va="center", fontsize=10.1, weight=weight)
        for condition, x in (("MR", 0.255), ("VA", 0.327)):
            fig.text(x, yy, f"{int(indexed[model, condition]['all_test_nonpass']):,}",
                     va="center", ha="center", color=COLORS[condition], fontsize=9.5, weight=weight)
    for condition, offset in (("MR", -0.17), ("VA", 0.17)):
        value = float(indexed["All observations", condition]["no_environment_among_nonpasses_pct"])
        left.text(value + 3, 10 + offset, f"{value:.2f}%", color=COLORS[condition],
                  va="center", fontsize=8.6, weight="bold")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=BLUE, lw=0, label="MR"),
                        Line2D([0], [0], marker="D", color=AMBER, lw=0, label="VA")],
               loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=2, frameon=False, fontsize=10)
    return save(fig, output, "fig6_dynamic_paired_intervals", [analysis / "rq3_execution_outcomes.csv"], {
        "population": "813-task third-party-library BigCodeBench subset",
        "panel_a": "No installed environment among responses without a test pass: (N-installed)/(N-test_pass); descriptive percentages without intervals",
        "panel_b": "Test pass among all 813 responses per model-condition, with precomputed task-bootstrap 95% intervals",
        "aggregate": "Counts summed before rates; panel-a denominators are 4692 MR and 6791 VA nonpasses; N=8130 per condition for panel b",
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=ROOT / "analysis")
    ap.add_argument("--output", type=Path, default=ROOT / "figures")
    ap.add_argument("--figure", choices=("stages", "execution", "both"), default="both")
    ap.add_argument("--sync-latex", action="store_true")
    args = ap.parse_args()
    reports = []
    if args.figure in ("stages", "both"):
        reports.append(build_construction_stages(args.analysis, args.output))
    if args.figure in ("execution", "both"):
        reports.append(build_execution_outcomes(args.analysis, args.output))
    if args.sync_latex:
        for report in reports:
            for name in report["outputs"]:
                source = Path(name)
                shutil.copyfile(source, ROOT / "latex/figures" / source.name)
    print(json.dumps({"figures": len(reports), "output": str(args.output), "latex_synchronized": args.sync_latex}))
