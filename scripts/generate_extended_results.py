"""Generate revision RQ2 figures, temporal analyses, and clustered uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

from plots.model_display import normalize_model_key, order_model_keys, paper_display_label
from paths import ANALYSIS_RQ2, EVALUATION, OSV_INDEX, PAPER_RESULTS, PYPI_INFO, RESOURCES, display_path

REFERENCE_TIME = datetime(2026, 2, 1, tzinfo=timezone.utc)
SEVERITY_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN")
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 365.25 / 12
EXPECTED_VALID_EXACT = 14_122
EXPECTED_DATED_EXACT = 14_075
MANUSCRIPT_FIGURE_DIR = PAPER_RESULTS / "figures"

AGE_BOX_FILL = "#A9C7DC"
AGE_BOX_EDGE = "#4C6475"
AGE_MEDIAN = "#163A5F"
CUTOFF_OFFICIAL = "#D55E00"
CUTOFF_SELF = "#7A5195"
CUTOFF_SENS3 = "#9D79AD"
CUTOFF_SENS6 = "#D8C7DF"
GRID_COLOR = "#D8D8D0"
SPINE_COLOR = "#8A8A8A"
TEXT_MUTED = "#5A5A5A"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_dated_age_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            age = row.get("age_at_reference_days")
            release = row.get("release_date")
            if age in ("", None) or release in ("", None):
                continue
            lag = row.get("lag_before_model_cutoff_days")
            row["age_at_reference_days"] = float(age)
            row["lag_before_model_cutoff_days"] = float(lag) if lag not in ("", None) else None
            rows.append(row)
    return rows


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _load_cutoffs(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key).lower(): value for key, value in raw.items()}


def _release_date(pypi_dir: Path, package: str, version: str) -> datetime | None:
    path = pypi_dir / f"pypi#{package.lower().replace('_', '-')}.json"
    if not path.is_file():
        return None
    try:
        files = (json.loads(path.read_text(encoding="utf-8")).get("releases") or {}).get(version) or []
    except (OSError, json.JSONDecodeError):
        return None
    dates = []
    for item in files:
        value = item.get("upload_time_iso_8601") or item.get("upload_time")
        if value:
            try:
                dates.append(_dt(str(value)))
            except ValueError:
                pass
    return min(dates) if dates else None


def _osv_published_index(osv_dir: Path) -> dict[str, datetime]:
    index: dict[str, datetime] = {}
    for path in osv_dir.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            published = _dt(str(item["published"]))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
        for advisory_id in [item.get("id"), *(item.get("aliases") or [])]:
            if advisory_id:
                index[str(advisory_id)] = published
    return index


def _bootstrap_funnels(task_rows: list[dict], clean_ids: set[str], iterations: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output: list[dict] = []
    for track in ("d1", "d2"):
        for mode in ("inline", "requirements.txt"):
            rows = [
                row for row in task_rows
                if row["track"] == track and row["mode"] == mode
                and (track == "d1" or str(row["task_id"]) in clean_ids)
            ]
            by_task: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_task[str(row["task_id"])].append(row)
            task_ids = sorted(by_task)
            for kind in ("exact", "range"):
                metric_values: dict[str, list[float]] = defaultdict(list)
                for _ in range(iterations):
                    sample = [rng.choice(task_ids) for _ in task_ids]
                    sampled_rows = [row for task_id in sample for row in by_task[task_id]]
                    n = len(sampled_rows)
                    s = sum(bool(row[f"has_model_{kind}_constraint"]) for row in sampled_rows)
                    p = sum(bool(row[f"has_valid_model_{kind}_constraint"]) for row in sampled_rows)
                    v = sum(bool(row[f"has_model_{kind}_cve"]) for row in sampled_rows)
                    metric_values["incidence"].append(s / n if n else 0.0)
                    if s:
                        metric_values["validity"].append(p / s)
                    if p:
                        metric_values["conditional_cve"].append(v / p)
                    metric_values["unconditional_cve"].append(v / n if n else 0.0)
                for metric, values in metric_values.items():
                    output.append({
                        "track": track,
                        "mode": mode,
                        "population": "bcb_tpl" if track == "d1" else "clean",
                        "constraint_kind": kind,
                        "metric": metric,
                        "bootstrap_iterations": iterations,
                        "cluster": "task_id",
                        "estimate_mean": sum(values) / len(values),
                        "ci95_low": _percentile(values, 0.025),
                        "ci95_high": _percentile(values, 0.975),
                    })
    return output


def _apply_manuscript_style() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": 8,
    })


def _style_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, color=GRID_COLOR, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=3.0, color=SPINE_COLOR, labelsize=7.2, pad=1.2)
    ax.tick_params(axis="y", length=0, pad=3.0, labelsize=7.6)


def _plot_version_age_figure(output: Path, age_rows: list[dict]) -> str | None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        return None

    models = order_model_keys(normalize_model_key(str(row["model"])) for row in age_rows)
    if not models:
        return None

    ages: list[list[float]] = []
    lags: list[float] = []
    sources: list[str] = []
    counts: list[int] = []
    for model in models:
        selected = [
            row for row in age_rows
            if normalize_model_key(str(row["model"])) == model
            and row.get("release_date")
            and row.get("age_at_reference_days") is not None
        ]
        model_ages = [float(row["age_at_reference_days"]) / DAYS_PER_MONTH for row in selected]
        ages.append(model_ages)
        model_lags = [
            float(row["lag_before_model_cutoff_days"]) / DAYS_PER_MONTH
            for row in selected
            if row.get("lag_before_model_cutoff_days") is not None
        ]
        lags.append(median(model_lags) if model_lags else float("nan"))
        sources.append(str(selected[0].get("cutoff_source") or "") if selected else "")
        counts.append(len(model_ages))
    if sum(counts) != EXPECTED_DATED_EXACT:
        raise SystemExit(
            f"Version-age n labels sum to {sum(counts):,}, expected {EXPECTED_DATED_EXACT:,} dated boxplot rows"
        )
    if any(len(model_ages) != count for model_ages, count in zip(ages, counts)):
        raise SystemExit("Version-age n labels do not match the boxplot series lengths")

    _apply_manuscript_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.55),
        sharey=True,
        gridspec_kw={"width_ratios": [0.58, 0.42], "wspace": 0.10},
    )
    fig.patch.set_facecolor("#FFFFFF")
    y = list(range(len(models)))

    artists = ax_a.boxplot(
        ages,
        positions=y,
        vert=False,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        zorder=3,
    )
    for box in artists["boxes"]:
        box.set(facecolor=AGE_BOX_FILL, edgecolor=AGE_BOX_EDGE, linewidth=0.9)
    for median_line in artists["medians"]:
        median_line.set(color=AGE_MEDIAN, linewidth=1.45)
    for line in [*artists["whiskers"], *artists["caps"]]:
        line.set(color=AGE_BOX_EDGE, linewidth=0.9)

    ax_a.set_xlim(0, 84.6)
    ax_a.set_xticks((0, 12, 24, 36, 48, 60, 72))
    ax_a.spines["bottom"].set_bounds(0, 72)
    ax_a.set_xlabel("Months before the 2026-02-01 experiment cutoff", fontsize=8.0, color="#222222")
    for yi, count in zip(y, counts):
        ax_a.text(
            73.4,
            yi,
            f"n={count:,}",
            ha="left",
            va="center",
            fontsize=6.2,
            color=TEXT_MUTED,
            clip_on=True,
        )
    _style_axes(ax_a)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([paper_display_label(model, model) for model in models], fontsize=7.6)
    ax_a.set_ylim(-0.42, len(models) - 0.58)
    ax_a.invert_yaxis()
    ax_a.set_title("(a) Age at experimental cutoff", loc="left", fontsize=9.0, pad=3, color="#222222")

    ax_b.axvline(0.0, color="#8A8A8A", linewidth=0.7, zorder=1)
    for yi, lag, source in zip(y, lags, sources):
        if source == "self_reported":
            ax_b.barh(
                yi,
                12.0,
                left=lag - 6.0,
                height=0.16,
                color=CUTOFF_SENS6,
                edgecolor="none",
                zorder=2,
            )
            ax_b.barh(
                yi,
                6.0,
                left=lag - 3.0,
                height=0.28,
                color=CUTOFF_SENS3,
                edgecolor="none",
                zorder=3,
            )
            ax_b.plot(
                lag,
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
                lag,
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
            37.2,
            yi,
            f"{lag:.1f}",
            ha="left",
            va="center",
            fontsize=6.2,
            color=TEXT_MUTED,
            clip_on=True,
        )
    ax_b.set_xlim(0, 42.8)
    ax_b.set_xticks((0, 6, 12, 18, 24, 30, 36))
    ax_b.spines["bottom"].set_bounds(0, 36)
    ax_b.set_xlabel("Median months before reported cutoff", fontsize=8.0, color="#222222")
    _style_axes(ax_b)
    ax_b.tick_params(axis="y", labelleft=False)
    ax_b.set_title("(b) Median lag before reported cutoff", loc="left", fontsize=9.0, pad=5, color="#222222")

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=5.6,
            markerfacecolor=CUTOFF_OFFICIAL,
            markeredgecolor=CUTOFF_OFFICIAL,
            label="Official cutoff",
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
            label="Self-reported cutoff",
        ),
        Patch(facecolor=CUTOFF_SENS3, edgecolor="none", label="Window if cutoff shifts \u00b13 mo"),
        Patch(facecolor=CUTOFF_SENS6, edgecolor="none", label="Window if cutoff shifts \u00b16 mo"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handlelength=1.2,
        handleheight=0.58,
        handletextpad=0.4,
        columnspacing=1.05,
        borderaxespad=0.0,
        bbox_to_anchor=(0.55, -0.01),
        bbox_transform=fig.transFigure,
    )
    fig.text(0.5, -0.028, " ", fontsize=8, alpha=0)
    fig.subplots_adjust(left=0.168, right=0.985, top=0.90, bottom=0.16, wspace=0.16)
    path = output / "model_exact_version_age.pdf"
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path.name


def _copy_manuscript_figure(source: Path) -> Path:
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    destination = MANUSCRIPT_FIGURE_DIR / source.name
    destination.write_bytes(source.read_bytes())
    return destination


def _version_age_acceptance(n_primary: int, age_rows: list[dict], age_summary: list[dict]) -> dict:
    n_dated = len(age_rows)
    coverage = n_dated / n_primary if n_primary else 0.0
    ages = [float(row["median_age_years"]) for row in age_summary]
    lags = [float(row["median_lag_before_cutoff_months"]) for row in age_summary if row["median_lag_before_cutoff_months"] is not None]
    min_age_row = min(age_summary, key=lambda row: float(row["median_age_years"]))
    max_age_row = max(age_summary, key=lambda row: float(row["median_age_years"]))
    other_ages = [
        float(row["median_age_years"])
        for row in age_summary
        if row is not min_age_row and row is not max_age_row
    ]
    report = {
        "valid_exact_occurrences": n_primary,
        "occurrences_with_release_date": n_dated,
        "missing_release_dates": n_primary - n_dated,
        "coverage": coverage,
        "median_age_years_range": [min(ages), max(ages)] if ages else None,
        "median_lag_months_range": [min(lags), max(lags)] if lags else None,
        "lowest_median_age": {
            "model": paper_display_label(str(min_age_row["model"]), str(min_age_row["model"])),
            "years": float(min_age_row["median_age_years"]),
        },
        "highest_median_age": {
            "model": paper_display_label(str(max_age_row["model"]), str(max_age_row["model"])),
            "years": float(max_age_row["median_age_years"]),
        },
    }
    checks = {
        "valid_exact": n_primary == EXPECTED_VALID_EXACT,
        "dated": n_dated == EXPECTED_DATED_EXACT,
        "coverage": n_dated == EXPECTED_DATED_EXACT and round(coverage * 100, 2) == 99.67,
        "median_age_range": bool(ages) and round(min(ages), 2) == 1.81 and round(max(ages), 2) == 3.61,
        "median_lag_range": bool(lags) and round(min(lags), 2) == 7.24 and round(max(lags), 2) == 30.36,
        "lowest_model": report["lowest_median_age"]["model"] == "GPT-5.4",
        "highest_model": report["highest_median_age"]["model"] == "Llama-4-Scout",
        "other_medians": all(2.06 <= age <= 2.88 for age in other_ages),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"Version-age acceptance checks failed: {', '.join(failed)}\n{json.dumps(report, indent=2)}")
    report["checks"] = checks
    return report


def _plot_outputs(output: Path, severity_rows: list[dict], age_rows: list[dict], convergence_rows: list[dict], disclosure_rows: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": 9,
    })

    def ordered_models(rows: list[dict]) -> list[str]:
        return order_model_keys(normalize_model_key(str(row["model"])) for row in rows)

    def label(model: str) -> str:
        return paper_display_label(model, model)

    palette = {
        "CRITICAL": "#A44A3F",
        "HIGH": "#C98267",
        "MEDIUM": "#D8B26E",
        "LOW": "#A8C7A3",
        "UNKNOWN": "#9AA3AE",
    }
    generated: list[str] = []
    age_name = _plot_version_age_figure(output, age_rows)
    if age_name:
        generated.append(age_name)

    severity_by_model = defaultdict(Counter)
    for row in severity_rows:
        severity_by_model[normalize_model_key(str(row["model"]))][row["severity"]] += int(row["occurrences"])
    if severity_by_model:
        models = order_model_keys(severity_by_model)
        fig, ax = plt.subplots(figsize=(8.6, 5.3))
        y = np.arange(len(models))
        left = np.zeros(len(models))
        for severity in SEVERITY_ORDER:
            values = np.array([severity_by_model[model][severity] for model in models])
            ax.barh(y, values, left=left, label=severity.title(), color=palette[severity], edgecolor="white", linewidth=0.5)
            left += values
        ax.set_yticks(y, labels=[label(model) for model in models])
        ax.set_xlabel("CVE-associated exact dependency occurrences")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5, color="#D7D7CF")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=5, frameon=False, fontsize=8)
        fig.tight_layout()
        path = output / "model_exact_vulnerability_severity.pdf"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path.name)

    top = convergence_rows[:15]
    if top:
        models = order_model_keys(normalize_model_key(model) for row in top for model in row["model_counts"])
        matrix = []
        for row in top:
            normalized_counts = Counter()
            for model, count in row["model_counts"].items():
                normalized_counts[normalize_model_key(model)] += count
            matrix.append([normalized_counts[model] for model in models])
        fig, ax = plt.subplots(figsize=(9.0, max(5, len(top) * 0.34)))
        image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(models)), labels=[label(model) for model in models], rotation=30, ha="right")
        ax.set_yticks(range(len(top)), labels=[row["pair"] for row in top])
        fig.colorbar(image, ax=ax, label="Exact-advice occurrences")
        fig.tight_layout()
        path = output / "model_exact_vulnerable_convergence.pdf"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path.name)

    central = [row for row in disclosure_rows if row["offset_months"] == 0]
    if central:
        indexed = {normalize_model_key(str(row["model"])): row for row in central}
        models = order_model_keys(indexed)
        values = [indexed[model]["pre_cutoff_rate"] for model in models]
        labels = [f"[{indexed[model]['source'][0].upper()}] {label(model)}" for model in models]
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.barh(labels, values, color="#6E9AAF", edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of unique advisories disclosed before cutoff")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.55, color="#D7D7CF")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.0, -0.12, "[O] official cutoff; [S] self-reported cutoff", transform=ax.transAxes, color="#64748B", fontsize=8)
        fig.tight_layout()
        path = output / "model_exact_cve_disclosure.pdf"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path.name)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=EVALUATION)
    parser.add_argument(
        "--clean-task-ids",
        type=Path,
        default=RESOURCES / "revision" / "pintrace_clean_task_ids.jsonl",
    )
    parser.add_argument("--pypi-info", type=Path, default=PYPI_INFO)
    parser.add_argument("--osv-dir", type=Path, default=OSV_INDEX)
    parser.add_argument(
        "--cutoff-json",
        type=Path,
        default=RESOURCES / "revision" / "model_knowledge_cutoff.json",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS_RQ2 / "unfiltered",
        help="Pre-withdrawn RQ2 CSVs. Official withdrawn-filtered tables stay in outputs/analysis/rq2/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dependencies = _read_jsonl(args.canonical_root / "direct_dependencies.jsonl")
    tasks = _read_jsonl(args.canonical_root / "task_outcomes.jsonl")
    clean_ids = {str(row["task_id"]) for row in _read_jsonl(args.clean_task_ids)}
    cutoffs = _load_cutoffs(args.cutoff_json)

    primary = [
        row for row in dependencies
        if row["track"] == "d2" and row["mode"] == "inline"
        and str(row["task_id"]) in clean_ids
        and row.get("resolution_origin") == "model_exact"
        and row.get("constraint_valid") is True
    ]
    severity_counter = Counter(
        (row["model"], str(row.get("max_severity") or "UNKNOWN"))
        for row in primary if row.get("is_cve_associated")
    )
    severity_rows = [
        {"model": model, "severity": severity, "occurrences": count, "unit": "dependency_occurrence"}
        for (model, severity), count in sorted(severity_counter.items())
    ]

    frozen_age_path = args.output / "version_age_exact.csv"
    frozen_summary_path = args.output / "version_age_exact_summary.csv"
    frozen_age_rows = _load_dated_age_rows(frozen_age_path) if frozen_age_path.is_file() else []
    use_frozen_age = len(frozen_age_rows) == EXPECTED_DATED_EXACT
    age_rows: list[dict] = []
    if use_frozen_age:
        age_rows = frozen_age_rows
    else:
        release_cache: dict[tuple[str, str], datetime | None] = {}
        for row in primary:
            pair = (str(row["pypi_name"]), str(row.get("constraint_resolved_version") or row.get("resolved_version")))
            if pair not in release_cache:
                release_cache[pair] = _release_date(args.pypi_info, *pair)
            release = release_cache[pair]
            if release is None:
                continue
            cutoff = cutoffs.get(str(row["model"]).lower())
            cutoff_date = _dt(cutoff["date"]) if cutoff else None
            age_rows.append({
                "model": row["model"],
                "task_id": row["task_id"],
                "pypi_name": pair[0],
                "version": pair[1],
                "release_date": release.isoformat(),
                "age_at_reference_days": (REFERENCE_TIME - release).total_seconds() / 86400,
                "lag_before_model_cutoff_days": (cutoff_date - release).total_seconds() / 86400 if cutoff_date else None,
                "cutoff_source": cutoff.get("source") if cutoff else None,
            })

    grouped_pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in primary:
        grouped_pairs[(str(row["task_id"]), str(row["pypi_name"]))].append(row)
    pair_summary = []
    for (task_id, package), rows in grouped_pairs.items():
        versions = Counter(str(row.get("constraint_resolved_version") or row.get("resolved_version")) for row in rows)
        pair_summary.append({
            "task_id": task_id,
            "pypi_name": package,
            "recommending_models": len({row["model"] for row in rows}),
            "distinct_versions": len(versions),
            "modal_share": max(versions.values()) / len(rows),
            "unanimous": len(versions) == 1,
        })
    vulnerable_pair_models: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in primary:
        if row.get("is_cve_associated"):
            version = str(row.get("constraint_resolved_version") or row.get("resolved_version"))
            vulnerable_pair_models[(str(row["pypi_name"]), version)][str(row["model"])] += 1
    convergence_rows = [
        {
            "pair": f"{package}=={version}",
            "global_occurrences": sum(counts.values()),
            "models": len(counts),
            "model_counts": dict(counts),
        }
        for (package, version), counts in sorted(
            vulnerable_pair_models.items(), key=lambda item: (-sum(item[1].values()), item[0])
        )
    ]

    osv_published = _osv_published_index(args.osv_dir)
    advisories_by_model: dict[str, dict[str, datetime]] = defaultdict(dict)
    for row in primary:
        if not row.get("is_cve_associated"):
            continue
        for advisory_id in [*(row.get("cve_ids") or []), *(row.get("osv_ids") or [])]:
            advisory_id = str(advisory_id)
            if advisory_id in osv_published:
                advisories_by_model[str(row["model"])][advisory_id] = osv_published[advisory_id]
    disclosure_rows: list[dict] = []
    for model, advisories in sorted(advisories_by_model.items()):
        cutoff = cutoffs.get(model.lower())
        if not cutoff:
            continue
        base = _dt(cutoff["date"])
        offsets = (-6, -3, 0, 3, 6) if cutoff["source"] == "self_reported" else (0,)
        for offset in offsets:
            shifted = base + timedelta(days=offset * 365.25 / 12)
            before = sum(published <= shifted for published in advisories.values())
            disclosure_rows.append({
                "model": model,
                "source": cutoff["source"],
                "offset_months": offset,
                "unique_advisories": len(advisories),
                "pre_cutoff": before,
                "pre_cutoff_rate": before / len(advisories) if advisories else None,
            })

    bootstrap_rows = _bootstrap_funnels(tasks, clean_ids, args.bootstrap_iterations, 20260812)
    age_summary = []
    for model in sorted({row["model"] for row in age_rows}):
        selected = [row for row in age_rows if row["model"] == model]
        ages = [float(row["age_at_reference_days"]) for row in selected]
        lags = [float(row["lag_before_model_cutoff_days"]) for row in selected if row["lag_before_model_cutoff_days"] is not None]
        age_summary.append({
            "model": model,
            "occurrences_with_release_date": len(selected),
            "median_age_years": median(ages) / DAYS_PER_YEAR,
            "median_lag_before_cutoff_months": median(lags) / DAYS_PER_MONTH if lags else None,
        })

    _write_csv(args.output / "vulnerability_severity_exact.csv", severity_rows)
    if not use_frozen_age:
        _write_csv(frozen_age_path, age_rows)
        _write_csv(frozen_summary_path, age_summary)
    _write_csv(args.output / "convergence_exact_task_package.csv", pair_summary)
    _write_csv(
        args.output / "convergence_exact_vulnerable_pairs.csv",
        [{**row, "model_counts": json.dumps(row["model_counts"], sort_keys=True)} for row in convergence_rows],
    )
    _write_csv(args.output / "cve_disclosure_exact.csv", disclosure_rows)
    _write_csv(args.output / "cluster_bootstrap_ci.csv", bootstrap_rows)
    age_acceptance = _version_age_acceptance(len(primary), age_rows, age_summary)
    figures = _plot_outputs(args.output, severity_rows, age_rows, convergence_rows, disclosure_rows)
    manuscript_copy = None
    age_pdf = args.output / "model_exact_version_age.pdf"
    if age_pdf.is_file():
        manuscript_copy = display_path(_copy_manuscript_figure(age_pdf))
    config = {
        "canonical_root": display_path(args.canonical_root),
        "population": "D2 clean inline",
        "origin": "model_exact",
        "reference_time": REFERENCE_TIME.isoformat(),
        "bootstrap_iterations": args.bootstrap_iterations,
        "release_date_coverage": {
            "resolved": len(age_rows),
            "total": len(primary),
            "missing": len(primary) - len(age_rows),
            "coverage": age_acceptance["coverage"],
        },
        "version_age_acceptance": age_acceptance,
        "figures": figures,
        "manuscript_figure": manuscript_copy,
    }
    (args.output / "extended_results_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
