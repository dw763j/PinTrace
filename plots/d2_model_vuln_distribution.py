"""D2 inline models: vulnerable-library/task distribution and risk composition.

Default ``--py-tag`` is ``py312``. Latin sans-serif (DejaVu Sans stack) for PDF figures.

Usage (from project root, with .venv activated):
    python -m plots.d2_model_vuln_distribution
    python -m plots.d2_model_vuln_distribution --py-tag py314
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from paths import OSV_INDEX
from plots.model_display import (
    infer_model_raw_name_from_m4,
    is_excluded_model,
    normalize_model_key,
    order_model_keys,
    paper_display_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D2_INLINE_ROOT = PROJECT_ROOT / "outputs" / "d2" / "inline"
OUTPUT_PATH = Path(__file__).resolve().parent / "d2_model_vuln_distribution.pdf"

VULN_BIN_LABELS = ("0", "1", "2", "3+")
SEV_LABELS = ("Low/None", "Medium", "High/Critical")

VULN_BIN_COLORS = {
    "0": "#D6E0D3",
    "1": "#C4D0B0",
    "2": "#B7BF93",
    "3+": "#9FAA76",
}
SEV_COLORS = {
    "Low/None": "#AFC0A9",
    "Medium": "#CDBD9A",
    "High/Critical": "#BE8674",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vulnerable-library count distribution and risk composition per model."
    )
    parser.add_argument("--d2-inline-root", type=Path, default=D2_INLINE_ROOT)
    parser.add_argument("--osv-index", type=Path, default=OSV_INDEX)
    parser.add_argument("--py-tag", default="py312", help="Python subdirectory tag, e.g. py312 / py314.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path for key statistics (default: same stem as --output).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _normalize_severity(value: str | None) -> str:
    if not value:
        return "Low/None"
    v = str(value).strip().upper()
    if v in {"HIGH", "CRITICAL"}:
        return "High/Critical"
    if v in {"MEDIUM", "MODERATE"}:
        return "Medium"
    return "Low/None"


def _classify_osv_id_type(osv_id: str) -> str:
    v = str(osv_id or "").strip().upper()
    if not v:
        return "UNKNOWN"
    if v.startswith("CVE-"):
        return "CVE"
    if v.startswith("GHSA-"):
        return "GHSA"
    if v.startswith("PYSEC-"):
        return "PYSEC"
    head = v.split("-", 1)[0]
    if head.isalpha():
        return head
    return "OTHER"


def collect_model_stats(
    d2_inline_root: Path, py_tag: str
) -> tuple[list[str], dict[str, str], dict[str, dict[str, int]], dict[str, dict[str, int]], Counter, Counter]:
    """Return per-model distributions and global OSV severity (dedup by unique id)."""
    pattern = f"*/{py_tag}/m4_vuln_records.json"
    files = sorted(d2_inline_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched {pattern!r} under {d2_inline_root}. "
            "Run vulnerability stage first or switch --py-tag."
        )

    model_raw_by_key: dict[str, str] = {}
    vuln_bins_by_model: dict[str, dict[str, int]] = {}
    severity_by_model: dict[str, dict[str, int]] = {}
    osv_to_sev: dict[str, str] = {}
    unique_osv_ids: set[str] = set()

    for path in files:
        raw_model_name = infer_model_raw_name_from_m4(path)
        model = normalize_model_key(raw_model_name)
        if is_excluded_model(model, raw_model_name):
            continue

        model_raw_by_key.setdefault(model, raw_model_name)
        records: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        task_bin_counter = Counter({k: 0 for k in VULN_BIN_LABELS})
        sev_counter = Counter({k: 0 for k in SEV_LABELS})

        for rec in records:
            vuln_lib_keys: set[tuple[str, str]] = set()
            findings = rec.get("vuln_findings", []) or []
            for finding in findings:
                if not finding.get("is_vulnerable", False):
                    continue
                pypi_name = str(finding.get("pypi_name") or "").strip()
                version = str(finding.get("version") or "").strip()
                if pypi_name and version:
                    vuln_lib_keys.add((pypi_name, version))

                sev_group = _normalize_severity(finding.get("max_severity"))
                sev_counter[sev_group] += 1

                for osv_id in finding.get("osv_ids", []) or []:
                    osv_key = str(osv_id).strip()
                    if not osv_key:
                        continue
                    unique_osv_ids.add(osv_key)
                    prev = osv_to_sev.get(osv_key)
                    # Keep the highest bucket seen for this unique ID.
                    if prev == "High/Critical":
                        continue
                    if prev == "Medium" and sev_group == "Low/None":
                        continue
                    osv_to_sev[osv_key] = sev_group

            vuln_lib_cnt = len(vuln_lib_keys)
            if vuln_lib_cnt == 0:
                task_bin_counter["0"] += 1
            elif vuln_lib_cnt == 1:
                task_bin_counter["1"] += 1
            elif vuln_lib_cnt == 2:
                task_bin_counter["2"] += 1
            else:
                task_bin_counter["3+"] += 1

        existing_bins = vuln_bins_by_model.setdefault(model, {k: 0 for k in VULN_BIN_LABELS})
        existing_sev = severity_by_model.setdefault(model, {k: 0 for k in SEV_LABELS})
        for k in VULN_BIN_LABELS:
            existing_bins[k] += int(task_bin_counter[k])
        for k in SEV_LABELS:
            existing_sev[k] += int(sev_counter[k])

    model_names = order_model_keys(vuln_bins_by_model.keys())

    # Keep dicts aligned to ordered keys only.
    vuln_bins_by_model = {m: vuln_bins_by_model[m] for m in model_names}
    severity_by_model = {m: severity_by_model[m] for m in model_names}

    # Persist resolved raw names for display fallback.
    for mk in model_names:
        model_raw_by_key.setdefault(mk, mk)

    global_osv_sev = Counter(osv_to_sev.values())
    global_osv_type = Counter(_classify_osv_id_type(v) for v in unique_osv_ids)
    return model_names, model_raw_by_key, vuln_bins_by_model, severity_by_model, global_osv_sev, global_osv_type


def _severity_from_osv_doc(doc: dict) -> str:
    """Extract severity bucket from an OSV record."""
    db_sev = (doc.get("database_specific") or {}).get("severity")
    if isinstance(db_sev, str) and db_sev.strip():
        return _normalize_severity(db_sev)

    entries = doc.get("severity") or []
    if isinstance(entries, list):
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            score = str(ent.get("score") or "").upper()
            if "CVSS" in score:
                # Use common CVSS textual fallback when present in score string.
                if "CRITICAL" in score or "HIGH" in score:
                    return "High/Critical"
                if "MEDIUM" in score:
                    return "Medium"
    return "Low/None"


def collect_osv_index_stats(osv_index: Path) -> dict[str, object]:
    """Collect full OSV_INDEX statistics from local JSON files."""
    files = sorted(osv_index.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json files found under OSV index path: {osv_index}")

    type_count = Counter()
    severity_count = Counter({k: 0 for k in SEV_LABELS})
    parse_error_files = 0
    excluded_mal_records = 0

    for fp in files:
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            parse_error_files += 1
            continue

        osv_id = str(doc.get("id") or fp.stem).strip()
        # Exclude malware advisory records from full-dataset vuln summary.
        if osv_id.upper().startswith("MAL-"):
            excluded_mal_records += 1
            continue

        type_count[_classify_osv_id_type(osv_id)] += 1
        severity_count[_severity_from_osv_doc(doc)] += 1

    total = sum(type_count.values())
    severity_percent = {
        k: round((severity_count.get(k, 0) / total * 100.0), 2) if total else 0.0
        for k in SEV_LABELS
    }

    return {
        "osv_index_path": str(osv_index),
        "total_osv_files": len(files),
        "excluded_mal_records": excluded_mal_records,
        "parsed_records": total,
        "parse_error_files": parse_error_files,
        "severity_count": dict(severity_count),
        "severity_percent": severity_percent,
        "type_count": dict(sorted(type_count.items(), key=lambda x: (-x[1], x[0]))),
    }


def _to_row_percentages(names: list[str], raw: dict[str, dict[str, int]], labels: tuple[str, ...]) -> np.ndarray:
    mat = np.array([[raw[m][lb] for lb in labels] for m in names], dtype=float)
    totals = mat.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return mat / totals * 100.0


def plot_joint_figure(
    model_names: list[str],
    model_raw_by_key: dict[str, str],
    vuln_bins_by_model: dict[str, dict[str, int]],
    severity_by_model: dict[str, dict[str, int]],
    output: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans", "sans-serif"],
            "axes.unicode_minus": False,
        }
    )
    y = np.arange(len(model_names))
    left_pct = _to_row_percentages(model_names, vuln_bins_by_model, VULN_BIN_LABELS)
    right_pct = _to_row_percentages(model_names, severity_by_model, SEV_LABELS)
    left_cnt = np.array([[vuln_bins_by_model[m][lb] for lb in VULN_BIN_LABELS] for m in model_names], dtype=int)
    right_cnt = np.array([[severity_by_model[m][lb] for lb in SEV_LABELS] for m in model_names], dtype=int)
    display_names = [paper_display_label(m, model_raw_by_key.get(m, m)) for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, max(5.8, len(model_names) * 0.52)), sharey=True)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")

    # Left: per-task vulnerable-library count distribution.
    ax = axes[0]
    left_base = np.zeros(len(model_names))
    for i, lb in enumerate(VULN_BIN_LABELS):
        vals = left_pct[:, i]
        cnts = left_cnt[:, i]
        ax.barh(
            y,
            vals,
            left=left_base,
            color=VULN_BIN_COLORS[lb],
            edgecolor="#FFFFFF",
            linewidth=0.6,
            label=f"{lb} vulnerable libs",
        )
        for yi, (start, width, cnt) in enumerate(zip(left_base, vals, cnts)):
            if width <= 0 or cnt <= 0:
                continue
            ax.text(start + width / 2.0, yi, str(int(cnt)), ha="center", va="center", fontsize=7, color="#2F2F2F")
        left_base += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("(a) Number of vulnerable libraries share by task (%)")
    ax.set_ylabel("Model")
    ax.set_yticks(y)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.55, color="#D7D7CF")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: vulnerable-version severity composition.
    ax2 = axes[1]
    right_base = np.zeros(len(model_names))
    for i, lb in enumerate(SEV_LABELS):
        vals = right_pct[:, i]
        cnts = right_cnt[:, i]
        ax2.barh(
            y,
            vals,
            left=right_base,
            color=SEV_COLORS[lb],
            edgecolor="#FFFFFF",
            linewidth=0.6,
            label=lb,
        )
        for yi, (start, width, cnt) in enumerate(zip(right_base, vals, cnts)):
            if width <= 0 or cnt <= 0:
                continue
            ax2.text(start + width / 2.0, yi, str(int(cnt)), ha="center", va="center", fontsize=7, color="#2F2F2F")
        right_base += vals
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("(b) Vulnerable-version severity share by library (%)")
    ax2.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.55, color="#D7D7CF")
    ax2.spines[["top", "right"]].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.1,
        handlelength=1.4,
    )
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.4,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.2, top=0.86, wspace=0.11)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PDF: {output}")


def print_inline_report(
    model_names: list[str],
    vuln_bins_by_model: dict[str, dict[str, int]],
    severity_by_model: dict[str, dict[str, int]],
    global_osv_sev: Counter,
    global_osv_type: Counter,
    osv_index_stats: dict[str, object],
) -> None:
    print("\n[Inline Results] Model security sensitivity snapshot")
    for model in model_names:
        bins = vuln_bins_by_model[model]
        total_tasks = sum(bins.values()) or 1
        exposed = bins["1"] + bins["2"] + bins["3+"]
        exposed_pct = exposed / total_tasks * 100

        sev = severity_by_model[model]
        sev_total = sum(sev.values()) or 1
        med_high_pct = (sev["Medium"] + sev["High/Critical"]) / sev_total * 100
        print(
            f"- {model}: exposed_tasks={exposed}/{total_tasks} ({exposed_pct:.1f}%), "
            f"task_bins[0/1/2/3+]={bins['0']}/{bins['1']}/{bins['2']}/{bins['3+']}, "
            f"med+high_severity_share={med_high_pct:.1f}%"
        )

    total_unique_osv = sum(global_osv_sev.values()) or 1
    low = global_osv_sev.get("Low/None", 0)
    med = global_osv_sev.get("Medium", 0)
    high = global_osv_sev.get("High/Critical", 0)
    print(
        "\n[Dataset-level unique-OSV severity ratio] "
        f"unique_osv_ids={total_unique_osv}, "
        f"Low/None={low / total_unique_osv * 100:.2f}%, "
        f"Medium={med / total_unique_osv * 100:.2f}%, "
        f"High/Critical={high / total_unique_osv * 100:.2f}%"
    )

    print("\n[Dataset-level unique-OSV type counts]")
    for t, c in sorted(global_osv_type.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {t}: {c}")

    print("\n[OSV_INDEX full-dataset stats]")
    print(
        f"- total_osv_files={osv_index_stats['total_osv_files']}, "
        f"excluded_mal_records={osv_index_stats['excluded_mal_records']}, "
        f"parsed_records={osv_index_stats['parsed_records']}, "
        f"parse_error_files={osv_index_stats['parse_error_files']}"
    )
    print("- type_count:")
    for t, c in (osv_index_stats.get("type_count") or {}).items():
        print(f"  - {t}: {c}")


def save_statistics_json(
    output_json: Path,
    model_names: list[str],
    model_raw_by_key: dict[str, str],
    vuln_bins_by_model: dict[str, dict[str, int]],
    severity_by_model: dict[str, dict[str, int]],
    global_osv_sev: Counter,
    global_osv_type: Counter,
    osv_index_stats: dict[str, object],
    py_tag: str,
) -> None:
    data: dict[str, object] = {
        "py_tag": py_tag,
        "total_models": len(model_names),
        "models": {},
    }

    model_payload: dict[str, dict[str, object]] = {}
    for model in model_names:
        bins = vuln_bins_by_model[model]
        total_tasks = sum(bins.values())
        exposed = bins["1"] + bins["2"] + bins["3+"]
        sev = severity_by_model[model]
        sev_total = sum(sev.values())

        model_payload[model] = {
            "display_name": paper_display_label(model, model_raw_by_key.get(model, model)),
            "task_bins_count": dict(bins),
            "task_bins_percent": {
                k: round((bins[k] / total_tasks * 100.0), 2) if total_tasks else 0.0
                for k in VULN_BIN_LABELS
            },
            "exposed_tasks": exposed,
            "total_tasks": total_tasks,
            "exposed_tasks_percent": round((exposed / total_tasks * 100.0), 2) if total_tasks else 0.0,
            "severity_count": dict(sev),
            "severity_percent": {
                k: round((sev[k] / sev_total * 100.0), 2) if sev_total else 0.0
                for k in SEV_LABELS
            },
        }

    total_unique_osv = sum(global_osv_sev.values())
    data["models"] = model_payload
    data["dataset_unique_osv"] = {
        "total_unique_osv_ids": total_unique_osv,
        "severity_count": dict(global_osv_sev),
        "severity_percent": {
            k: round((global_osv_sev.get(k, 0) / total_unique_osv * 100.0), 2) if total_unique_osv else 0.0
            for k in SEV_LABELS
        },
        "type_count": dict(sorted(global_osv_type.items(), key=lambda x: (-x[1], x[0]))),
    }
    data["osv_index_full"] = osv_index_stats

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON: {output_json}")


def main() -> None:
    args = parse_args()
    model_names, model_raw_by_key, vuln_bins_by_model, severity_by_model, global_osv_sev, global_osv_type = collect_model_stats(
        args.d2_inline_root, args.py_tag
    )
    osv_index_stats = collect_osv_index_stats(args.osv_index)
    plot_joint_figure(
        model_names=model_names,
        model_raw_by_key=model_raw_by_key,
        vuln_bins_by_model=vuln_bins_by_model,
        severity_by_model=severity_by_model,
        output=args.output,
        dpi=args.dpi,
    )
    print_inline_report(
        model_names,
        vuln_bins_by_model,
        severity_by_model,
        global_osv_sev,
        global_osv_type,
        osv_index_stats,
    )

    output_json = args.output_json or (args.output.parent / f"{args.output.stem}.json")
    save_statistics_json(
        output_json=output_json,
        model_names=model_names,
        model_raw_by_key=model_raw_by_key,
        vuln_bins_by_model=vuln_bins_by_model,
        severity_by_model=severity_by_model,
        global_osv_sev=global_osv_sev,
        global_osv_type=global_osv_type,
        osv_index_stats=osv_index_stats,
        py_tag=args.py_tag,
    )


if __name__ == "__main__":
    main()
