"""D2 inline: box plot of PyPI release dates for model-specified TPL versions vs knowledge cutoff.

For each ``m2_extracted_records.json`` under ``outputs/d2/inline/*/<py-tag>/``, we resolve
each pinned ``(library, version)`` to the earliest non-yanked upload time from cached
PyPI JSON (``resources/pypi_info/pypi#<name>.json``), then draw one vertical box per model
on a **reversed** date axis (newer dates toward the bottom). A diamond marks the model's
knowledge cutoff from the paper table.

Output: ``plots/d2_model_tpl_release_date_boxplot.pdf``

Default ``--py-tag`` is ``py312`` (paper experiment layout). Latin sans-serif; no figure title. A small legend in the upper right marks the red diamond as knowledge cutoff. Outliers match each model’s color; cutoff diamonds use a unified red.

Usage (from project root, with .venv activated)::

    python -m plots.d2_model_tpl_release_date_boxplot
    python -m plots.d2_model_tpl_release_date_boxplot --py-tag py314
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paths import PYPI_INFO
from stages.utils import get_pkg_name, load_mapping
from stages.version_resolver import _build_release_time_index, _load_pypi_json_from_cache
from plots.model_display import MODEL_LABEL as PAPER_MODEL_LABEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D2_INLINE_ROOT = PROJECT_ROOT / "outputs" / "d2" / "inline"
OUTPUT_PATH = Path(__file__).resolve().parent / "d2_model_tpl_release_date_boxplot.pdf"

# Knowledge cutoff: YYYY.MM from Table (providers / self-reported where noted).
MODEL_KNOWLEDGE_CUTOFF: dict[str, str] = {
    "gpt-5.4": "2025.08",
    "claude-sonnet-4-6": "2025.05",
    "gemini-3.1-pro-preview": "2025.01",
    "DeepSeek-V3.2": "2025.05",
    "kimi-k2.5": "2024.04",
    "Qwen3.5-397B-A17B": "2026.01",
    "Qwen3-235B-A22B-Instruct-2507": "2024.12",
    "Qwen3-30B-A3B-Instruct-2507": "2024.12",
    "MiniMax-M2.5": "2024.06",
    "meta-llama_llama-4-scout": "2024.08",
}

MODEL_ORDER = list(MODEL_KNOWLEDGE_CUTOFF.keys())

MODEL_LABEL = PAPER_MODEL_LABEL

def _brighten_hex(hex_color: str, toward_white: float) -> str:
    """Mix ``hex_color`` toward white (0 = unchanged, 1 = white)."""
    h = hex_color.strip().lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    t = toward_white
    r = int(r + (255 - r) * t)
    g = int(g + (255 - g) * t)
    b = int(b + (255 - b) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# Base 5-tone palette (green, yellow-brown, rose-brown, blue-grey, purple-grey), slightly brightened
_BASE_BOX = ["#A8C7A3", "#E2C89A", "#CFA39A", "#9EB4C9", "#B7A8C8"]
BOX_PALETTE = [_brighten_hex(c, 0.14) for c in _BASE_BOX]
TEXT_COLOR = "#444444"
# Knowledge-cutoff markers: one color for all models (pure red)
KNOWLEDGE_CUTOFF_COLOR = "#D32F2F"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot TPL version release date distribution per model.")
    p.add_argument("--d2-inline-root", type=Path, default=D2_INLINE_ROOT)
    p.add_argument(
        "--py-tag",
        default="py312",
        help="Python subdir under each run (default: py312, paper experiments).",
    )
    p.add_argument("--pypi-info-dir", type=Path, default=PYPI_INFO)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--output-json", type=Path, default=None, help="Save key statistics to JSON file (optional).")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _short_model_name(run_dir_name: str) -> str:
    name = run_dir_name
    if name.startswith("d2_"):
        name = name[3:]
    for suffix in ("_blind_inline", "_blind_requirements", "_inline", "_requirements"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _parse_cutoff_yyyy_mm(s: str) -> datetime:
    s = s.strip()
    y, m = s.split(".", 1)
    return datetime(int(y), int(m), 1)


def _lookup_release_date(
    import_name: str,
    version: str,
    *,
    mapping: dict,
    pypi_info_dir: str,
    pkg_index_cache: dict[str, dict],
) -> datetime | None:
    if not version or version in {"(unspecified)", ""}:
        return None
    pypi = get_pkg_name(import_name, mapping)
    if pypi not in pkg_index_cache:
        data = _load_pypi_json_from_cache(pypi, pypi_info_dir)
        if not data:
            pkg_index_cache[pypi] = {}
        else:
            rt = _build_release_time_index(data, exclude_yanked=True)
            pkg_index_cache[pypi] = {k: v for k, v in rt.items()}
    idx = pkg_index_cache.get(pypi, {})
    return idx.get(version)


def collect_release_dates_per_model(
    d2_inline_root: Path,
    py_tag: str,
    pypi_info_dir: str,
    mapping: dict,
) -> tuple[list[str], dict[str, list[datetime]], dict[str, int]]:
    pattern = f"*/{py_tag}/m2_extracted_records.json"
    files = sorted(d2_inline_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No m2_extracted_records.json matched {pattern!r} under {d2_inline_root}."
        )

    by_model: dict[str, list[datetime]] = {}
    stats = {"resolved": 0, "unresolved": 0, "skipped_unspecified": 0}
    pkg_cache: dict[str, dict] = {}

    for path in files:
        model = _short_model_name(path.parents[1].name)
        records: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        if model not in by_model:
            by_model[model] = []

        for rec in records:
            for entry in rec.get("extracted_versions", []):
                lib = str(entry.get("library") or "").strip()
                ver = entry.get("version")
                if not lib:
                    continue
                if ver is None:
                    stats["skipped_unspecified"] += 1
                    continue
                ver_str = str(ver).strip()
                if not ver_str:
                    stats["skipped_unspecified"] += 1
                    continue

                dt = _lookup_release_date(lib, ver_str, mapping=mapping, pypi_info_dir=pypi_info_dir, pkg_index_cache=pkg_cache)
                if dt is None:
                    stats["unresolved"] += 1
                    continue
                by_model[model].append(dt)
                stats["resolved"] += 1

    ordered = [m for m in MODEL_ORDER if m in by_model]
    leftovers = sorted(m for m in by_model if m not in MODEL_ORDER)
    # Keep a stable paper-oriented order if possible; otherwise append leftovers.
    model_names = ordered + leftovers
    return model_names, by_model, stats


def collect_versions_beyond_cutoff(
    d2_inline_root: Path,
    py_tag: str,
    pypi_info_dir: str,
    mapping: dict,
) -> dict[str, dict]:
    """Analyze versions that exceed knowledge cutoff, excluding 'latest_at_cutoff' type.
    
    Returns dict with per-model statistics:
    {
        "model_name": {
            "total_pinned": int,  # Total pinned versions
            "latest_at_cutoff": int,  # Count of latest_at_cutoff (excluded from beyond)
            "beyond_cutoff": int,  # Count of versions with release_date > cutoff
            "unresolved": int,  # Count of unresolvable versions
            "beyond_cutoff_ratio": float,  # Percentage of beyond_cutoff / (total - latest_at_cutoff)
            "sample_beyond": list[dict],  # Sample of beyond-cutoff versions with details
        }
    }
    """
    pattern = f"*/{py_tag}/m3_resolved_records.json"
    files = sorted(d2_inline_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No m3_resolved_records.json matched {pattern!r} under {d2_inline_root}."
        )
    
    results = {}
    pkg_cache: dict[str, dict] = {}
    
    for path in files:
        model = _short_model_name(path.parents[1].name)
        records: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        
        # Parse cutoff date
        cutoff_key = _find_matching_cutoff_key(model)
        if not cutoff_key:
            continue
        
        cutoff_date = _parse_cutoff_yyyy_mm(MODEL_KNOWLEDGE_CUTOFF[cutoff_key])
        
        total_pinned = 0
        latest_at_cutoff_count = 0
        beyond_cutoff = []
        unresolved = 0
        
        for rec in records:
            for lib_item in rec.get("per_lib", []):
                specified_ver = lib_item.get("specified_version")
                if not specified_ver or specified_ver == "(unspecified)":
                    continue
                
                total_pinned += 1
                
                # Skip 'latest_at_cutoff' type
                resolution_method = lib_item.get("resolution_method", "")
                if resolution_method == "latest_at_cutoff":
                    latest_at_cutoff_count += 1
                    continue
                
                # Look up release date
                pypi_name = lib_item.get("pypi_name", "")
                if not pypi_name:
                    import_name = lib_item.get("library", "")
                    if import_name:
                        pypi_name = get_pkg_name(import_name, mapping)
                
                if not pypi_name:
                    unresolved += 1
                    continue
                
                dt = _lookup_release_date(
                    pypi_name,
                    specified_ver,
                    mapping=mapping,
                    pypi_info_dir=pypi_info_dir,
                    pkg_index_cache=pkg_cache,
                )
                
                if dt is None:
                    unresolved += 1
                    continue
                
                # Normalize datetime to naive (remove timezone if present)
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                
                # Check if beyond cutoff
                if dt > cutoff_date:
                    beyond_cutoff.append({
                        "library": lib_item.get("library", pypi_name),
                        "specified_version": specified_ver,
                        "release_date": dt.isoformat(),
                        "import_line": lib_item.get("import_line", "")[:100],
                    })
        
        # Calculate ratio (excluding latest_at_cutoff)
        non_latest_count = total_pinned - latest_at_cutoff_count
        beyond_ratio = len(beyond_cutoff) / non_latest_count if non_latest_count > 0 else 0.0
        
        results[model] = {
            "total_pinned": total_pinned,
            "latest_at_cutoff": latest_at_cutoff_count,
            "beyond_cutoff": len(beyond_cutoff),
            "unresolved": unresolved,
            "beyond_cutoff_ratio": round(beyond_ratio, 4),
            "beyond_cutoff_percentage": round(beyond_ratio * 100, 2),
            "sample_beyond": sorted(beyond_cutoff, key=lambda x: x["release_date"], reverse=True)[:10],
        }
    
    return results


def _find_matching_cutoff_key(model_short_name: str) -> str | None:
    """Find knowledge cutoff key that matches the model short name."""
    for key in MODEL_KNOWLEDGE_CUTOFF.keys():
        if key == model_short_name:
            return key
    return None


def plot_boxplot(
    model_names: list[str],
    by_model: dict[str, list[datetime]],
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

    model_names = [m for m in model_names if len(by_model.get(m, [])) > 0]
    if not model_names:
        raise RuntimeError("No models with at least one resolved release date; check PyPI cache and m2 data.")

    n = len(model_names)
    positions = np.arange(1, n + 1)
    data_num: list[list[float]] = []
    cutoffs_num: list[float] = []

    for m in model_names:
        dates = by_model.get(m, [])
        data_num.append([mdates.date2num(d) for d in dates])

        ks = MODEL_KNOWLEDGE_CUTOFF.get(m)
        if ks:
            cutoffs_num.append(mdates.date2num(_parse_cutoff_yyyy_mm(ks)))
        else:
            cutoffs_num.append(np.nan)

    fig_w = max(10.0, n * 0.95)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bp = ax.boxplot(
        data_num,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(BOX_PALETTE[i % len(BOX_PALETTE)])
        patch.set_edgecolor("#FFFFFF")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.94)
    for w in bp["whiskers"]:
        w.set(color="#8E8E8A", linewidth=0.8)
    for cap in bp["caps"]:
        cap.set(color="#8E8E8A", linewidth=0.8)
    for med in bp["medians"]:
        med.set(color=TEXT_COLOR, linewidth=1.2)
    for i, fl in enumerate(bp["fliers"]):
        c = BOX_PALETTE[i % len(BOX_PALETTE)]
        fl.set(
            markerfacecolor=c,
            markeredgecolor=c,
            marker="o",
            markersize=3.2,
            alpha=0.82,
            markeredgewidth=0.35,
        )

    # Knowledge cutoff diamonds (same red for every model)
    for x, y in zip(positions, cutoffs_num):
        if np.isfinite(y):
            ax.scatter(
                [x],
                [y],
                marker="D",
                s=40,
                color=KNOWLEDGE_CUTOFF_COLOR,
                edgecolors="#FFFFFF",
                linewidths=0.55,
                zorder=5,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in model_names], rotation=22, ha="right", fontsize=8.5)
    ax.set_xlabel("Model", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel(
        "Specified TPL version release date (newer toward bottom)",
        fontsize=10,
        color=TEXT_COLOR,
    )
    ax.yaxis_date()
    ax.invert_yaxis()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.55, color="#D0D0C8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)

    cutoff_legend = Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor=KNOWLEDGE_CUTOFF_COLOR,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.55,
        markersize=8,
        linestyle="none",
        label="Knowledge cutoff (red diamond)",
    )
    ax.legend(
        handles=[cutoff_legend],
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        edgecolor="#E0E0DC",
        facecolor="#FFFFFF",
        fontsize=8.5,
    )

    fig.subplots_adjust(bottom=0.22, top=0.96, left=0.09, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PDF: {output}")


def save_statistics_json(
    output_json: Path,
    model_names: list[str],
    by_model: dict[str, list[datetime]],
    stats: dict[str, int],
    beyond_cutoff_stats: dict[str, dict] | None = None,
) -> None:
    """Save key statistics to JSON file."""
    data = {
        "py_tag": "py312",  # Will be set by caller if needed
        "total_models": len(model_names),
        "global_stats": stats,
        "models": {},
    }
    
    for model in model_names:
        dates = by_model.get(model, [])
        if not dates:
            continue
        
        dates_sorted = sorted(dates)
        dates_num = [mdates.date2num(d) for d in dates_sorted]
        
        # Compute summary statistics
        cutoff_str = MODEL_KNOWLEDGE_CUTOFF.get(model, "N/A")
        cutoff_date = None
        if cutoff_str != "N/A":
            try:
                cutoff_date = _parse_cutoff_yyyy_mm(cutoff_str).isoformat()
            except:
                pass
        
        model_data = {
            "count": len(dates),
            "knowledge_cutoff": cutoff_str,
            "knowledge_cutoff_date": cutoff_date,
            "earliest_release": dates_sorted[0].isoformat() if dates_sorted else None,
            "latest_release": dates_sorted[-1].isoformat() if dates_sorted else None,
            "median_release": np.median(dates_num),
            "q1_release": np.percentile(dates_num, 25),
            "q3_release": np.percentile(dates_num, 75),
        }
        # Convert matplotlib numeric dates back to ISO strings
        if np.isfinite(model_data["median_release"]):
            model_data["median_release"] = mdates.num2date(model_data["median_release"]).isoformat()
        if np.isfinite(model_data["q1_release"]):
            model_data["q1_release"] = mdates.num2date(model_data["q1_release"]).isoformat()
        if np.isfinite(model_data["q3_release"]):
            model_data["q3_release"] = mdates.num2date(model_data["q3_release"]).isoformat()
        
        # Add beyond_cutoff statistics if available
        if beyond_cutoff_stats and model in beyond_cutoff_stats:
            model_data["beyond_cutoff_analysis"] = beyond_cutoff_stats[model]
        
        data["models"][model] = model_data
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved statistics JSON: {output_json}")


def main() -> None:
    args = parse_args()
    mapping, _ = load_mapping()
    model_names, by_model, stats = collect_release_dates_per_model(
        args.d2_inline_root,
        args.py_tag,
        str(args.pypi_info_dir),
        mapping,
    )
    print(
        f"Resolved release dates: {stats['resolved']}, "
        f"unresolved_pin: {stats['unresolved']}, "
        f"skipped_unspecified: {stats['skipped_unspecified']}"
    )
    for m in model_names:
        n = len(by_model.get(m, []))
        print(f"  {MODEL_LABEL.get(m, m)}: n={n} pinned versions with PyPI release time")

    plot_boxplot(model_names, by_model, args.output, args.dpi)
    
    # Analyze versions released after each model's knowledge cutoff
    print("\n[ANALYZING VERSIONS BEYOND KNOWLEDGE CUTOFF]")
    beyond_cutoff_stats = collect_versions_beyond_cutoff(
        args.d2_inline_root,
        args.py_tag,
        str(args.pypi_info_dir),
        mapping,
    )
    
    # Print textual summary for inspection
    print("\nPer-model beyond-cutoff analysis (excluding latest_at_cutoff):")
    for model in sorted(beyond_cutoff_stats.keys()):
        stats_dict = beyond_cutoff_stats[model]
        print(f"\n  {MODEL_LABEL.get(model, model)}:")
        print(f"    Total pinned versions: {stats_dict['total_pinned']}")
        print(f"    latest_at_cutoff (excluded): {stats_dict['latest_at_cutoff']}")
        print(f"    Beyond cutoff: {stats_dict['beyond_cutoff']} ({stats_dict['beyond_cutoff_percentage']:.2f}%)")
        print(f"    Unresolved: {stats_dict['unresolved']}")
        if stats_dict['sample_beyond']:
            print(f"    Top samples beyond cutoff:")
            for sample in stats_dict['sample_beyond'][:3]:
                print(f"      - {sample['library']}@@{sample['specified_version']} (released {sample['release_date']})")
    
    # Persist the numeric summary next to the figure unless overridden
    output_json = args.output_json or (args.output.parent / f"{args.output.stem}.json")
    save_statistics_json(output_json, model_names, by_model, stats, beyond_cutoff_stats)


if __name__ == "__main__":
    main()
