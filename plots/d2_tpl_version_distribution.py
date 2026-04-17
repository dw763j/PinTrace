"""D2 dataset — TPL version specification distribution across all models (inline mode).

For every ``m2_extracted_records.json`` found under ``outputs/d2/inline/**/py38/``,
the script counts how many times each (library, version) pair was specified by any
model.  The result is a stacked bar chart where:

- X-axis: top-N TPLs sorted by total specification count (all models combined).
- Y-axis: total number of times the TPL was specified (stacked by version).
- Each colour band = one version; within each bar, one hue is lightened from dark (largest
  count in that library) to pale (smallest); annotation = percentage within the bar.
- Top of each bar: total count.

Output:
- ``plots/d2_tpl_version_distribution.pdf``
- ``plots/d2_tpl_version_distribution.json``

Usage (from project root, with .venv activated)::

    python -m plots.d2_tpl_version_distribution --specified-only
    python -m plots.d2_tpl_version_distribution --top-n 10 --py-tag py312 --specified-only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# mpl.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
#     "mathtext.fontset": "stix",
# })

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D2_INLINE_ROOT = PROJECT_ROOT / "outputs" / "d2" / "inline"
OUTPUT_PATH = Path(__file__).resolve().parent / "d2_tpl_version_distribution.pdf"
DETAIL_JSON_PATH = Path(__file__).resolve().parent / "d2_tpl_version_distribution.json"


_MORANDI = [
    "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD", "#CD5582", "#E1BE6A", "#E89A7A", "#EC6B2D",
]


# Per-column ramp: fixed hue, wide value range; keep saturation so pale bands do not wash out
_HSV_V_DARK = 0.75
_HSV_V_LIGHT = 0.95
# Gamma on u∈[0,1] to exaggerate extremes (and spread mid tones)
_HSV_U_GAMMA = 0.05
_HSV_S_DARK_BOOST = 1.12
_HSV_S_LIGHT_FLOOR = 0.05


def _fill_rgb_for_rank_u(rgb_base: tuple[float, float, float], u: float) -> tuple[float, float, float]:
    """Map rank parameter u∈[0,1] to fill colors; u=0 is darkest (largest count), u=1 lightest."""
    u = max(0.0, min(1.0, u))
    u = u**_HSV_U_GAMMA
    h, s, _v = mcolors.rgb_to_hsv(np.asarray(rgb_base, dtype=float).reshape(1, 1, 3))[0, 0]
    # u=0 darkest (largest slice), u=1 lightest
    v_out = _HSV_V_DARK + u * (_HSV_V_LIGHT - _HSV_V_DARK)
    s_dark = min(1.0, float(s) * _HSV_S_DARK_BOOST + 0.06)
    s_light = max(_HSV_S_LIGHT_FLOOR, float(s) * 0.88)
    s_out = s_dark + u * (s_light - s_dark)
    rgb = mcolors.hsv_to_rgb(np.array([[[float(h), s_out, v_out]]], dtype=float))[0, 0]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _lib_version_rank_by_count(
    lib: str,
    lib_versions: dict[str, Counter],
    kept_versions: list[str],
    kept_set: set[str],
) -> dict[str, int]:
    """Per-library version ranks by descending count: rank 0 = deepest color."""
    pairs: list[tuple[str, int]] = []
    for v in kept_versions:
        c = lib_versions[lib].get(v, 0)
        if c > 0:
            pairs.append((v, c))
    other_sum = sum(cnt for v, cnt in lib_versions[lib].items() if v not in kept_set)
    if other_sum > 0:
        pairs.append(("(other)", other_sum))
    pairs.sort(key=lambda x: -x[1])
    return {v: r for r, (v, _) in enumerate(pairs)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot D2 TPL version distribution.")
    p.add_argument("--d2-inline-root", type=Path, default=D2_INLINE_ROOT)
    p.add_argument("--py-tag", default="py312", help="Python-version subdirectory (e.g. py38, py312).")
    p.add_argument("--specified-only", action="store_true", help="Only count entries that explicitly specify a version.")
    p.add_argument("--top-n", type=int, default=10, help="Number of top TPLs to show.")
    p.add_argument("--min-count", type=int, default=2, help="Minimum total count to include a version band.")
    p.add_argument("--legend-top-k", type=int, default=15, help="Show only the top-K most-frequent versions in the legend; the rest appear in the chart but are unlabelled.")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--detail-json", type=Path, default=DETAIL_JSON_PATH, help="Path to save per-TPL version-count details as JSON.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def save_detail_json(
    lib_versions: dict[str, Counter],
    *,
    d2_inline_root: Path,
    py_tag: str,
    specified_only: bool,
    top_n: int,
    path: Path,
) -> None:
    totals = {lib: sum(counter.values()) for lib, counter in lib_versions.items()}
    payload = {
        "d2_inline_root": str(d2_inline_root.resolve()),
        "py_tag": py_tag,
        "specified_only": specified_only,
        "top_n_for_plot": top_n,
        "library_count": len(lib_versions),
        "libraries": {
            lib: {
                "total": totals[lib],
                "versions": dict(sorted(lib_versions[lib].items(), key=lambda kv: (-kv[1], kv[0]))),
            }
            for lib in sorted(lib_versions.keys(), key=lambda name: (-totals[name], name))
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def collect_version_counts(
    d2_inline_root: Path,
    py_tag: str,
    specified_only: bool,
) -> dict[str, Counter]:
    """Return {library: Counter({version_str: count})}."""
    lib_versions: dict[str, Counter] = defaultdict(Counter)
    pattern = f"*/{py_tag}/m2_extracted_records.json"
    files = sorted(d2_inline_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No m2_extracted_records.json found under {d2_inline_root} "
            f"matching pattern '{pattern}'.\n"
            "Try a different --py-tag."
        )
    for path in files:
        records: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            for entry in rec.get("extracted_versions", []):
                lib = str(entry.get("library") or "").strip()
                ver = entry.get("version")
                if not lib:
                    continue
                ver_str = str(ver).strip() if ver is not None else ""
                if specified_only and not ver_str:
                    continue
                if not ver_str:
                    ver_str = "(unspecified)"
                lib_versions[lib][ver_str] += 1
    return dict(lib_versions)


def select_top_libs(
    lib_versions: dict[str, Counter],
    top_n: int,
) -> list[str]:
    totals = {lib: sum(ctr.values()) for lib, ctr in lib_versions.items()}
    return [lib for lib, _ in sorted(totals.items(), key=lambda x: -x[1])[:top_n]]


def _shorten_version(v: str, max_len: int = 9) -> str:
    return v if len(v) <= max_len else v[:max_len - 1] + "…"


def _concentration_stats(lib_versions: dict[str, Counter], top_libs: list[str]) -> dict:
    """Compute avg top-1 and top-3 version share across top_libs."""
    top1_shares, top3_shares = [], []
    for lib in top_libs:
        ctr = lib_versions[lib]
        total = sum(ctr.values())
        if total == 0:
            continue
        sorted_counts = sorted(ctr.values(), reverse=True)
        top1_shares.append(sorted_counts[0] / total)
        top3_shares.append(sum(sorted_counts[:3]) / total)
    return {
        "n_models": None,   # filled by caller
        "top1_avg": sum(top1_shares) / len(top1_shares) if top1_shares else 0,
        "top3_avg": sum(top3_shares) / len(top3_shares) if top3_shares else 0,
    }


def plot(
    lib_versions: dict[str, Counter],
    top_libs: list[str],
    *,
    min_count: int,
    legend_top_k: int,
    n_models: int,
    py_tag: str,
    output: Path,
    dpi: int,
) -> None:
    n = len(top_libs)
    totals = [sum(lib_versions[lib].values()) for lib in top_libs]

    # Collect global version order (most-frequent overall), one colour per version
    global_version_totals: Counter = Counter()
    for lib in top_libs:
        for ver, cnt in lib_versions[lib].items():
            global_version_totals[ver] += cnt

    all_versions: list[str] = [v for v, _ in global_version_totals.most_common()]
    # Versions below min_count across all selected libs → fold into "(other)"
    kept_versions: list[str] = []
    for v in all_versions:
        total_in_top = sum(lib_versions[lib].get(v, 0) for lib in top_libs)
        if total_in_top >= min_count:
            kept_versions.append(v)
    kept_set = set(kept_versions)

    # One Morandi base hue per column; within column darker = higher count
    col_base_rgb = [mcolors.to_rgb(_MORANDI[i % len(_MORANDI)]) for i in range(n)]
    rank_by_lib: dict[str, dict[str, int]] = {
        lib: _lib_version_rank_by_count(lib, lib_versions, kept_versions, kept_set)
        for lib in top_libs
    }
    n_seg_by_lib: dict[str, int] = {
        lib: max(len(rank_by_lib[lib]), 1) for lib in top_libs
    }

    # Only the top-K most-frequent versions will appear in the legend
    legend_versions: set[str] = set(kept_versions[:legend_top_k]) | {"(other)"}

    fig_w = max(12, n * 0.82)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(n)
    bottoms = np.zeros(n)

    # Plot each version band
    drawn_labels: set[str] = set()
    for ver in kept_versions + ["(other)"]:
        heights = np.array([
            lib_versions[lib].get(ver, 0) if ver != "(other)"
            else sum(lib_versions[lib].get(v, 0) for v in lib_versions[lib] if v not in kept_set)
            for lib in top_libs
        ])
        if heights.sum() == 0:
            continue
        label = _shorten_version(ver) if ver != "(other)" else "(other)"
        in_legend = ver in legend_versions and label not in drawn_labels
        bar_colors: list[tuple[float, float, float]] = []
        for i, lib in enumerate(top_libs):
            h = heights[i]
            if h <= 0:
                bar_colors.append((1.0, 1.0, 1.0))
                continue
            rnk = rank_by_lib[lib][ver]
            ns = n_seg_by_lib[lib]
            u = rnk / (ns - 1) if ns > 1 else 0.0
            bar_colors.append(_fill_rgb_for_rank_u(col_base_rgb[i], u))
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=bar_colors,
            edgecolor="#FFFFFF",
            linewidth=0.4,
            label=label if in_legend else "_nolegend_",
        )
        if in_legend:
            drawn_labels.add(label)

        # Annotate percentage inside band only when the band is tall enough to fit
        # the text without overlapping: require both a minimum % share AND a minimum
        # absolute height (expressed as a fraction of the tallest bar).
        min_abs_fraction = 0.02   # band must be ≥ 3.5 % of the tallest bar's height
        min_pct_share   = 1    # band must be ≥ 5 % of its own TPL total
        for i, (h, b, tot) in enumerate(zip(heights, bottoms, totals)):
            if h == 0 or tot == 0:
                continue
            pct = h / tot * 100
            abs_fraction = h / max(totals)
            if pct >= min_pct_share and abs_fraction >= min_abs_fraction:
                mid = b + h / 2
                if ver == "(other)":
                    label_text = f"(other) ({pct:.0f}%)"
                else:
                    label_text = f"{_shorten_version(ver, max_len=14)} ({pct:.0f}%)"
                ax.text(
                    x[i], mid, label_text,
                    ha="center", va="center",
                    fontsize=5.5, color="#3A3A3A",
                    fontweight="normal",
                )
        bottoms = bottoms + heights

    # Total count labels on top of each bar
    for i, (tot, bot) in enumerate(zip(totals, bottoms)):
        ax.text(
            x[i], bot + max(totals) * 0.008, str(tot),
            ha="center", va="bottom",
            fontsize=7.5, color="#444444", fontweight="bold",
        )

    # Axes styling
    ax.set_xticks(x)
    ax.set_xticklabels(top_libs, rotation=0, ha="center", fontsize=8.5)
    ax.set_ylabel("Total version specification count", fontsize=10)
    ax.set_xlabel("Third-party libraries", fontsize=10)
    # ax.set_title(
    #     "Version Specification Distribution of Top TPLs\n"
    #     "across All Models — D2 Dataset (inline mode)",
    #     fontsize=11, pad=12,
    # )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8D8D0", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(totals) * 1.12)

    # # Legend (outside on right, compact)
    # handles, labels_ = ax.get_legend_handles_labels()
    # if handles:
    #     legend = ax.legend(
    #         handles, labels_,
    #         title="Version", title_fontsize=8,
    #         fontsize=7, ncol=2,
    #         loc="upper left", bbox_to_anchor=(1.01, 1),
    #         frameon=True, framealpha=0.9,
    #         edgecolor="#CCCCCC",
    #     )

    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.22, top=0.91)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    args = parse_args()
    print(f"Scanning: {args.d2_inline_root} (py_tag={args.py_tag}) …")
    lib_versions = collect_version_counts(args.d2_inline_root, args.py_tag, args.specified_only)
    pattern = f"*/{args.py_tag}/m2_extracted_records.json"
    n_models = len(sorted(args.d2_inline_root.glob(pattern)))
    print(f"Found {len(lib_versions)} distinct libraries across {n_models} models.")
    top_libs = select_top_libs(lib_versions, args.top_n)
    print(f"Top-{args.top_n} TPLs: {top_libs} …")
    save_detail_json(
        lib_versions,
        d2_inline_root=args.d2_inline_root,
        py_tag=args.py_tag,
        specified_only=args.specified_only,
        top_n=args.top_n,
        path=args.detail_json,
    )
    plot(
        lib_versions, top_libs,
        min_count=args.min_count,
        legend_top_k=args.legend_top_k,
        n_models=n_models,
        py_tag=args.py_tag,
        output=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
