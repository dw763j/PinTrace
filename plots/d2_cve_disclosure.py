#!/usr/bin/env python3
# Usage (repository root; optional: source .venv/bin/activate):
#   python -m plots.d2_cve_disclosure \
#     --outputs-d2 outputs/d2 \
#     --python-version 3.12 \
#     --cutoff-json resources/model_knowledge_cutoff.example.json
"""
Direction D: distribution of ``Δ = disclosure_time − knowledge_cutoff`` for unique CVEs
(intentionally orthogonal to Fig.3's TPL release-date boxplots).

- Build a disclosure-time index from OSV JSON; scan ``m4_vuln_records.json`` and aggregate per model.
- **Figure**: for each model, one point per **unique CVE**; **X = Δ** (days or months), **Y = model**
  (horizontal violin/box). The **x = 0** line marks the knowledge cutoff (negative ⇒ disclosed before cutoff).
  White background + single cool fill color (not per-model hue), unlike ``d2_model_tpl_release_date_boxplot``.
- JSON summaries mirror the old logic; ``--cutoff-json`` is required for meaningful plots (models without cutoffs are skipped).
- Drop disclosures earlier than **2008-01-01 (UTC)**; runs in ``EXCLUDED_MODEL_KEYS`` are ignored.

Example cutoff JSON (keys are ``normalize_model_key`` values, same as metrics)::

  {"gpt-5.4": "2025-04-01T00:00:00+00:00", "default": "2024-06-01T00:00:00+00:00"}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# import sys
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from paths import OSV_INDEX, OUTPUTS
from plots.model_display import (
    EXCLUDED_MODEL_KEYS,
    infer_model_raw_name_from_m4,
    is_excluded_model,
    normalize_model_key,
    order_model_keys,
    paper_display_label,
)

PLOTS_DIR = Path(__file__).resolve().parent


def _brighten_hex(hex_color: str, toward_white: float) -> str:
    """Mix ``hex_color`` toward white (0 = unchanged, 1 = white)."""
    h = hex_color.strip().lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    t = toward_white
    r = int(r + (255 - r) * t)
    g = int(g + (255 - g) * t)
    b = int(b + (255 - b) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# Fig.3 (TPL release dates) uses warm boxplots + diamond cutoffs; this figure stays monochrome + only x=0
_TEXT_MAIN = "#1E293B"
_TEXT_MUTED = "#64748B"
_FIG_BG = "#FFFFFF"
_AX_BG = "#FFFFFF"
_ZERO_LINE = "#C62828"
# Same fill for every model: separates shapes from the background without hue encoding
_DISTRIBUTION_FILL = _brighten_hex("#5B8FA8", 0.12)

# Ignore ultra-early disclosures (UTC); treat them as noise for both plots and summaries
MIN_DISCLOSURE_UTC = datetime(2008, 1, 1, tzinfo=timezone.utc)

def keep_disclosure_datetime(dt: datetime) -> bool:
    u = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return u >= MIN_DISCLOSURE_UTC


def partition_events_by_disclosure_floor(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop events whose ``disclosure_time`` is before ``MIN_DISCLOSURE_UTC``; keep unresolved rows."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for ev in events:
        ds = ev.get("disclosure_time")
        if not ds:
            kept.append(ev)
            continue
        try:
            dt = datetime.fromisoformat(str(ds).replace("Z", "+00:00"))
        except ValueError:
            kept.append(ev)
            continue
        if keep_disclosure_datetime(dt):
            kept.append(ev)
        else:
            dropped += 1
    return kept, dropped


def read_model_name(m4_path: Path) -> str | None:
    # Deprecated (kept for backward compatibility with old notebooks/scripts).
    # New code should use plots.model_display.infer_model_raw_name_from_m4().
    ms = m4_path.parent / "metrics_summary.json"
    if not ms.is_file():
        return None
    try:
        obj = json.loads(ms.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mn = obj.get("model_name")
    return str(mn) if mn else None


def discover_m4_files(outputs_d2: Path, python_version: str | None) -> list[Path]:
    files: list[Path] = []
    for p in sorted(outputs_d2.rglob("m4_vuln_records.json")):
        if python_version:
            tag = f"py{python_version.replace('.', '')}"
            if f"/{tag}/" not in str(p).replace("\\", "/"):
                continue
        files.append(p)
    return files


def _parse_published(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_osv_disclosure_index(osv_dir: Path) -> dict[str, datetime]:
    """
    Walk OSV JSON dumps: map ``published`` timestamps to advisory ids plus CVE/PYSEC/GHSA aliases.

    Later files overwrite earlier keys (typically one canonical row per CVE).
    """
    index: dict[str, datetime] = {}
    for path in sorted(osv_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pub = _parse_published(data.get("published"))
        if pub is None:
            continue
        oid = data.get("id")
        if isinstance(oid, str) and oid:
            index[oid] = pub
        for alias in data.get("aliases") or []:
            if isinstance(alias, str) and alias:
                index[alias] = pub
    return index


def resolve_disclosure(
    cve_ids: list[str],
    osv_ids: list[str],
    index: dict[str, datetime],
) -> tuple[datetime | None, str | None]:
    """Return ``(disclosure_time, matched_id)`` preferring CVE ids, then OSV ids."""
    for cid in cve_ids or []:
        if isinstance(cid, str) and cid in index:
            return index[cid], cid
    for oid in osv_ids or []:
        if isinstance(oid, str) and oid in index:
            return index[oid], oid
    return None, None


def collect_cve_events_from_m4(
    m4_path: Path,
    disclosure_index: dict[str, datetime],
) -> list[dict[str, Any]]:
    """Expand each vulnerable finding in one ``m4`` dump into CVE-level (or OSV-only) events."""
    out: list[dict[str, Any]] = []
    try:
        raw = json.loads(m4_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(raw, list):
        return out
    for rec in raw:
        for vf in rec.get("vuln_findings") or []:
            if not vf.get("is_vulnerable"):
                continue
            cve_ids = [str(x) for x in (vf.get("cve_ids") or []) if x]
            osv_ids = [str(x) for x in (vf.get("osv_ids") or []) if x]
            if cve_ids:
                for cid in cve_ids:
                    # Prefer direct CVE hits; otherwise fall back to companion osv_ids (GHSA/PYSEC, etc.)
                    dtc, hid = resolve_disclosure([cid], osv_ids, disclosure_index)
                    if dtc is None:
                        resolved_via = "unresolved"
                    elif hid == cid:
                        resolved_via = "cve"
                    else:
                        resolved_via = "osv_id_fallback"  # e.g., GHSA matched when CVE id missing from index
                    out.append(
                        {
                            "task_id": rec.get("task_id"),
                            "pypi_name": vf.get("pypi_name"),
                            "pinned_version": vf.get("version"),
                            "cve_id": cid,
                            "disclosure_time": dtc.isoformat() if dtc else None,
                            "disclosure_hit_key": hid,
                            "resolved_via": resolved_via,
                        }
                    )
            elif osv_ids:
                dt, hit_id = resolve_disclosure([], osv_ids, disclosure_index)
                out.append(
                    {
                        "task_id": rec.get("task_id"),
                        "pypi_name": vf.get("pypi_name"),
                        "pinned_version": vf.get("version"),
                        "cve_id": None,
                        "osv_ids": osv_ids,
                        "disclosure_time": dt.isoformat() if dt else None,
                        "disclosure_hit_key": hit_id,
                        "resolved_via": "osv_only",
                    }
                )
    return out


def load_cutoff_config(path: Path | None) -> dict[str, datetime]:
    if not path or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, datetime] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            dt = _parse_published(v)
            if dt:
                out[str(k).strip().lower()] = dt
    return out


def cutoff_for_model(model_key: str, cutoffs: dict[str, datetime]) -> datetime | None:
    if model_key in cutoffs:
        return cutoffs[model_key]
    return cutoffs.get("default")


def _utc_dt(d: datetime) -> datetime:
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


def disclosure_minus_cutoff_days_unique(
    rows: list[dict[str, Any]],
    mk: str,
    cutoffs: dict[str, datetime],
) -> list[float]:
    """
    One sample per unique CVE: Δ days = (disclosure − knowledge_cutoff).

    Negative values mean public before the cutoff; positive values mean disclosed afterward.
    """
    co = cutoff_for_model(mk, cutoffs)
    if co is None:
        return []
    co_u = _utc_dt(co)
    by_cve: dict[str, datetime] = {}
    for r in rows:
        cid = r.get("cve_id") or r.get("disclosure_hit_key")
        ds = r.get("disclosure_time")
        if not ds or not cid:
            continue
        try:
            dt = datetime.fromisoformat(ds.replace("Z", "+00:00"))
        except ValueError:
            continue
        if not keep_disclosure_datetime(dt):
            continue
        k = str(cid)
        if k not in by_cve:
            by_cve[k] = dt
    out: list[float] = []
    for dt in by_cve.values():
        u = _utc_dt(dt)
        out.append((u - co_u).total_seconds() / 86400.0)
    return out


def _to_plot_delta(days: list[float], unit: str) -> list[float]:
    if unit == "months":
        return [d / (365.25 / 12.0) for d in days]
    return days


def plot_disclosure_delta_figure(
    *,
    by_model: dict[str, list[dict[str, Any]]],
    display_names: dict[str, str],
    cutoffs: dict[str, datetime],
    figure_path: Path,
    dpi: int = 300,
    plot_style: str = "violin",
    delta_unit: str = "days",
) -> bool:
    """Plot Δ (disclosure − cutoff) on X and models on Y; x=0 marks the knowledge cutoff."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
        # from matplotlib.patches import Patch
    except ImportError:
        return False

    if not cutoffs:
        return False

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans", "sans-serif"],
            "axes.unicode_minus": True,
        }
    )

    model_keys_all = order_model_keys(by_model.keys())
    data_per_model: list[list[float]] = []
    labels: list[str] = []

    for mk in model_keys_all:
        days = disclosure_minus_cutoff_days_unique(by_model[mk], mk, cutoffs)
        if not days:
            continue
        data_per_model.append(_to_plot_delta(days, delta_unit))
        labels.append(display_names.get(mk, mk)[:40])

    if not labels:
        return False

    n = len(labels)
    positions = np.arange(1, n + 1)
    fig_w = 9.5
    fig_h = max(4.8, n * 0.42 + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_AX_BG)

    ax.axvline(
        0.0,
        color=_ZERO_LINE,
        linestyle="-",
        linewidth=2.0,
        zorder=8,
        label="Δ = 0 (knowledge cutoff of the model)",
    )

    if plot_style == "violin":
        parts = ax.violinplot(
            data_per_model,
            positions=positions,
            vert=False,
            widths=0.78,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        bodies = list(parts.get("bodies") or [])  # type: ignore
        for b in bodies:
            b.set_facecolor(_DISTRIBUTION_FILL)
            b.set_edgecolor("#FFFFFF")
            b.set_alpha(0.88)
            b.set_linewidth(0.9)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set(color=_TEXT_MUTED, linewidth=0.85)
        if "cmedians" in parts:
            parts["cmedians"].set(color=_TEXT_MAIN, linewidth=1.35)
    else:
        bp = ax.boxplot(
            data_per_model,
            positions=positions,
            vert=False,
            widths=0.52,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(_DISTRIBUTION_FILL)
            patch.set_edgecolor("#FFFFFF")
            patch.set_linewidth(0.75)
            patch.set_alpha(0.9)
        for w in bp["whiskers"]:
            w.set(color=_TEXT_MUTED, linewidth=0.85)
        for cap in bp["caps"]:
            cap.set(color=_TEXT_MUTED, linewidth=0.85)
        for med in bp["medians"]:
            med.set(color=_TEXT_MAIN, linewidth=1.25)
        for fl in bp["fliers"]:
            fl.set(
                markerfacecolor=_DISTRIBUTION_FILL,
                markeredgecolor=_DISTRIBUTION_FILL,
                marker="o",
                markersize=2.8,
                alpha=0.75,
            )

    # Annotate left of x=0 with the share of unique CVEs disclosed before the cutoff
    x_min, x_max = ax.get_xlim()
    span_neg = 0.0 - x_min
    if span_neg > 1e-9:
        x_pct = x_min + 0.06 * span_neg
    else:
        pad = 0.02 * max(x_max - x_min, 1e-6)
        x_pct = -pad
        ax.set_xlim(x_min - pad, x_max)

    for i, pos in enumerate(positions):
        deltas = data_per_model[i]
        fr = sum(1 for x in deltas if x < 0) / len(deltas)
        ax.text(
            x_pct,
            pos,
            f"{100.0 * fr:.2f}%",
            ha="left",
            va="center",
            fontsize=7.6,
            color=_TEXT_MUTED,
            zorder=9,
            clip_on=True,
        )

    unit_lbl = "months" if delta_unit == "months" else "days"
    ax.set_xlabel(
        f"Δ = disclosure − knowledge cutoff of the model ({unit_lbl}); negative ⇒ disclosed before cutoff",
        fontsize=10,
        color=_TEXT_MAIN,
    )
    ax.set_ylabel("Model", fontsize=10, color=_TEXT_MUTED)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.75, color="#94A3B8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", labelsize=8.8, colors=_TEXT_MAIN)

    leg_line = Line2D([0], [0], color=_ZERO_LINE, linewidth=2.0, linestyle="-", label="Δ = 0 (knowledge cutoff)")

    ax.legend(
        handles=[leg_line],
        loc="upper left",
        bbox_to_anchor=(0.02, 1.01),
        frameon=True,
        framealpha=0.96,
        edgecolor="#CBD5E1",
        facecolor="#FFFFFF",
        fontsize=8.5,
    )

    fig.subplots_adjust(left=0.26, right=0.97, top=0.96, bottom=0.12)
    fig.savefig(figure_path, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVE disclosure age vs model knowledge cutoff (M4 / OSV).")
    p.add_argument("--outputs-d2", type=Path, default=OUTPUTS / "d2")
    p.add_argument("--python-version", type=str, default=None)
    p.add_argument("--osv-dir", type=Path, default=OSV_INDEX)
    p.add_argument(
        "--cutoff-json",
        type=Path,
        default=None,
        help='Model knowledge cutoffs as JSON: {"model_key": "ISO8601", "default": "..."}',
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=f"Default: {PLOTS_DIR.name}/d2_cve_disclosure_age.json",
    )
    p.add_argument(
        "--figure-path",
        type=Path,
        default=PLOTS_DIR / "d2_cve_disclosure_delta_violin.pdf",
        help="Default PDF path for the Δ distribution (distinct filename from Fig.3)",
    )
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for PDF output")
    p.add_argument(
        "--plot-style",
        choices=["violin", "box"],
        default="violin",
        help="Horizontal distribution style: violin or box",
    )
    p.add_argument(
        "--delta-unit",
        choices=["days", "months"],
        default="months",
        help="Unit for Δ on the X axis: days or months (months use 365.25/12 days)",
    )
    p.add_argument(
        "--include-events",
        action="store_true",
        help="Include every by_model event in JSON (default keeps compact summaries only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.outputs_d2
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    print("Building OSV disclosure index...", flush=True)
    disc_idx = build_osv_disclosure_index(args.osv_dir)
    print(f"  indexed keys: {len(disc_idx)}", flush=True)

    m4_files = discover_m4_files(root, args.python_version)
    if not m4_files:
        raise SystemExit("no m4_vuln_records.json found")

    cutoffs = load_cutoff_config(args.cutoff_json)
    by_model_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    display_by_key: dict[str, str] = {}
    global_events: list[dict[str, Any]] = []
    dropped_pre_2008 = 0
    skipped_excluded_m4: list[str] = []

    for m4_path in m4_files:
        raw_name = infer_model_raw_name_from_m4(m4_path)
        mk = normalize_model_key(raw_name)
        if is_excluded_model(mk, raw_name):
            skipped_excluded_m4.append(str(m4_path.resolve()))
            continue
        model_display = paper_display_label(mk, raw_name)
        display_by_key[mk] = model_display
        events = collect_cve_events_from_m4(m4_path, disc_idx)
        events, nd = partition_events_by_disclosure_floor(events)
        dropped_pre_2008 += nd
        for ev in events:
            ev["m4_path"] = str(m4_path.resolve())
            ev["model_key"] = mk
            ev["model_name"] = model_display
        by_model_events[mk].extend(events)
        global_events.extend(events)

    if skipped_excluded_m4:
        print(
            f"Skipped {len(skipped_excluded_m4)} m4 file(s) (excluded models: {sorted(EXCLUDED_MODEL_KEYS)}).",
            flush=True,
        )

    # Summaries: disclosure dates per unique CVE relative to each model cutoff
    def summarize_model(rows: list[dict[str, Any]], mk: str) -> dict[str, Any]:
        co = cutoff_for_model(mk, cutoffs)
        by_cve: dict[str, datetime] = {}
        for r in rows:
            cid = r.get("cve_id") or r.get("disclosure_hit_key")
            ds = r.get("disclosure_time")
            if not ds or not cid:
                continue
            try:
                dt = datetime.fromisoformat(ds.replace("Z", "+00:00"))
            except ValueError:
                continue
            if not keep_disclosure_datetime(dt):
                continue
            k = str(cid)
            if k not in by_cve:
                by_cve[k] = dt
        days_before_cutoff: list[float] = []
        for dt in by_cve.values():
            if co is not None:
                days_before_cutoff.append((co - dt).total_seconds() / 86400.0)
        disclosed_before_cutoff = (
            sum(1 for d in days_before_cutoff if d > 0) if co is not None else None
        )
        return {
            "unique_cves_with_disclosure": len(by_cve),
            "knowledge_cutoff": co.isoformat() if co else None,
            "days_cutoff_minus_disclosure_stats": {
                "count": len(days_before_cutoff),
                "positive_meaning": "cutoff is after disclosure => model could have seen public CVE",
                "values_sample": days_before_cutoff[:20],
            }
            if days_before_cutoff
            else None,
            "cves_disclosed_strictly_before_cutoff": disclosed_before_cutoff,
        }

    per_model_summary = {mk: summarize_model(rows, mk) for mk, rows in by_model_events.items()}
    global_cves: dict[str, datetime] = {}
    for r in global_events:
        cid = r.get("cve_id") or r.get("disclosure_hit_key")
        ds = r.get("disclosure_time")
        if not ds or not cid:
            continue
        try:
            dt = datetime.fromisoformat(ds.replace("Z", "+00:00"))
        except ValueError:
            continue
        if not keep_disclosure_datetime(dt):
            continue
        k = str(cid)
        if k not in global_cves:
            global_cves[k] = dt
    missing_disclosure = sum(
        1
        for r in global_events
        if (r.get("cve_id") or r.get("osv_ids")) and not r.get("disclosure_time")
    )

    sample: dict[str, Any] = {}
    for mk, rows in sorted(by_model_events.items()):
        sample[mk] = rows[:5]

    out: dict[str, Any] = {
        "config": {
            "outputs_d2": str(root.resolve()),
            "plots_dir": str(PLOTS_DIR.resolve()),
            "python_version_filter": args.python_version,
            "osv_dir": str(args.osv_dir.resolve()),
            "osv_disclosure_index_size": len(disc_idx),
            "m4_files": len(m4_files),
            "cutoff_json": str(args.cutoff_json) if args.cutoff_json else None,
            "min_disclosure_utc": MIN_DISCLOSURE_UTC.isoformat(),
            "excluded_model_keys_from_display": sorted(EXCLUDED_MODEL_KEYS),
            "m4_files_skipped_excluded_models": skipped_excluded_m4,
            "figure_delta_definition": "Δ = disclosure_time − knowledge_cutoff; negative ⇒ disclosed before cutoff",
            "figure_plot_style": args.plot_style,
            "figure_delta_unit": args.delta_unit,
        },
        "global": {
            "total_cve_events": len(global_events),
            "unique_cve_ids_observed": len(global_cves),
            "events_missing_disclosure_in_index": missing_disclosure,
            "events_excluded_disclosure_before_min_utc": dropped_pre_2008,
            "m4_files_count_skipped_excluded_models": len(skipped_excluded_m4),
        },
        "per_model_summary": per_model_summary,
        "by_model_event_counts": {k: len(v) for k, v in sorted(by_model_events.items())},
        "by_model_events_sample": sample,
    }
    if args.include_events:
        out["by_model_events"] = {k: v for k, v in sorted(by_model_events.items())}

    out_path = args.output_json or (PLOTS_DIR / "d2_cve_disclosure_age.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_figure:
        if not cutoffs:
            print("Warning: --cutoff-json missing or empty; delta figure needs per-model cutoffs. Skipped figure.", flush=True)
        else:
            ok = plot_disclosure_delta_figure(
                by_model=dict(by_model_events),
                display_names=display_by_key,
                cutoffs=cutoffs,
                figure_path=args.figure_path,
                dpi=args.dpi,
                plot_style=args.plot_style,
                delta_unit=args.delta_unit,
            )
            if ok:
                print(f"Wrote figure {args.figure_path}", flush=True)
            else:
                print(
                    "matplotlib unavailable, or no unique CVE with disclosure+cutoff for any model; skipped figure.",
                    flush=True,
                )

    print(json.dumps(out["global"], ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
