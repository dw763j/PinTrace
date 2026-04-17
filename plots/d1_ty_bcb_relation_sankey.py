#!/usr/bin/env python3
"""
Analyze TY compatibility vs BCB runtime outcomes for one D1 model/mode.

Usage:
    python -m plots.d1_ty_bcb_relation_sankey \
      --model "gpt-5.4" \
      --mode inline \
      --python-version 3.12
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from paths import OUTPUTS
from plots.model_display import normalize_model_key, paper_display_label
from stages.summarize_bcb_error_rules import infer_bcb_rule_from_test, infer_mode_and_model

PLOTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D1 TY compatibility to BCB runtime association (single model, Sankey)."
    )
    p.add_argument("--outputs-d1", type=Path, default=OUTPUTS / "d1")
    p.add_argument("--model", type=str, required=True, help="Model key/name, e.g. gpt-5.4.")
    p.add_argument(
        "--mode",
        type=str,
        default="inline",
        choices=["inline", "inline_no_vuln", "requirements.txt"],
    )
    p.add_argument("--python-version", type=str, default="3.12")
    p.add_argument(
        "--rule-summary-json",
        type=Path,
        default=OUTPUTS / "d1" / "bcb_error_rule_summary.py312.json",
        help="Optional summary produced by stages/summarize_bcb_error_rules.py",
    )
    p.add_argument(
        "--bcb-task-filter-jsonl",
        type=Path,
        default=OUTPUTS / "d1" / "bcb_tasks_tpl_in_osv_matrix.jsonl",
        help="Only keep D1 records whose task_id appears in this JSONL (from scripts/stats_d1_bcb_osv_tasks.py).",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--figure-path", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_task_filter_ids(filter_jsonl: Path) -> set[str]:
    if not filter_jsonl.exists():
        raise SystemExit(f"Task filter JSONL not found: {filter_jsonl}")

    allowed: set[str] = set()
    with filter_jsonl.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON at {filter_jsonl}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                continue
            task_id = obj.get("task_id")
            if task_id is None:
                continue
            allowed.add(str(task_id))

    if not allowed:
        raise SystemExit(f"No task_id loaded from task filter JSONL: {filter_jsonl}")
    return allowed


def _is_ty_compatible(rec: dict[str, Any]) -> bool:
    compat = rec.get("compat_results")
    if not isinstance(compat, dict):
        return False
    ty = compat.get("ty")
    if not isinstance(ty, dict):
        return False
    return bool(ty.get("is_compatible"))


def _has_ty_install_error(rec: dict[str, Any]) -> bool:
    compat = rec.get("compat_results")
    if not isinstance(compat, dict):
        return False
    ty = compat.get("ty")
    if not isinstance(ty, dict):
        return False
    errors = ty.get("errors")
    if not isinstance(errors, list):
        return False
    return any(isinstance(e, dict) and str(e.get("rule") or "") == "ty-install-error" for e in errors)


def _bcb_status(rec: dict[str, Any]) -> tuple[str, str]:
    """
    Returns (mid_status, fail_bucket).
    mid_status: Runtime Pass / Runtime Fail
    fail_bucket: one of Fail-Exception / Fail-Assertion / Build-Env Error / Others
    """
    compat = rec.get("compat_results")
    bcb = compat.get("bcb_test") if isinstance(compat, dict) else None
    if not isinstance(bcb, dict):
        return "Runtime Fail", "Others"

    status = str(bcb.get("status") or "").strip().lower()
    if status == "pass":
        return "Runtime Pass", ""

    if status == "runner_error":
        return "Runtime Fail", "Others"

    if status == "error":
        if _has_ty_install_error(rec):
            return "Runtime Fail", "Build-Env Error"
        return "Runtime Fail", "Build-Env Error"

    if status == "fail":
        rule = infer_bcb_rule_from_test(bcb).lower()
        if "assertionerror" in rule:
            return "Runtime Fail", "Fail-Assertion"
        if any(
            k in rule
            for k in [
                "module-not-found",
                "modulenotfounderror",
                "importerror",
                "dependency-unsatisfiable",
                "python-version-constraint",
                "build-backend-failure",
                "py312-distutils-removed",
                "permission",
                "disk",
                "network",
            ]
        ):
            return "Runtime Fail", "Build-Env Error"
        return "Runtime Fail", "Fail-Exception"

    return "Runtime Fail", "Others"


def find_target_m5(outputs_d1: Path, model_norm: str, mode: str, py_tag: str) -> Path:
    candidates = sorted(outputs_d1.glob(f"{mode}/*/{py_tag}/m5_compat_records.json"))
    if not candidates:
        raise SystemExit(f"No m5_compat_records.json found under {outputs_d1}/{mode}/*/{py_tag}")
    for path in candidates:
        m, md = infer_mode_and_model(path)
        if m != mode:
            continue
        if normalize_model_key(md) == model_norm:
            return path
    available = sorted({infer_mode_and_model(p)[1] for p in candidates})
    raise SystemExit(f"Model {model_norm!r} not found for mode={mode}, py_tag={py_tag}. Available: {available}")


def _draw_flow(
    ax: Any,
    x0: float,
    x1: float,
    y0a: float,
    y0b: float,
    y1a: float,
    y1b: float,
    color: str,
    alpha: float = 0.82,
) -> None:
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    c = 0.24 * (x1 - x0)
    verts = [
        (x0, y0a),
        (x0 + c, y0a),
        (x1 - c, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - c, y1b),
        (x0 + c, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def plot_sankey(
    left_to_mid: Counter[tuple[str, str]],
    mid_to_right: Counter[tuple[str, str]],
    ty_compatible_to_fail_reason: Counter[str],
    figure_path: Path,
    title: str | None = None,
    dpi: int=300,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    left_nodes = ["ty Comp.", "ty Incomp."]
    mid_nodes = ["Runtime Pass", "Runtime Fail"]
    right_nodes = ["Fail-Exception", "Fail-Assertion", "Build-Env Error", "Others"]

    # Bright, light Morandi-like palette
    node_color = "#9FB79C"
    flow_palette = {
        ("ty Comp.", "Runtime Pass"): "#AFC7A7",
        ("ty Comp.", "Runtime Fail"): "#D8BBB8",
        ("ty Incomp.", "Runtime Pass"): "#BFD4B8",
        ("ty Incomp.", "Runtime Fail"): "#C9A9A2",
        ("Runtime Fail", "Fail-Exception"): "#D2B7B1",
        ("Runtime Fail", "Fail-Assertion"): "#E0CDA6",
        ("Runtime Fail", "Build-Env Error"): "#BEB7D3",
        ("Runtime Fail", "Others"): "#A8BDD2",
    }

    totals_left = {n: sum(v for (a, _), v in left_to_mid.items() if a == n) for n in left_nodes}
    totals_mid = {
        n: sum(v for (_, b), v in left_to_mid.items() if b == n) if n == "Runtime Pass" else sum(v for (_, b), v in left_to_mid.items() if b == n)
        for n in mid_nodes
    }
    totals_right = {n: sum(v for (_, b), v in mid_to_right.items() if b == n) for n in right_nodes}

    grand = max(sum(totals_left.values()), 1)
    gap = 0.035
    top = 0.96

    def build_positions(nodes: list[str], totals: dict[str, int]) -> dict[str, tuple[float, float]]:
        available = 0.88 - gap * (len(nodes) - 1)
        pos: dict[str, tuple[float, float]] = {}
        y = top
        for n in nodes:
            h = available * (totals[n] / grand)
            pos[n] = (y - h, y)
            y = y - h - gap
        return pos

    pos_l = build_positions(left_nodes, totals_left)
    pos_m = build_positions(mid_nodes, totals_mid)
    # Runtime Fail breakdown: vertical band = 200% of Runtime Fail node height, bottom = rf_low.
    pos_r: dict[str, tuple[float, float]] = {}
    rf_low, rf_high = pos_m["Runtime Fail"]
    rf_span = rf_high - rf_low
    # Bottom-aligned band: 200% of Runtime Fail height (sketch), extra room for labels.
    right_band_low = rf_low
    right_band_high = min(0.985, rf_low + rf_span * 2.0)
    right_span = right_band_high - right_band_low
    rf_total = max(totals_mid["Runtime Fail"], 1)
    right_nonzero = [n for n in right_nodes if totals_right.get(n, 0) > 0]
    if right_nonzero:
        gap_r = min(0.035, right_span * 0.09)
        available_r = max(0.0, right_span - gap_r * (len(right_nonzero) - 1))
        y = right_band_high
        for n in right_nodes:
            c = totals_right.get(n, 0)
            if c <= 0:
                pos_r[n] = (right_band_low, right_band_low)
                continue
            h = available_r * (c / rf_total)
            pos_r[n] = (y - h, y)
            y = y - h - gap_r
    else:
        for n in right_nodes:
            pos_r[n] = (right_band_low, right_band_low)

    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x_l, x_m, x_r, w = 0.03, 0.49, 0.95, 0.015

    total_tasks = grand

    # Draw node rectangles + share of all tasks (sketch: xx.x% on left & middle columns)
    for n in left_nodes:
        ya, yb = pos_l[n]
        ax.add_patch(Rectangle((x_l - w, ya), w, yb - ya, color=node_color, alpha=0.95))
        share = totals_left[n] / total_tasks * 100.0 if total_tasks else 0.0
        ax.text(
            x_l - w - 0.016,
            (ya + yb) / 2,
            f"{n} ({share:.2f}%)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            color="#2E3A30",
        )
    for n in mid_nodes:
        ya, yb = pos_m[n]
        ax.add_patch(Rectangle((x_m - w / 2, ya), w, yb - ya, color="#B9A39D", alpha=0.95))
        share = totals_mid[n] / total_tasks * 100.0 if total_tasks else 0.0
        ax.text(
            x_m - w / 2 - 0.016,
            (ya + yb) / 2,
            f"{n} ({share:.2f}%)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            color="#4A3A36",
        )
    for n in right_nodes:
        ya, yb = pos_r[n]
        ax.add_patch(Rectangle((x_r, ya), w, yb - ya, color="#B3C5AA", alpha=0.95))
        c = totals_right.get(n, 0)
        pct = (c / rf_total * 100.0) if rf_total > 0 else 0.0
        ax.text(
            x_r,
            (ya + yb) / 2,
            f"{n} ({pct:.2f}%)",
            ha="right",
            va="center",
            fontsize=9.0,
            color="#3A4435",
        )

    # Draw left -> mid flows from TOP to BOTTOM so Runtime Pass stays on top in each TY block.
    # TY branch percentages are shown on the left TY (green) column by total-task share.
    x_mid_left = x_m - w / 2
    cur_l = {k: pos_l[k][1] for k in left_nodes}
    cur_m_in = {k: pos_m[k][1] for k in mid_nodes}
    for a in left_nodes:
        for b in mid_nodes:
            v = left_to_mid.get((a, b), 0)
            if v <= 0:
                continue
            h = (pos_l[a][1] - pos_l[a][0]) * (v / max(totals_left[a], 1))
            h2 = (pos_m[b][1] - pos_m[b][0]) * (v / max(totals_mid[b], 1))
            y0a, y0b = cur_l[a] - h, cur_l[a]
            y1a, y1b = cur_m_in[b] - h2, cur_m_in[b]
            _draw_flow(ax, x_l, x_mid_left, y0a, y0b, y1a, y1b, flow_palette.get((a, b), "#CBB8B0"))
            pct_total = (v / total_tasks * 100.0) if total_tasks else 0.0
            cx = x_l + 0.004
            cy = (y0a + y0b) / 2
            ax.text(
                cx,
                cy,
                f"{pct_total:.2f}%",
                ha="left",
                va="center",
                fontsize=9.0,
                color="#3D4A42",
                fontweight="medium",
            )
            cur_l[a] -= h
            cur_m_in[b] -= h2

    # Draw Runtime Fail -> right from TOP to BOTTOM.
    # This guarantees the top edge of the first fail bucket aligns with Runtime Fail top edge.
    a = "Runtime Fail"
    cur_m = pos_m[a][1]
    cur_r = {k: pos_r[k][1] for k in right_nodes}
    for b in right_nodes:
        v = mid_to_right.get((a, b), 0)
        if v <= 0:
            continue
        h = (pos_m[a][1] - pos_m[a][0]) * (v / max(totals_mid[a], 1))
        h2 = (pos_r[b][1] - pos_r[b][0]) * (v / max(totals_right[b], 1))
        y0a, y0b = cur_m - h, cur_m
        y1a, y1b = cur_r[b] - h2, cur_r[b]
        _draw_flow(ax, x_m + w / 2, x_r, y0a, y0b, y1a, y1b, flow_palette.get((a, b), "#C5BCB0"))
        cur_m -= h
        cur_r[b] -= h2
    # if title is not None:
    ax.text(0.0, 1.01, title, transform=ax.transAxes, fontsize=12.5, color="#2F3B33")
    # else:
    #     ax.text(0.0, 1.01, transform=ax.transAxes, fontsize=12.5, color="#2F3B33")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    py_tag = f"py{args.python_version.replace('.', '')}"
    model_norm = normalize_model_key(args.model)
    mode = args.mode

    m5_path = find_target_m5(args.outputs_d1, model_norm=model_norm, mode=mode, py_tag=py_tag)
    data = json.loads(m5_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Invalid JSON list: {m5_path}")

    allowed_task_ids = load_task_filter_ids(args.bcb_task_filter_jsonl)
    raw_total_tasks = len(data)
    data = [rec for rec in data if str(rec.get("task_id") or "") in allowed_task_ids]

    left_to_mid: Counter[tuple[str, str]] = Counter()
    mid_to_right: Counter[tuple[str, str]] = Counter()
    ty_compatible_to_fail_reason: Counter[str] = Counter()
    bcb_status_counts: Counter[str] = Counter()

    for rec in data:
        left = "ty Comp." if _is_ty_compatible(rec) else "ty Incomp."
        mid, right = _bcb_status(rec)
        left_to_mid[(left, mid)] += 1
        bcb_status_counts[mid] += 1
        if mid == "Runtime Fail":
            mid_to_right[(mid, right)] += 1
            if left == "ty Comp.":
                ty_compatible_to_fail_reason[right] += 1

    raw_model_name = infer_mode_and_model(m5_path)[1]
    model_display = paper_display_label(model_norm, raw_model_name)

    out_json = args.output_json or (PLOTS_DIR / f"d1_ty_bcb_relation_{model_norm.replace('/', '_')}_{mode}.json")
    fig_path = args.figure_path or (PLOTS_DIR / f"d1_ty_bcb_relation_{model_norm.replace('/', '_')}_{mode}.pdf")

    side_summary: dict[str, Any] = {}
    if args.rule_summary_json.is_file():
        try:
            obj = json.loads(args.rule_summary_json.read_text(encoding="utf-8"))
            side_summary = ((obj.get(raw_model_name) or {}).get(mode) or (obj.get(args.model) or {}).get(mode) or {})
        except (OSError, json.JSONDecodeError):
            side_summary = {}

    out = {
        "config": {
            "outputs_d1": str(args.outputs_d1.resolve()),
            "m5_path": str(m5_path.resolve()),
            "bcb_task_filter_jsonl": str(args.bcb_task_filter_jsonl.resolve()),
            "model_input": args.model,
            "model_key_normalized": model_norm,
            "model_display_name": model_display,
            "mode": mode,
            "python_version": args.python_version,
            "rule_summary_json": str(args.rule_summary_json.resolve()) if args.rule_summary_json else None,
        },
        "global": {
            "total_tasks_before_filter": raw_total_tasks,
            "total_tasks": len(data),
            "left_to_mid": {f"{a} -> {b}": int(v) for (a, b), v in sorted(left_to_mid.items())},
            "mid_to_right": {f"{a} -> {b}": int(v) for (a, b), v in sorted(mid_to_right.items())},
            "runtime_status_counts": dict(bcb_status_counts),
        },
        "reference_from_rule_summary": side_summary,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # title = f"D1 TY-to-BCB Association · {model_display} · {mode}"
    plot_sankey(
        left_to_mid=left_to_mid,
        mid_to_right=mid_to_right,
        ty_compatible_to_fail_reason=ty_compatible_to_fail_reason,
        figure_path=fig_path,
        # title=title,
        dpi=args.dpi,
    )

    print(json.dumps(out["global"], ensure_ascii=False, indent=2))
    print(f"\nWrote {out_json}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()

