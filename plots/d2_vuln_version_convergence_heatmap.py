#!/usr/bin/env python3
# Usage (repository root; optional: source .venv/bin/activate):
#   python -m plots.d2_vuln_version_convergence_heatmap --outputs-d2 outputs/d2 --python-version 3.12
#   python -m plots.d2_vuln_version_convergence_heatmap --d2-modes inline,inline_no_vuln --python-version 3.12
"""
Figure E: cross-model convergence heatmap for vulnerable (TPL, version) pins.

- Rows: Top-N most frequent vulnerable pairs (sorted by total assignments)
- Columns: models
- Cells: assignment counts (``count``) or binary presence (``binary``)
- ``--d2-modes`` optionally limits scanning to comma-separated modes under ``outputs/d2/``
- JSON ``global`` reports coverage metrics such as ``assignments_covered_by_top_n_pairs``
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paths import OUTPUTS
from plots.model_display import (
    infer_model_raw_name_from_m4,
    is_excluded_model,
    normalize_model_key,
    order_model_keys,
    paper_display_label,
)

Pair = tuple[str, str]
PLOTS_DIR = Path(__file__).resolve().parent


def parse_d2_modes_arg(raw: str | None) -> set[str] | None:
    """Parse comma-separated modes (``inline,inline_no_vuln``); ``None`` means scan all."""
    if raw is None or not str(raw).strip():
        return None
    modes = {x.strip() for x in str(raw).split(",") if x.strip()}
    return modes or None


def discover_m4_files(
    outputs_d2: Path,
    python_version: str | None,
    modes: set[str] | None = None,
) -> list[Path]:
    files: list[Path] = []
    for p in sorted(outputs_d2.rglob("m4_vuln_records.json")):
        if modes is not None:
            loc = parse_run_location(p, outputs_d2)
            if loc["mode"] not in modes:
                continue
        if python_version:
            tag = f"py{python_version.replace('.', '')}"
            if f"/{tag}/" not in str(p).replace("\\", "/"):
                continue
        files.append(p)
    return files


def parse_run_location(m4_path: Path, outputs_d2: Path) -> dict[str, str]:
    """Return ``mode``, ``run_dir``, and ``py_tag`` relative to ``outputs_d2``."""
    try:
        rel = m4_path.relative_to(outputs_d2)
    except ValueError:
        return {"mode": "", "run_dir": "", "py_tag": ""}
    parts = rel.parts
    mode = parts[0] if len(parts) > 0 else ""
    run_dir = parts[1] if len(parts) > 1 else ""
    py_tag = parts[2] if len(parts) > 2 else ""
    return {"mode": mode, "run_dir": run_dir, "py_tag": py_tag}


def collect_vulnerable_pair_counts(raw: list[dict[str, Any]]) -> Counter[Pair]:
    """
    Count vulnerable ``(pypi_name, version)`` occurrences inside one ``m4_vuln_records`` dump.

    Each vulnerable ``vuln_findings`` entry increments the counter.
    """
    out: Counter[Pair] = Counter()
    for rec in raw:
        for vf in rec.get("vuln_findings") or []:
            if not vf.get("is_vulnerable"):
                continue
            pkg = str(vf.get("pypi_name") or "").strip().lower()
            ver = str(vf.get("version") or "").strip()
            if not pkg or not ver:
                continue
            out[(pkg, ver)] += 1
    return out


def to_serializable_pair(pair: Pair) -> dict[str, str]:
    return {"pypi_name": pair[0], "version": pair[1]}


def try_plot_heatmap(
    *,
    matrix: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    figure_path: Path,
    value_mode: str,
    annotate: bool,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not matrix or not col_labels:
        return False

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(matrix)
    n_cols = len(col_labels)
    fig_w = max(10, n_cols * 1.2)
    fig_h = max(6, n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("count" if value_mode == "count" else "binary hit (0/1)")

    ax.set_xticks(list(range(n_cols)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(list(range(n_rows)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Models")
    ax.set_ylabel(f"Count of top vulnerable (TPL, version) pairs top-{n_rows}")
    # ax.set_title(
    #     "Direction E: Cross-model convergence on vulnerable version pins "
    #     f"({value_mode}, Top-{n_rows})"
    # )

    if annotate:
        vmax = max((max(r) for r in matrix), default=0)
        threshold = vmax / 2.0 if vmax > 0 else 0.5
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val == 0:
                    continue
                color = "white" if val > threshold else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-model vulnerable version convergence heatmap (M4 / D2).")
    p.add_argument("--outputs-d2", type=Path, default=OUTPUTS / "d2")
    p.add_argument(
        "--python-version",
        type=str,
        default=None,
        help="If set (e.g. 3.12), only include .../py312/m4_vuln_records.json paths",
    )
    p.add_argument(
        "--d2-modes",
        type=str,
        default=None,
        metavar="MODES",
        help=(
            "Comma-separated pinning modes under outputs/d2 (e.g. inline,inline_no_vuln,inline_api_rag). "
            "Default scans every mode."
        ),
    )
    p.add_argument("--top-n", type=int, default=10, help="Number of heatmap rows (Top-N vulnerable pairs)")
    p.add_argument(
        "--value-mode",
        choices=["count", "binary"],
        default="count",
        help="Cell values: count=occurrences, binary=0/1 presence",
    )
    p.add_argument(
        "--model-keys-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON whitelist of model keys. Supports either "
            "1) {\"model_key\": \"...\"} (use dict keys, ignore default/_usage) or "
            "2) [\"model_key1\", \"model_key2\", ...]"
        ),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=f"Default: {PLOTS_DIR.name}/d2_vuln_version_convergence_heatmap.json",
    )
    p.add_argument(
        "--figure-path",
        type=Path,
        default=PLOTS_DIR / "d2_vuln_version_convergence_heatmap.pdf",
    )
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--annotate", action="store_true", help="Annotate each cell (can get crowded for large Top-N)")
    return p.parse_args()


def load_allowed_model_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"model keys json not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"failed to parse --model-keys-json: {e}") from e

    out: set[str] = set()
    if isinstance(raw, dict):
        for k in raw.keys():
            kk = str(k).strip().lower()
            if kk in {"default", "_usage"}:
                continue
            if kk:
                out.add(kk)
    elif isinstance(raw, list):
        for x in raw:
            kk = str(x).strip().lower()
            if kk:
                out.add(kk)
    else:
        raise SystemExit("--model-keys-json must be a JSON object or JSON list")
    return out


def main() -> None:
    args = parse_args()
    root = args.outputs_d2
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    if args.top_n <= 0:
        raise SystemExit("--top-n must be > 0")
    allowed_model_keys = load_allowed_model_keys(args.model_keys_json)
    d2_modes = parse_d2_modes_arg(args.d2_modes)

    m4_files = discover_m4_files(root, args.python_version, d2_modes)
    if not m4_files:
        raise SystemExit(
            f"no m4_vuln_records.json under {root} "
            f"(py filter={args.python_version!r}, d2_modes={sorted(d2_modes) if d2_modes else None!r})"
        )

    by_model_pair_counts: dict[str, Counter[Pair]] = defaultdict(Counter)
    display_by_key: dict[str, str] = {}
    global_pair_counts: Counter[Pair] = Counter()
    file_errors: list[dict[str, str]] = []
    scanned_runs = 0
    skipped_excluded = 0

    for m4_path in m4_files:
        raw_name = infer_model_raw_name_from_m4(m4_path)
        mk = normalize_model_key(raw_name)
        if is_excluded_model(mk, raw_name):
            skipped_excluded += 1
            continue
        model_display = paper_display_label(mk, raw_name) or raw_name
        if allowed_model_keys is not None and mk not in allowed_model_keys:
            continue
        display_by_key[mk] = model_display

        try:
            raw = json.loads(m4_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            file_errors.append({"m4_path": str(m4_path.resolve()), "error": str(e)})
            continue
        if not isinstance(raw, list):
            file_errors.append({"m4_path": str(m4_path.resolve()), "error": "not a JSON list"})
            continue

        run_counter = collect_vulnerable_pair_counts(raw)
        if not run_counter:
            scanned_runs += 1
            continue

        for pair, c in run_counter.items():
            by_model_pair_counts[mk][pair] += c
            global_pair_counts[pair] += c
        scanned_runs += 1

    if not global_pair_counts:
        raise SystemExit("no vulnerable (pypi_name, version) pairs found in scanned files")

    model_keys = [k for k in order_model_keys(by_model_pair_counts.keys()) if k in by_model_pair_counts]
    top_pairs: list[Pair] = [
        pair
        for pair, _ in sorted(global_pair_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[: args.top_n]
    ]

    matrix_count: list[list[int]] = []
    for pair in top_pairs:
        row: list[int] = []
        for mk in model_keys:
            row.append(by_model_pair_counts[mk].get(pair, 0))
        matrix_count.append(row)

    if args.value_mode == "binary":
        matrix_value = [[1 if v > 0 else 0 for v in row] for row in matrix_count]
    else:
        matrix_value = matrix_count

    row_labels = [f"{pkg}=={ver}" for pkg, ver in top_pairs]
    col_labels = [display_by_key.get(mk, mk) for mk in model_keys]

    top_rows_json: list[dict[str, Any]] = []
    for i, pair in enumerate(top_pairs):
        row_counts = {mk: matrix_count[i][j] for j, mk in enumerate(model_keys)}
        non_zero_models = sum(1 for v in matrix_count[i] if v > 0)
        top_rows_json.append(
            {
                **to_serializable_pair(pair),
                "global_count": global_pair_counts[pair],
                "models_hit_count": non_zero_models,
                "model_counts": row_counts,
            }
        )

    by_model_summary: dict[str, Any] = {}
    for mk in model_keys:
        cnt = by_model_pair_counts[mk]
        by_model_summary[mk] = {
            "model_display_name": display_by_key.get(mk, mk),
            "total_vulnerable_assignments": int(sum(cnt.values())),
            "unique_vulnerable_pairs": len(cnt),
            "top_10_pairs": [
                {**to_serializable_pair(pair), "count": c}
                for pair, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:10]
            ],
        }

    total_assignments = int(sum(global_pair_counts.values()))
    assignments_in_top_n = int(sum(global_pair_counts[p] for p in top_pairs))
    top_n_share = (assignments_in_top_n / total_assignments) if total_assignments else 0.0

    out: dict[str, Any] = {
        "config": {
            "outputs_d2": str(root.resolve()),
            "plots_dir": str(PLOTS_DIR.resolve()),
            "python_version_filter": args.python_version,
            "d2_modes": sorted(d2_modes) if d2_modes else None,
            "m4_files_found": len(m4_files),
            "m4_files_scanned": scanned_runs,
            "m4_files_skipped_excluded_models": skipped_excluded,
            "top_n": args.top_n,
            "value_mode": args.value_mode,
            "model_keys_json": str(args.model_keys_json.resolve()) if args.model_keys_json else None,
            "allowed_model_keys_count": len(allowed_model_keys) if allowed_model_keys is not None else None,
        },
        "global": {
            "total_vulnerable_assignments": total_assignments,
            "unique_vulnerable_pairs": len(global_pair_counts),
            "models": len(model_keys),
            "rows_in_heatmap": len(top_pairs),
            "pairs_hit_by_at_least_2_models_within_top_n": sum(
                1 for row in matrix_count if sum(1 for v in row if v > 0) >= 2
            ),
            "assignments_covered_by_top_n_pairs": assignments_in_top_n,
            "fraction_of_assignments_covered_by_top_n_pairs": round(top_n_share, 6),
        },
        "model_order": [
            {"model_key": mk, "model_display_name": display_by_key.get(mk, mk)}
            for mk in model_keys
        ],
        "top_pairs": top_rows_json,
        "matrix_row_labels": row_labels,
        "matrix_model_keys": model_keys,
        "matrix_counts": matrix_count,
        "matrix_values": matrix_value,
        "by_model_summary": by_model_summary,
        "file_errors": file_errors[:50],
    }

    out_path = args.output_json or (PLOTS_DIR / "d2_vuln_version_convergence_heatmap.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_figure:
        ok = try_plot_heatmap(
            matrix=matrix_value,
            row_labels=row_labels,
            col_labels=col_labels,
            figure_path=args.figure_path,
            value_mode=args.value_mode,
            annotate=args.annotate,
        )
        if ok:
            print(f"Wrote figure {args.figure_path}", flush=True)
        else:
            print("matplotlib not available or no plottable matrix; skipped figure.", flush=True)

    print(json.dumps(out["global"], ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
