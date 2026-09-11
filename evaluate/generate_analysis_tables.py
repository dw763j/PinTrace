"""Generate auditable revision CSV/LaTeX tables from canonical task rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from paths import ANALYSIS_FUNNELS, EVALUATION, RESOURCES, display_path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {str(row["task_id"]) for row in _read_jsonl(path)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int | None, denominator: int) -> float | None:
    return numerator / denominator if numerator is not None and denominator else None


def _funnel_row(group: tuple[str, str, str, str], rows: list[dict], constraint_scope: str = "all") -> dict:
    track, mode, model, population = group
    suffix = "" if constraint_scope == "all" else f"_{constraint_scope}"
    n = len(rows)
    s = sum(bool(row.get(f"has_model{suffix}_constraint")) for row in rows)
    p = sum(bool(row.get(f"has_valid_model{suffix}_constraint")) for row in rows)
    security_available = all(row.get("security_status") == "available" for row in rows)
    v = sum(bool(row.get(f"has_model{suffix}_cve")) for row in rows) if security_available else None
    return {
        "track": track,
        "mode": mode,
        "model": model,
        "population": population,
        "constraint_scope": constraint_scope,
        "N": n,
        "S": s,
        "P": p,
        "V": v,
        "incidence_numerator": s,
        "incidence_denominator": n,
        "incidence_rate": _rate(s, n),
        "validity_numerator": p,
        "validity_denominator": s,
        "validity_rate": _rate(p, s),
        "conditional_cve_numerator": v,
        "conditional_cve_denominator": p,
        "conditional_cve_rate": _rate(v, p),
        "unconditional_cve_numerator": v,
        "unconditional_cve_denominator": n,
        "unconditional_cve_rate": _rate(v, n),
    }


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_funnel_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Track & Mode & Model & Population & $N$ & $S$ & $P$ & $V$ \\",
        r"\midrule",
    ]
    for row in rows:
        values = [row[key] for key in ("track", "mode", "model", "population", "N", "S", "P", "V")]
        lines.append(" & ".join(_latex_escape(value if value is not None else "--") for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=EVALUATION)
    parser.add_argument(
        "--clean-task-ids",
        type=Path,
        default=RESOURCES / "revision" / "pintrace_clean_task_ids.jsonl",
    )
    parser.add_argument(
        "--bcb-task-ids",
        type=Path,
        default=RESOURCES / "revision" / "bcb_tpl_task_ids.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ANALYSIS_FUNNELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_rows = _read_jsonl(args.canonical_root / "task_outcomes.jsonl")
    clean_ids = _read_ids(args.clean_task_ids)
    bcb_ids = _read_ids(args.bcb_task_ids)
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in task_rows:
        base = (str(row["track"]), str(row["mode"]), str(row["model"]))
        if row["track"] == "d1" and (bcb_ids is None or str(row["task_id"]) in bcb_ids):
            grouped[(*base, "bcb_tpl")].append(row)
        if row["track"] == "d2":
            grouped[(*base, "full")].append(row)
            if clean_ids is not None and str(row["task_id"]) in clean_ids:
                grouped[(*base, "clean")].append(row)
    funnel_rows = [_funnel_row(group, rows) for group, rows in sorted(grouped.items())]
    exact_funnel_rows = [_funnel_row(group, rows, "exact") for group, rows in sorted(grouped.items())]
    range_funnel_rows = [_funnel_row(group, rows, "range") for group, rows in sorted(grouped.items())]

    layer_rows: list[dict] = []
    distutils_rows: list[dict] = []
    for group, rows in sorted(grouped.items()):
        track, mode, model, population = group
        counts = Counter(str(row.get("compatibility_layer") or "unavailable") for row in rows)
        for layer, count in sorted(counts.items()):
            layer_rows.append(
                {
                    "track": track,
                    "mode": mode,
                    "model": model,
                    "population": population,
                    "layer": layer,
                    "numerator": count,
                    "denominator": len(rows),
                    "rate": _rate(count, len(rows)),
                }
            )
        if track == "d1" or track == "d2":
            retained = [row for row in rows if row.get("ty_install_category") != "missing_distutils"]
            for sensitivity, selected in (("all", rows), ("exclude_distutils", retained)):
                compatible = sum(row.get("ty_compatible") is True for row in selected)
                distutils_rows.append(
                    {
                        "track": track,
                        "mode": mode,
                        "model": model,
                        "population": population,
                        "sensitivity": sensitivity,
                        "compatible_numerator": compatible,
                        "denominator": len(selected),
                        "compatibility_rate": _rate(compatible, len(selected)),
                        "excluded_tasks": len(rows) - len(selected),
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "constraint_funnel.csv", funnel_rows)
    _write_csv(args.output / "constraint_funnel_all.csv", funnel_rows)
    _write_csv(args.output / "constraint_funnel_exact.csv", exact_funnel_rows)
    _write_csv(args.output / "constraint_funnel_range.csv", range_funnel_rows)
    _write_csv(args.output / "compatibility_layers.csv", layer_rows)
    _write_csv(args.output / "distutils_sensitivity.csv", distutils_rows)
    _write_funnel_tex(args.output / "constraint_funnel.tex", exact_funnel_rows)
    _write_funnel_tex(args.output / "constraint_funnel_all.tex", funnel_rows)
    _write_funnel_tex(args.output / "constraint_funnel_range.tex", range_funnel_rows)
    metadata = {
        "canonical_root": display_path(args.canonical_root),
        "clean_task_ids": display_path(args.clean_task_ids) if args.clean_task_ids else None,
        "bcb_task_ids": display_path(args.bcb_task_ids) if args.bcb_task_ids else None,
        "row_counts": {
            "constraint_funnel": len(funnel_rows),
            "constraint_funnel_exact": len(exact_funnel_rows),
            "constraint_funnel_range": len(range_funnel_rows),
            "compatibility_layers": len(layer_rows),
            "distutils_sensitivity": len(distutils_rows),
        },
    }
    (args.output / "table_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["row_counts"]))


if __name__ == "__main__":
    main()
