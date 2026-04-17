#!/usr/bin/env python3
"""
Batch-scan every ``m4_vuln_records.json`` under ``outputs/d2`` (all modes/models/Python tags).

Reuses ``analyze_m4_safe_alternative_major`` and emits:

- **per_run**: unique vulnerable ``(TPL, version)`` sets plus per-file summaries
- **by_model**: union of pairs for each ``metrics_summary.model_name`` across runs/modes
- **global**: union across every scanned file

A ``(pkg, version)`` cache deduplicates expensive PyPI/OSV work across files.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m scripts.analyze_m4_safe_alternative_all_runs --outputs-d2 outputs/d2 --python-version 3.12
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import GLOBAL_CACHE, OSV_INDEX, OUTPUTS, PYPI_INFO  # noqa: E402
from scripts.analyze_m4_safe_alternative_major import (  # noqa: E402
    PairResult,
    _parse_cutoff_datetime,
    collect_vulnerable_pairs,
    compute_safe_alternative_analysis,
    load_cve_cache_optional,
)
from stages.vuln_checker import load_osv_index  # noqa: E402


def normalize_model_key(name: str) -> str:
    return name.strip().lower()


def read_model_name(m4_path: Path) -> str | None:
    ms = m4_path.parent / "metrics_summary.json"
    if not ms.is_file():
        return None
    try:
        obj = json.loads(ms.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mn = obj.get("model_name")
    return str(mn) if mn else None


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


def discover_m4_files(outputs_d2: Path, python_version: str | None) -> list[Path]:
    files: list[Path] = []
    for p in sorted(outputs_d2.rglob("m4_vuln_records.json")):
        if python_version:
            tag = f"py{python_version.replace('.', '')}"
            if f"/{tag}/" not in str(p).replace("\\", "/"):
                continue
        files.append(p)
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch safe-alternative (same major) analysis for all D2 m4 runs.")
    p.add_argument(
        "--outputs-d2",
        type=Path,
        default=OUTPUTS / "d2",
        help="D2 outputs root (default: outputs/d2)",
    )
    p.add_argument(
        "--python-version",
        type=str,
        default=None,
        help="If set (e.g. 3.12), only include .../py312/m4_vuln_records.json paths; omit for all py* dirs",
    )
    p.add_argument("--reference-date", type=str, default=None)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/d2/m4_safe_alt_major_all_runs.json)",
    )
    p.add_argument(
        "--skip-per-run",
        action="store_true",
        help="Skip writing per_run rows (union stats still warm the pair cache during traversal)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.outputs_d2
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    m4_files = discover_m4_files(root, args.python_version)
    if not m4_files:
        raise SystemExit(f"no m4_vuln_records.json under {root} (py filter={args.python_version!r})")

    osv_index = load_osv_index(str(OSV_INDEX))
    cve_cache = load_cve_cache_optional(GLOBAL_CACHE / "local_cve_cache.json")
    ref_dt = _parse_cutoff_datetime(args.reference_date)
    pair_cache: dict[tuple[str, str], PairResult] = {}

    pairs_by_model: dict[str, set[tuple[str, str]]] = defaultdict(set)
    model_display_by_key: dict[str, str] = {}
    all_pairs: set[tuple[str, str]] = set()
    per_run: list[dict[str, Any]] = []

    for m4_path in m4_files:
        try:
            raw = json.loads(m4_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            per_run.append({"m4_path": str(m4_path), "error": str(e)})
            continue
        if not isinstance(raw, list):
            continue

        pairs = collect_vulnerable_pairs(raw)
        loc = parse_run_location(m4_path, root)
        model_display = read_model_name(m4_path) or loc["run_dir"]
        mk = normalize_model_key(model_display)
        model_display_by_key[mk] = model_display

        for pr in pairs:
            pairs_by_model[mk].add(pr)
            all_pairs.add(pr)

        if args.skip_per_run:
            # Still warm the cache by running analysis even when per_run output is disabled
            compute_safe_alternative_analysis(
                pairs,
                osv_index=osv_index,
                cve_cache=cve_cache,
                pypi_info_dir=PYPI_INFO,
                reference_cutoff=ref_dt,
                pair_cache=pair_cache,
            )
            continue

        summary, _ = compute_safe_alternative_analysis(
            pairs,
            osv_index=osv_index,
            cve_cache=cve_cache,
            pypi_info_dir=PYPI_INFO,
            reference_cutoff=ref_dt,
            pair_cache=pair_cache,
        )
        per_run.append(
            {
                "m4_path": str(m4_path.resolve()),
                "mode": loc["mode"],
                "run_dir": loc["run_dir"],
                "py_tag": loc["py_tag"],
                "model_name": model_display,
                "model_key": mk,
                **summary,
            }
        )

    by_model: dict[str, Any] = {}
    for mk, pset in sorted(pairs_by_model.items(), key=lambda x: x[0]):
        sm, _ = compute_safe_alternative_analysis(
            pset,
            osv_index=osv_index,
            cve_cache=cve_cache,
            pypi_info_dir=PYPI_INFO,
            reference_cutoff=ref_dt,
            pair_cache=pair_cache,
        )
        display = model_display_by_key.get(mk, mk)
        by_model[mk] = {"model_display_name": display, **sm}

    global_summary, _ = compute_safe_alternative_analysis(
        all_pairs,
        osv_index=osv_index,
        cve_cache=cve_cache,
        pypi_info_dir=PYPI_INFO,
        reference_cutoff=ref_dt,
        pair_cache=pair_cache,
    )

    out: dict[str, Any] = {
        "config": {
            "outputs_d2": str(root.resolve()),
            "python_version_filter": args.python_version,
            "reference_date": args.reference_date,
            "m4_files_scanned": len(m4_files),
            "pair_cache_size": len(pair_cache),
        },
        "global": global_summary,
        "by_model": by_model,
        "per_run": per_run if not args.skip_per_run else [],
    }

    out_path = args.output_json or (root / "m4_safe_alt_major_all_runs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console: global summary + per-model leaderboard
    print(json.dumps({"global": global_summary, "models": len(by_model), "runs": len(m4_files)}, ensure_ascii=False, indent=2))
    print()
    g = global_summary
    print(
        f"[global] unique vulnerable (TPL,version)={g['unique_vulnerable_tpl_versions']}, "
        f"analyzed={g['analyzed_pairs']}, with_same_major_safe_alt={g['with_safe_alternative_same_major']}, "
        f"pct(among analyzed)={100.0 * (g['ratio_among_analyzed'] or 0):.2f}%"
    )
    print("\n[by_model] union of (TPL,version) pairs:")
    for mk, row in sorted(by_model.items(), key=lambda x: (-(x[1].get("with_safe_alternative_same_major") or 0), x[0])):
        sm = row
        r = sm.get("ratio_among_analyzed") or 0.0
        print(
            f"  {sm.get('model_display_name', mk)!s}: "
            f"unique={sm['unique_vulnerable_tpl_versions']}, "
            f"with_alt={sm['with_safe_alternative_same_major']}, "
            f"pct={100.0 * r:.2f}%"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
