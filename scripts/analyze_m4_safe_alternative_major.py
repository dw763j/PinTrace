#!/usr/bin/env python3
"""
Direction C: availability of same-major, non-vulnerable alternatives before a cutoff.

For each vulnerable ``(pypi_name, version)`` in M4 outputs, scan PyPI metadata for other releases in
the **same major** published on/before the earlier of (a) the vulnerable wheel's first upload time
and (b) an optional global reference timestamp. OSV decides whether candidate builds are clean.

Emits a console summary plus JSON with per-pair detail and skip reasons.

For all models/modes at once, run ``python -m scripts.analyze_m4_safe_alternative_all_runs``.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m scripts.analyze_m4_safe_alternative_major --m4-json path/to/m4_vuln_records.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

# Ensure repository root is importable when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import GLOBAL_CACHE, OSV_INDEX, PYPI_INFO
from stages.version_resolver import _build_release_time_index, _load_pypi_json_from_cache
from stages.vuln_checker import check_vulnerability, load_osv_index


@dataclass
class PairResult:
    pypi_name: str
    version: str
    has_safe_alternative_same_major: bool
    vulnerable_release_time: str | None
    cutoff_iso: str | None
    candidate_safe_versions: list[str]
    reason_skip: str | None


def _parse_cutoff_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _same_major(v1: str, v2: str) -> bool:
    try:
        return Version(v1).major == Version(v2).major
    except InvalidVersion:
        return False


def collect_vulnerable_pairs(records: list[dict]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for rec in records:
        for vf in rec.get("vuln_findings") or []:
            if not vf.get("is_vulnerable"):
                continue
            pkg = str(vf.get("pypi_name") or "").replace("_", "-").strip().lower()
            ver = str(vf.get("version") or "").strip()
            if pkg and ver:
                pairs.add((pkg, ver))
    return pairs


def analyze_pair(
    *,
    pypi_name: str,
    vuln_version: str,
    osv_index: dict[str, Any],
    cve_cache: dict[str, Any],
    pypi_info_dir: Path,
    reference_cutoff: datetime | None,
) -> PairResult:
    """``reason_skip`` is set when PyPI metadata or the cutoff cannot be resolved."""
    cached = _load_pypi_json_from_cache(pypi_name, str(pypi_info_dir))
    if not cached:
        return PairResult(
            pypi_name,
            vuln_version,
            False,
            None,
            None,
            [],
            "no_pypi_json",
        )

    release_times = _build_release_time_index(cached, exclude_yanked=True)
    t_vuln = release_times.get(vuln_version)
    vuln_iso = t_vuln.isoformat() if t_vuln else None

    cutoff = t_vuln
    if reference_cutoff is not None:
        if cutoff is None:
            cutoff = reference_cutoff
        else:
            cutoff = min(cutoff, reference_cutoff)
    if cutoff is None:
        return PairResult(
            pypi_name,
            vuln_version,
            False,
            vuln_iso,
            None,
            [],
            "no_release_time_and_no_reference_cutoff",
        )

    cutoff_iso = cutoff.isoformat()
    try:
        _ = Version(vuln_version)
    except InvalidVersion:
        return PairResult(
            pypi_name,
            vuln_version,
            False,
            vuln_iso,
            cutoff_iso,
            [],
            "invalid_vulnerable_version",
        )

    safe_candidates: list[str] = []
    for other_ver, t_other in release_times.items():
        if other_ver == vuln_version:
            continue
        if not _same_major(other_ver, vuln_version):
            continue
        if t_other > cutoff:
            continue
        try:
            vr = check_vulnerability(pypi_name, other_ver, osv_index, cve_cache)
        except Exception:
            continue
        if not vr.is_vulnerable:
            safe_candidates.append(other_ver)

    # Deduplicate and sort by packaging.Version when possible
    def _sort_key(v: str) -> tuple[int, Version]:
        try:
            return (0, Version(v))
        except InvalidVersion:
            return (1, Version("0"))

    safe_candidates = sorted(set(safe_candidates), key=_sort_key)

    return PairResult(
        pypi_name,
        vuln_version,
        bool(safe_candidates),
        vuln_iso,
        cutoff_iso,
        safe_candidates,
        None,
    )


def compute_safe_alternative_analysis(
    pairs: set[tuple[str, str]],
    *,
    osv_index: dict[str, Any],
    cve_cache: dict[str, Any],
    pypi_info_dir: Path,
    reference_cutoff: datetime | None,
    pair_cache: dict[tuple[str, str], PairResult] | None = None,
) -> tuple[dict[str, Any], list[PairResult]]:
    """Analyze every unique ``(pypi_name, version)`` pair and return aggregate + per-pair rows.

    Optional ``pair_cache`` memoizes repeated pairs when scanning many M4 dumps.
    """
    results: list[PairResult] = []
    for pkg, ver in sorted(pairs):
        key = (pkg, ver)
        if pair_cache is not None and key in pair_cache:
            results.append(pair_cache[key])
            continue
        r = analyze_pair(
            pypi_name=pkg,
            vuln_version=ver,
            osv_index=osv_index,
            cve_cache=cve_cache,
            pypi_info_dir=pypi_info_dir,
            reference_cutoff=reference_cutoff,
        )
        if pair_cache is not None:
            pair_cache[key] = r
        results.append(r)
    analyzed = [r for r in results if r.reason_skip is None]
    skipped = [r for r in results if r.reason_skip is not None]
    with_alt = sum(1 for r in analyzed if r.has_safe_alternative_same_major)
    n_analyzed = len(analyzed)
    summary = {
        "unique_vulnerable_tpl_versions": len(pairs),
        "analyzed_pairs": n_analyzed,
        "skipped_pairs": len(skipped),
        "with_safe_alternative_same_major": with_alt,
        "ratio_among_analyzed": (with_alt / n_analyzed) if n_analyzed else None,
        "ratio_among_all_unique": (with_alt / len(pairs)) if pairs else None,
    }
    return summary, results


def load_cve_cache_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M4: same-major safe alternative availability before vulnerable release / reference date."
    )
    p.add_argument(
        "--m4-json",
        type=Path,
        required=True,
        help="Path to m4_vuln_records.json (list of records)",
    )
    p.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="Optional ISO cutoff (e.g. 2025-12-31T00:00:00+00:00); min(reference, vulnerable upload) bounds candidates",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Detailed JSON output path (default: <m4-json stem>_safe_alt_major.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.m4_json.is_file():
        raise SystemExit(f"file not found: {args.m4_json}")

    raw = json.loads(args.m4_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("m4 json must be a list of records")

    pairs = collect_vulnerable_pairs(raw)
    osv_index = load_osv_index(str(OSV_INDEX))
    cve_cache = load_cve_cache_optional(GLOBAL_CACHE / "local_cve_cache.json")

    ref_dt = _parse_cutoff_datetime(args.reference_date)
    out_path = args.output_json or (args.m4_json.parent / f"{args.m4_json.stem}_safe_alt_major.json")

    summary_core, results = compute_safe_alternative_analysis(
        pairs,
        osv_index=osv_index,
        cve_cache=cve_cache,
        pypi_info_dir=PYPI_INFO,
        reference_cutoff=ref_dt,
    )
    summary = {
        "m4_json": str(args.m4_json.resolve()),
        "pypi_info_dir": str(PYPI_INFO.resolve()),
        "osv_dir": str(OSV_INDEX.resolve()),
        "reference_date": args.reference_date,
        **summary_core,
    }

    payload = {
        "summary": summary,
        "per_pair": [asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with_alt = summary_core["with_safe_alternative_same_major"]
    n_analyzed = summary_core["analyzed_pairs"]
    pct_a = 100.0 * with_alt / n_analyzed if n_analyzed else 0.0
    pct_u = 100.0 * with_alt / len(pairs) if pairs else 0.0
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(
        f"Among {n_analyzed} vulnerable (TPL, version) pairs with a resolved cutoff, "
        f"{with_alt} ({pct_a:.1f}%) admit a non-vulnerable same-major release before the cutoff."
    )
    print(
        f"That is {pct_u:.1f}% of all {len(pairs)} unique vulnerable pairs "
        "(denominator includes skipped pairs lacking PyPI/timing data)."
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
