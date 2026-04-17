"""Build/query the OSV-backed version matrix cache used during evaluation.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m stages.osv_version_matrix --help
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from stages.osv_utils import build_osv_alias_canonical_map


def _compare_versions(left: str, right: str) -> int:
    if left == right:
        return 0
    try:
        lv = Version(left)
        rv = Version(right)
        if lv < rv:
            return -1
        if lv > rv:
            return 1
        return 0
    except InvalidVersion:
        pass

    left_tokens = re.findall(r"\d+|[A-Za-z]+", left)
    right_tokens = re.findall(r"\d+|[A-Za-z]+", right)
    for ltok, rtok in zip(left_tokens, right_tokens):
        if ltok == rtok:
            continue
        if ltok.isdigit() and rtok.isdigit():
            return -1 if int(ltok) < int(rtok) else 1
        return -1 if ltok < rtok else 1
    if len(left_tokens) == len(right_tokens):
        return -1 if left < right else 1
    return -1 if len(left_tokens) < len(right_tokens) else 1


def _is_after_or_equal(version: str, lower: str) -> bool:
    if lower in {"", "0", "*"}:
        return True
    return _compare_versions(version, lower) >= 0


def _match_range_events(version: str, events: list[dict[str, str]]) -> bool:
    affected = False
    for event in events:
        introduced = event.get("introduced")
        if introduced is not None:
            affected = _is_after_or_equal(version, str(introduced))
            continue

        if not affected:
            continue

        fixed = event.get("fixed")
        if fixed is not None:
            return _compare_versions(version, str(fixed)) < 0

        last_affected = event.get("last_affected")
        if last_affected is not None:
            return _compare_versions(version, str(last_affected)) <= 0

        limit = event.get("limit")
        if limit is not None:
            return _compare_versions(version, str(limit)) < 0

    return affected


def _candidate_pkg_names(pkg: str) -> list[str]:
    p = pkg.strip().lower()
    candidates = [p, p.replace("_", "-"), p.replace("-", "_")]
    out = []
    for name in candidates:
        if name not in out:
            out.append(name)
    return out


def _find_pypi_info_path(pkg: str, pypi_info_dir: str) -> tuple[str, Path] | tuple[None, None]:
    base = Path(pypi_info_dir)
    for candidate in _candidate_pkg_names(pkg):
        path = base / f"pypi#{candidate}.json"
        if path.exists():
            return candidate, path
    return None, None


def _load_package_versions(cache: dict[str, list[str]], key: str, path: Path) -> list[str]:
    if key in cache:
        return cache[key]
    payload = json.loads(path.read_text(encoding="utf-8"))
    releases = payload.get("releases", {})
    versions = sorted([str(v) for v in releases.keys()])
    cache[key] = versions
    return versions


def build_osv_version_matrix(
    *,
    osv_dir: str,
    pypi_info_dir: str = "pypi_info_vul_tpls",
) -> dict[str, Any]:
    version_cache: dict[str, list[str]] = {}
    matrix: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    canonical_map = build_osv_alias_canonical_map(osv_dir)

    stats = {
        "total_osv_files": 0,
        "parsed_osv_files": 0,
        "ignored_non_json": 0,
        "ignored_missing_osv_id": 0,
        "ignored_missing_package_in_pypi_info": 0,
        "affected_entries_seen": 0,
        "affected_entries_with_versions_only": 0,
        "affected_entries_with_ranges_only": 0,
        "affected_entries_with_both": 0,
    }
    ignored_packages: set[str] = set()

    for name in sorted(os.listdir(osv_dir)):
        path = os.path.join(osv_dir, name)
        if not os.path.isfile(path):
            continue
        stats["total_osv_files"] += 1
        if not name.endswith(".json"):
            stats["ignored_non_json"] += 1
            continue

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        osv_id = data.get("id")
        if not isinstance(osv_id, str) or not osv_id:
            stats["ignored_missing_osv_id"] += 1
            continue
        stats["parsed_osv_files"] += 1

        for item in data.get("affected", []) or []:
            stats["affected_entries_seen"] += 1
            package = (item.get("package") or {}).get("name")
            if not package:
                continue

            normalized_pkg, pypi_json_path = _find_pypi_info_path(str(package), pypi_info_dir)
            if normalized_pkg is None or pypi_json_path is None:
                ignored_packages.add(str(package).lower())
                stats["ignored_missing_package_in_pypi_info"] += 1
                continue

            all_versions = _load_package_versions(version_cache, normalized_pkg, pypi_json_path)
            all_versions_set = set(all_versions)

            versions = [str(v) for v in (item.get("versions") or [])]
            ranges = [r for r in (item.get("ranges") or []) if str(r.get("type", "")).upper() == "ECOSYSTEM"]
            has_versions = len(versions) > 0
            has_ranges = len(ranges) > 0
            if has_versions and has_ranges:
                stats["affected_entries_with_both"] += 1
            elif has_versions:
                stats["affected_entries_with_versions_only"] += 1
            elif has_ranges:
                stats["affected_entries_with_ranges_only"] += 1

            affected_versions = set(v for v in versions if v in all_versions_set)
            for range_item in ranges:
                events = range_item.get("events", []) or []
                for v in all_versions:
                    if _match_range_events(v, events):
                        affected_versions.add(v)

            canonical_id = canonical_map.get(osv_id, osv_id)
            for v in affected_versions:
                matrix[normalized_pkg][v].add(canonical_id)

    packages_out: dict[str, Any] = {}
    for pkg, by_ver in sorted(matrix.items()):
        versions_out: dict[str, Any] = {}
        for version, osv_ids in sorted(by_ver.items(), key=lambda x: x[0]):
            ordered = sorted(osv_ids)
            versions_out[version] = {
                "vuln_count": len(ordered),
                "osv_ids": ordered,
            }
        packages_out[pkg] = {
            "affected_versions": len(versions_out),
            "versions": versions_out,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "osv_dir": str(Path(osv_dir).resolve()),
        "pypi_info_dir": str(Path(pypi_info_dir).resolve()),
        "stats": {
            **stats,
            "packages_in_matrix": len(packages_out),
            "ignored_packages_sample": sorted(list(ignored_packages))[:200],
        },
        "packages": packages_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build package-version vulnerability matrix from OSV + local PyPI metadata.")
    parser.add_argument("--osv-dir", default=None, help="Default: paths.OSV_INDEX")
    parser.add_argument("--pypi-info-dir", default=None, help="Default: paths.RESOURCES/pypi_info_vul_tpls")
    parser.add_argument("--out-path", default=None, help="Default: global_cache/osv_version_matrix.json")
    return parser.parse_args()


def main() -> None:
    from paths import GLOBAL_CACHE, OSV_INDEX, RESOURCES, ensure_dirs

    args = parse_args()
    ensure_dirs()
    osv_dir = args.osv_dir or str(OSV_INDEX)
    pypi_info_dir = args.pypi_info_dir or str(RESOURCES / "pypi_info_vul_tpls")
    out_path = Path(args.out_path) if args.out_path else GLOBAL_CACHE / "osv_version_matrix.json"
    result = build_osv_version_matrix(
        osv_dir=osv_dir,
        pypi_info_dir=pypi_info_dir,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
