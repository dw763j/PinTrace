"""Collect installable TPL versions from completed D1/D2 pipeline experiment results.

Scans all ``m5_compat_records.json`` files under ``outputs/d1/`` and ``outputs/d2/``,
identifies samples where installation succeeded (no ``ty-install-error``), and builds
a statistics table keyed by ``(python_version, pypi_name, version)``.

Output is saved to ``global_cache/installable_tpl_versions.json`` for use in
Ablation 2 prompt construction.

Run from project root:
    python -m scripts.collect_installable_tpl_versions
    python -m scripts.collect_installable_tpl_versions --outputs-root outputs --out global_cache/installable_tpl_versions.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from paths import GLOBAL_CACHE, OUTPUTS, ensure_dirs


def _parse_python_version(path: Path) -> str | None:
    """Extract python version from a path component like ``py312`` → ``'3.12'``."""
    for part in path.parts:
        m = re.match(r"^py(\d)(\d{1,2})$", part)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return None


def _has_ty_install_error(record: dict) -> bool:
    """Return True if the record has any ty-install-error in its compat results."""
    compat = record.get("compat_results") or {}
    ty = compat.get("ty") or {}
    errors = ty.get("errors") or []
    return any(e.get("rule") == "ty-install-error" for e in errors)


def _collect_from_file(
    json_path: Path,
    python_version: str,
    stats: dict[str, dict[str, dict[str, int]]],
    dataset_tag: str,
) -> tuple[int, int]:
    """Read one m5_compat_records.json and accumulate stats.

    Returns (total_records, installable_records).
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return 0, 0

    if not isinstance(records, list):
        logger.warning(f"Unexpected JSON structure (not a list) in {json_path}")
        return 0, 0

    total = len(records)
    installable_count = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        if _has_ty_install_error(record):
            continue

        per_lib = record.get("per_lib") or []
        if not per_lib:
            continue

        installable_count += 1
        for lib in per_lib:
            if not isinstance(lib, dict):
                continue
            pypi_name = lib.get("pypi_name")
            resolved_version = lib.get("resolved_version")
            if not pypi_name or not resolved_version:
                continue
            pypi_name = str(pypi_name).lower().strip()
            resolved_version = str(resolved_version).strip()

            pv_map = stats.setdefault(python_version, {})
            pkg_map = pv_map.setdefault(pypi_name, {})
            pkg_map[resolved_version] = pkg_map.get(resolved_version, 0) + 1

    return total, installable_count


def collect_installable_versions(
    outputs_root: Path,
    *,
    datasets: tuple[str, ...] = ("d1", "d2"),
) -> dict:
    """Scan all m5_compat_records.json under outputs_root/{dataset}/ and aggregate.

    Returns a dict with structure::

        {
            "python_versions": {
                "3.12": {
                    "numpy": {"1.26.4": 42, "2.0.0": 15, ...},
                    ...
                },
                ...
            },
            "summary": {
                "total_files": int,
                "total_records": int,
                "installable_records": int,
                "total_tpl_entries": int,
            }
        }
    """
    stats: dict[str, dict[str, dict[str, int]]] = {}
    total_files = 0
    total_records = 0
    installable_records = 0

    for dataset in datasets:
        dataset_dir = outputs_root / dataset
        if not dataset_dir.exists():
            logger.info(f"Dataset dir not found, skipping: {dataset_dir}")
            continue

        for json_path in sorted(dataset_dir.rglob("m5_compat_records.json")):
            python_version = _parse_python_version(json_path)
            if python_version is None:
                logger.warning(f"Could not parse python version from path: {json_path}")
                continue

            total_files += 1
            t, i = _collect_from_file(json_path, python_version, stats, dataset)
            total_records += t
            installable_records += i
            logger.info(
                f"[{dataset}] {json_path.parent.name}/py{python_version}: "
                f"{i}/{t} installable records"
            )

    total_tpl_entries = sum(
        len(versions)
        for pv in stats.values()
        for versions in pv.values()
    )

    return {
        "python_versions": stats,
        "summary": {
            "total_files": total_files,
            "total_records": total_records,
            "installable_records": installable_records,
            "installable_rate": installable_records / total_records if total_records else 0.0,
            "total_tpl_entries": total_tpl_entries,
            "python_version_count": len(stats),
            "package_count_per_version": {
                pv: len(pkgs) for pv, pkgs in stats.items()
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect installable TPL versions from D1/D2 experiment results."
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="Root outputs directory. Default: project outputs/",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: global_cache/installable_tpl_versions.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["d1", "d2"],
        help="Dataset subdirectories to scan. Default: d1 d2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    outputs_root = Path(args.outputs_root) if args.outputs_root else OUTPUTS
    out_path = Path(args.out) if args.out else GLOBAL_CACHE / "installable_tpl_versions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning outputs under: {outputs_root}")
    result = collect_installable_versions(outputs_root, datasets=tuple(args.datasets))

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    s = result["summary"]
    logger.success(
        f"Done. files={s['total_files']}, records={s['total_records']}, "
        f"installable={s['installable_records']} ({s['installable_rate']:.1%}), "
        f"tpl_entries={s['total_tpl_entries']}"
    )
    logger.success(f"Saved to: {out_path}")

    for pv, count in s.get("package_count_per_version", {}).items():
        logger.info(f"  Python {pv}: {count} distinct packages")


if __name__ == "__main__":
    main()
