"""Build and load the vulnerable ∩ top-PyPI target package pool.

Usage: imported by ``dataset_builder.build_stackoverflow_dataset`` (no CLI).

    from dataset_builder.package_pool import load_target_packages, build_package_pool
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import argparse
from stages.utils import get_vul_tpls, parse_top_pypi_package_names


@dataclass
class PackagePool:
    vulnerable_top_intersection: list[str]
    vulnerable_all: list[str]
    top_packages: list[str]
    stats: dict


def _norm_name(name: str) -> str:
    return name.strip().lower()


def _load_pipreqs_mapping(mapping_path: str) -> dict[str, str]:
    path = Path(mapping_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("mapping", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for import_name, pypi_name in raw.items():
        if not isinstance(import_name, str) or not isinstance(pypi_name, str):
            continue
        out[_norm_name(import_name)] = _norm_name(pypi_name)
    return out


def build_package_pool(
    *,
    top_pypi_path: str | None = None,
    osv_dir: str | None = None,
    mapping_path: str | None = None,
    top_limit: int = 1000,
    out_path: str | None = None,
) -> PackagePool:
    from paths import GLOBAL_CACHE, OUTPUTS, RESOURCES, TOP_PYPI_PACKAGES, OSV_INDEX, ensure_dirs
    ensure_dirs()
    mapping_path = mapping_path or str(GLOBAL_CACHE / "mapping.json")
    top_pypi_path = top_pypi_path or str(TOP_PYPI_PACKAGES)
    if not os.path.isabs(top_pypi_path):
        top_pypi_path = str(RESOURCES / top_pypi_path)
    osv_dir = osv_dir or str(OSV_INDEX)
    out_path = out_path or str(OUTPUTS / "dataset_builder" / "package_pool.json")
    top_raw = parse_top_pypi_package_names(top_pypi_path, limit=top_limit)
    top_names = sorted({_norm_name(name) for name in top_raw})
    vul_tpls = get_vul_tpls(osv_dir)
    vul_names = sorted({_norm_name(name) for name in vul_tpls.keys()})
    direct_inter = sorted(set(top_names).intersection(vul_names))

    mapping = _load_pipreqs_mapping(mapping_path)
    expanded_vul_names = set(vul_names)
    for import_name, pypi_name in mapping.items():
        if import_name in expanded_vul_names:
            expanded_vul_names.add(pypi_name)
    mapped_inter = sorted(set(top_names).intersection(expanded_vul_names))

    pool = PackagePool(
        vulnerable_top_intersection=mapped_inter,
        vulnerable_all=vul_names,
        top_packages=top_names,
        stats={
            "top_limit": top_limit,
            "mapping_path": mapping_path,
            "mapping_entries": len(mapping),
            "vulnerable_total": len(vul_names),
            "top_total": len(top_names),
            "direct_intersection_total": len(direct_inter),
            "mapped_intersection_total": len(mapped_inter),
        },
    )
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "vulnerable_top_intersection": pool.vulnerable_top_intersection,
                "vulnerable_all": pool.vulnerable_all,
                "top_packages": pool.top_packages,
                "stats": pool.stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return pool

def load_target_packages(args: argparse.Namespace, run_dir: Path) -> set[str]:
    if args.target_packages_path:
        with open(args.target_packages_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return {x.lower() for x in payload if isinstance(x, str)}
        if isinstance(payload, dict):
            if "vulnerable_top_intersection" in payload:
                values = payload.get("vulnerable_top_intersection", [])
                return {x.lower() for x in values if isinstance(x, str)}
            # Accept alias mapping style: {canonical_pkg: [alias1, alias2]}
            if all(isinstance(v, list) for v in payload.values()):
                return {str(k).lower() for k in payload.keys() if isinstance(k, str)}
        raise ValueError("--target-packages-path must be JSON list or dict.")

    pool = build_package_pool(
        top_pypi_path=args.top_pypi_path,
        osv_dir=args.osv_dir,
        mapping_path=args.mapping_path,
        top_limit=args.top_limit,
        out_path=str(run_dir / "package_pool.json"),
    )
    return set(pool.vulnerable_top_intersection)