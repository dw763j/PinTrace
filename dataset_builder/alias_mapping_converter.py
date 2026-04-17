"""Convert pipreqs-style mappings into alias tables for dataset_builder.

Usage: imported by the StackOverflow build pipeline (no CLI).

    from dataset_builder.alias_mapping_converter import build_alias_mapping_for_targets
"""
from __future__ import annotations

import json
from pathlib import Path


def _norm_name(name: str) -> str:
    return name.strip().lower()


def _load_pipreqs_mapping(mapping_path: str) -> dict[str, str]:
    payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    raw = payload.get("mapping", payload)
    if not isinstance(raw, dict):
        return {}

    out: dict[str, str] = {}
    for import_name, pypi_name in raw.items():
        if not isinstance(import_name, str) or not isinstance(pypi_name, str):
            continue
        alias = _norm_name(import_name)
        canonical = _norm_name(pypi_name)
        if alias and canonical:
            out[alias] = canonical
    return out


def build_alias_mapping_for_targets(
    *,
    mapping_path: str,
    target_packages: set[str],
    out_path: str,
) -> dict:
    """
    Convert pipreqs mapping (import -> package) to alias mapping (package -> [aliases]).
    """
    target_set = {_norm_name(x) for x in target_packages if isinstance(x, str) and x.strip()}
    if not Path(mapping_path).exists():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("{}", encoding="utf-8")
        return {
            "mapping_path": mapping_path,
            "out_path": out_path,
            "raw_mapping_entries": 0,
            "target_packages": len(target_set),
            "canonical_with_aliases": 0,
            "alias_total": 0,
            "missing_mapping_file": True,
        }

    import_to_pkg = _load_pipreqs_mapping(mapping_path)
    aliases: dict[str, set[str]] = {}
    for import_name, pypi_name in import_to_pkg.items():
        if pypi_name not in target_set:
            continue
        if import_name == pypi_name:
            continue
        aliases.setdefault(pypi_name, set()).add(import_name)

    payload = {pkg: sorted(values) for pkg, values in sorted(aliases.items()) if values}
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "mapping_path": mapping_path,
        "out_path": out_path,
        "raw_mapping_entries": len(import_to_pkg),
        "target_packages": len(target_set),
        "canonical_with_aliases": len(payload),
        "alias_total": sum(len(v) for v in payload.values()),
        "missing_mapping_file": False,
    }
