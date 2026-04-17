"""Cache PyPI release lists for third-party imports used in downstream checks.

Usage: import-only helpers (no ``python -m`` entrypoint). From the repository root::

    from stages.fetch_pypi_info import fetch_and_cache_pypi_versions, validate_version_exists
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# import argparse
import ast
import json
import os
# import re
from collections import defaultdict

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from stdlib_list import stdlib_list
from stages.utils import get_pkg_name, get_pypi_versions, load_mapping


def _pypi_versions_cache_file() -> str:
    from paths import global_cache_path
    return str(global_cache_path("pypi_versions_cache.json"))

MAPPING, REPEATED_IMPORT_NAMES = load_mapping()


def parse_libs(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def count_third_party(python_version: str, dataset: str, split: str):
    ds = load_dataset(dataset)
    ds = ds[split]
    third_party_samples = []
    all_third_party = set()
    stdlibs_set = set(stdlib_list(python_version))
    for sample in ds:
        libs = [lib.strip() for lib in parse_libs(sample.get("libs")) if lib]
        third_party = sorted({lib for lib in libs if lib not in stdlibs_set})
        if not third_party:
            continue
        all_third_party.update(third_party)
        third_party_samples.append((sample.get("task_id"), third_party))

    return third_party_samples, all_third_party


def fetch_and_cache_pypi_versions(lib_names: set, cache_file: str | None = None) -> dict:
    cache_file = cache_file or _pypi_versions_cache_file()
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    pypi_names = set([get_pkg_name(lib, MAPPING) for lib in lib_names])

    missing_libs = pypi_names - set(cache.keys())
    if missing_libs:
        logger.info(f"Querying PyPI for {len(missing_libs)} missing packages...")
        for pypi_name in tqdm(missing_libs, desc="PyPI fetch"):
            versions = get_pypi_versions(pypi_name)
            cache[pypi_name] = versions
            if not versions:
                logger.warning(f"  warning: could not fetch versions for {pypi_name!r}")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Version cache saved to: {cache_file}")
    else:
        logger.info(f"All {len(pypi_names)} packages already present in PyPI version cache")

    return cache


# def load_pypi_versions_cache(cache_file: str | None = None) -> dict:
#     cache_file = cache_file or _pypi_versions_cache_file()
#     if os.path.exists(cache_file):
#         with open(cache_file, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {}


def validate_version_exists(pypi_name: str, version: str, versions_cache: dict, python_version: str) -> bool:
    stdlibs_set = set(stdlib_list(python_version))
    if pypi_name in stdlibs_set:
        return True
    if pypi_name in versions_cache:
        return version in versions_cache[pypi_name]
    return False


def load_vul_tpls(osv_path: str):
    vul_tpls = defaultdict(list)
    for vul in os.listdir(osv_path):
        with open(os.path.join(osv_path, vul), "r") as f:
            data = json.load(f)
            versions = data["affected"][0].get("versions", [])
            vul_tpls[data["affected"][0]["package"]["name"].lower()].extend(versions)
    return vul_tpls
