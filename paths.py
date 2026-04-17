"""Centralized path configuration for llm_tpl.

Usage (repository root; optional: ``source .venv/bin/activate``):
    Import this module from tools launched via ``python -m …`` (keep the working directory at the repo root)::

        from paths import PROJECT_ROOT, OUTPUTS

Directory layout:
- global_cache: shared caches for all experiments (pypi versions, cve, mapping, etc.)
- resources: raw input data (datasets, StackOverflow, OSV, etc.)
- outputs: all output files (run results, metrics, inference outputs)
- logs: log files
- .cache (XDG_CACHE_HOME): application caches (uv, HuggingFace)
"""

from pathlib import Path

# Project root: parent of this file
PROJECT_ROOT = Path(__file__).resolve().parent

GLOBAL_CACHE = PROJECT_ROOT / "global_cache"
OUTPUTS = PROJECT_ROOT / "outputs"
LOGS = PROJECT_ROOT / "logs"
XDG_CACHE_DIR = PROJECT_ROOT / ".cache"

# need to download and put these files in the resources directory
RESOURCES = PROJECT_ROOT / "resources"
PYPI_INFO = RESOURCES / "pypi_info"
OSV_INDEX = RESOURCES / "osv#pypi#vulns#20260301"
STACKOVERFLOW = RESOURCES / "stackoverflow_20251231"
TOP_PYPI_PACKAGES = RESOURCES / "top-pypi-packages.min.json"
CVE_DUMP = RESOURCES / "2026-02-27_all_CVEs_at_midnight"
BIGCODEBENCH = RESOURCES / "BigCodeBench-v0.1.4-local.jsonl"
STACKEXCHANGE_INDEX_DB = RESOURCES / "stackexchange_index.sqlite3"
ANSWER_TIME_INDEX_DB = RESOURCES / "answer_time_index.sqlite3"

def ensure_dirs() -> None:
    """Create base directories if they do not exist."""
    for d in (GLOBAL_CACHE, RESOURCES, OUTPUTS, LOGS, XDG_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Common subpaths under global_cache
def global_cache_path(*parts: str) -> Path:
    return GLOBAL_CACHE.joinpath(*parts)


def resources_path(*parts: str) -> Path:
    return RESOURCES.joinpath(*parts)


def outputs_path(*parts: str) -> Path:
    return OUTPUTS.joinpath(*parts)


def logs_path(*parts: str) -> Path:
    return LOGS.joinpath(*parts)
