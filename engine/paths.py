"""Centralized path configuration for this repository.

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
XDG_CACHE_DIR = PROJECT_ROOT / ".cache"  # default cache directory for uv, hf, etc. may create many files

# need to download and put these files in the resources directory
RESOURCES = PROJECT_ROOT / "resources"
PYPI_INFO = RESOURCES / "pypi_info"  # created by stages/utils.py
OSV_INDEX = RESOURCES / "osv#pypi#vulns#20260301"  # download from https://github.com/ossf/osv-vulnerabilities/releases
STACKOVERFLOW = RESOURCES / "stackoverflow_20251231"  # download following https://google.github.io/osv.dev/data/
TOP_PYPI_PACKAGES = RESOURCES / "top-pypi-packages.min.json"  # download from https://github.com/hugovk/top-pypi-packages
CVE_DUMP = RESOURCES / "2026-02-27_all_CVEs_at_midnight"  # download from https://github.com/CVEProject/cvelistV5/releases
BIGCODEBENCH = RESOURCES / "BigCodeBench-v0.1.4-local.jsonl"  # gain from https://github.com/bigcode-project/bigcodebench
STACKEXCHANGE_INDEX_DB = RESOURCES / "stackexchange_index.sqlite3"  # created by dataset_builder/stackexchange_builder.py
ANSWER_TIME_INDEX_DB = RESOURCES / "answer_time_index.sqlite3"  # created by dataset_builder/stackexchange_builder.py

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


# Completed evaluation records (M6) and paper analysis products.
EVALUATION = OUTPUTS / "evaluation"
ANALYSIS = OUTPUTS / "analysis"
ANALYSIS_RQ2 = ANALYSIS / "rq2"
ANALYSIS_FUNNELS = ANALYSIS / "funnels"
ANALYSIS_NEIGHBOR = ANALYSIS / "neighbor"
ANALYSIS_ROBUSTNESS = ANALYSIS / "robustness"
PAPER_RESULTS = PROJECT_ROOT / "paper_results"


def display_path(path: Path | str) -> str:
    """Return a repository-relative path for logs and verification reports."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
