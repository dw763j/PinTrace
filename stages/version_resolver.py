"""Resolve TPL versions for security-oriented evaluation.

Usage (import-only; consumed by the evaluation pipelines)::

    import stages.version_resolver as version_resolver

This module keeps version resolution independent from vulnerability and
compatibility checks, so each stage can be tested and reused separately.

Failure taxonomy: F0 (Spec/Resolution Failure) is managed here. When
version_exists=False or resolved_version=None for required libs, the
record is classified as F0 in stages/metrics.py.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

import requests
from requests.exceptions import RequestException
from loguru import logger
from packaging.version import InvalidVersion, Version
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from tqdm import tqdm

from stages.utils import get_pkg_name, load_mapping, extract_code_from_content
from stdlib_list import stdlib_list
from collections import defaultdict

# Match fenced code blocks: ```lang?\n...\n```
_CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_.]+)?\s*\n([\s\S]*?)```", re.MULTILINE)

ResolutionMethod = Literal[
    "specified",
    "version_spec_satisfied",
    "latest_at_cutoff",
    "unavailable",
    "pypi_not_found",
    "pypi_http_error",
    "pypi_json_invalid",
    "pypi_request_error",
]
FetchError = Literal["pypi_not_found", "pypi_http_error", "pypi_json_invalid", "pypi_request_error"]

@dataclass
class VersionResolution:
    import_name: str
    pypi_name: str
    specified_version: str | None
    version_spec: str | None  # e.g. ">=1.0", ">=1.0,<2.0" for constraint-based specs
    resolved_version: str | None
    version_exists: bool
    resolution_method: ResolutionMethod
    version_status: str | None = None  # "yanked", "deprecated", etc. when version is non-resolvable


def _parse_reference_date(reference_date: str) -> datetime:
    return datetime.fromisoformat(reference_date.replace("Z", "+00:00"))


def _load_pypi_json_from_cache(package: str, pypi_info_dir: str) -> dict | None:
    path = os.path.join(pypi_info_dir, f"pypi#{package}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load PyPI JSON from cache {path}: {e}")
        return None


def _fetch_pypi_json(
    package: str,
    pypi_info_dir: str,
    *,
    max_attempts: int = 3,
    connect_timeout: float = 15.0,
    read_timeout: float = 60.0,
) -> tuple[dict | None, FetchError | None]:
    cached = _load_pypi_json_from_cache(package, pypi_info_dir)
    if cached is not None:  # cache hit for this package's PyPI JSON
        return cached, None

    url = f"https://pypi.org/pypi/{package}/json"
    timeout = (connect_timeout, read_timeout)
    data: dict | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                if resp.status_code == 404:
                    logger.warning(f"PyPI package not found (404) for package={package!r} url={url}")
                    return None, "pypi_not_found"
                logger.warning(
                    f"PyPI HTTP {resp.status_code} for package={package!r} url={url}"
                )
                return None, "pypi_http_error"
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                logger.warning(f"PyPI response is not valid JSON for package={package!r}: {e}")
                return None, "pypi_json_invalid"
            if not isinstance(data, dict):
                logger.warning(f"PyPI JSON root is not an object for package={package!r}")
                return None, "pypi_json_invalid"
            break
        except RequestException as e:
            logger.warning(
                f"PyPI request failed for package={package!r} "
                f"(attempt {attempt}/{max_attempts}): {type(e).__name__}: {e}"
            )
            if attempt >= max_attempts:
                return None, "pypi_request_error"
            time.sleep(min(2 ** (attempt - 1), 30.0))

    if data is None:
        return None, "pypi_request_error"

    try:
        os.makedirs(pypi_info_dir, exist_ok=True)
        with open(os.path.join(pypi_info_dir, f"pypi#{package}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning(f"Could not write PyPI cache for package={package!r}: {e}")
    return data, None

def _extract_code_blocks_by_lang(content: str) -> dict[str, str]:
    """Extract fenced code blocks by language. Returns {lang: block_content} for first of each type."""
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>")[1].strip()
    result: dict[str, str] = {}
    for m in _CODE_BLOCK_RE.finditer(content):
        lang = (m.group(1) or "").strip().lower()
        body = m.group(2).strip()
        if not lang:
            lang = "generic"
        if lang not in result:
            result[lang] = body
    return result


def _parse_requirements_txt(text: str) -> list[dict]:
    """Parse requirements.txt into [{library, version?, version_spec?}, ...].
    version_spec: constraint string like ">=1.0", ">=1.0,<2.0" for non-pinned specs.
    """
    items: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        version = None
        version_spec = None
        if "==" in line:
            parts = line.split("==", 1)
            library = parts[0].strip()
            version = parts[1].strip() if len(parts) > 1 else None
        elif "~=" in line:
            parts = line.split("~=", 1)
            library = parts[0].strip()
            version = parts[1].strip() if len(parts) > 1 else None
        elif any(op in line for op in [">=", "<=", ">", "<", "!="]):
            for op in [">=", "<=", "!=", ">", "<"]:
                if op in line:
                    library, rest = line.split(op, 1)
                    library = library.strip()
                    version_spec = rest.strip()  # may include more constraints (e.g. "1.0,<2.0")
                    # Build full spec: ">=1.0" or ">=1.0,<2.0"
                    version_spec = op + version_spec
                    break
            else:
                library = line
        else:
            library = line
        if library:
            items.append({
                "library": library,
                "version": version,
                "version_spec": version_spec,
                "import_line": f"requirements.txt: {library}",
            })
    return items


def extract_tpl_versions(content: str, python_version: str = "3.12") -> list[dict]:
    """Extract TPL versions from code string (for D2 / single-record use)."""
    return _extract_tpl_versions_from_code(content, python_version)


def extract_d2_code_and_tpl_versions(
    llm_output_text: str,
    *,
    python_version: str = "3.12",
    mapping: dict | None = None,
) -> tuple[str, list[dict]]:
    """Pull Python snippet and merged TPL versions from raw D2 LLM output.

    If a ``requirements.txt`` fenced block is present, parse it and merge with
    import-line VERSION annotations from the Python block (same rules as D1).
    Otherwise behave like :func:`extract_tpl_versions` on extracted code only.
    """
    if mapping is None:
        mapping, _ = load_mapping()
    blocks = _extract_code_blocks_by_lang(llm_output_text)
    req_items: list[dict] = []
    if "requirements.txt" in blocks:
        req_items = _parse_requirements_txt(blocks["requirements.txt"])
    python_code = (
        blocks.get("python")
        or blocks.get("py")
        or blocks.get("generic")
        or extract_code_from_content(llm_output_text)
    )
    code_items = _extract_tpl_versions_from_code(python_code, python_version)
    if req_items:
        extracted = _merge_requirements_and_code(req_items, code_items, python_version, mapping)
    else:
        extracted = code_items
    return python_code.strip(), extracted


def _extract_tpl_versions_from_code(content: str, python_version: str = "3.12") -> list[dict]:
    """Extract third-party library imports and optional VERSION annotations from code.

    Records every import, setting version=None when no VERSION comment is present.
    """
    STDLIBS_SET = set(stdlib_list(python_version))
    versions = []

    pattern1 = r"^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)(?:\s+as\s+\w+)?(?:\s*#\s*VERSION\s*=\s*([^\s\n]+))?$"
    pattern2 = r"^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+.+?(?:\s*#\s*VERSION\s*=\s*([^\s\n]+))?$"

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern1, line)
        if match:
            lib_name = match.group(1).split(".")[0]
            version = match.group(2).strip() if match.group(2) else None
            record = {"library": lib_name, "version": version, "version_spec": None, "import_line": line}
            if record["library"] not in STDLIBS_SET:
                versions.append(record)
            continue

        match = re.match(pattern2, line)
        if match:
            lib_name = match.group(1).split(".")[0]
            version = match.group(2).strip() if match.group(2) else None
            record = {"library": lib_name, "version": version, "version_spec": None, "import_line": line}
            if record["library"] not in STDLIBS_SET:
                versions.append(record)
    return versions


def _merge_requirements_and_code(
    req_items: list[dict],
    code_items: list[dict],
    python_version: str,
    mapping: dict,
) -> list[dict]:
    """Merge TPL from requirements.txt and code. Prefer req version/version_spec when both present."""
    STDLIBS = set(stdlib_list(python_version))
    req_by_pypi: dict[str, dict] = {}
    for r in req_items:
        key = r["library"].lower().replace("_", "-")
        req_by_pypi[key] = {"version": r.get("version"), "version_spec": r.get("version_spec")}
    seen_pypi: set[str] = set()
    merged: list[dict] = []

    for item in code_items:
        lib = item["library"]
        if lib in STDLIBS:
            continue
        pypi = get_pkg_name(lib, mapping).lower().replace("_", "-")
        req = req_by_pypi.pop(pypi, None)
        version = (req["version"] if req else None) or item.get("version")
        version_spec = (req["version_spec"] if req else None) or item.get("version_spec")
        merged.append({
            "library": lib,
            "version": version,
            "version_spec": version_spec,
            "import_line": item.get("import_line", ""),
        })
        seen_pypi.add(pypi)

    for lib, req in req_by_pypi.items():
        if lib not in seen_pypi:
            merged.append({
                "library": lib,
                "version": req.get("version"),
                "version_spec": req.get("version_spec"),
                "import_line": f"requirements.txt: {lib}",
            })

    return merged


def extract_tpl_versions_from_llm_output(llm_output_path: str, python_version: str = "3.12") -> tuple[list[dict], dict]:
    mapping, _ = load_mapping()
    all_records = []
    version_stats = defaultdict(lambda: defaultdict(int))
    with open(llm_output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_id = record.get("task_id", "")
            llm_output = record.get("llm_output", {})
            content = llm_output.get("content", "")
            third_party_libs = record.get("third_party_libs", [])

            blocks = _extract_code_blocks_by_lang(content)
            req_items: list[dict] = []
            if "requirements.txt" in blocks:
                req_items = _parse_requirements_txt(blocks["requirements.txt"])

            python_code = blocks.get("python") or blocks.get("py") or blocks.get("generic") or extract_code_from_content(content)
            code_items = _extract_tpl_versions_from_code(python_code, python_version)

            extracted_versions = (
                _merge_requirements_and_code(req_items, code_items, python_version, mapping)
                if req_items
                else code_items
            )

            result = {
                "task_id": task_id,
                "expected_libs": third_party_libs,
                "code": python_code,
                "extracted_versions": extracted_versions,
            }
            all_records.append(result)

            for v in extracted_versions:
                if v["version"] is not None:
                    version_stats[v["library"]][v["version"]] += 1

    return all_records, version_stats


def _is_version_yanked(releases: dict, version: str) -> bool:
    """Check if a release version is yanked (PEP 592). Any file yanked => whole release yanked."""
    files = releases.get(version, [])
    for file_meta in files:
        if file_meta.get("yanked") is True:
            return True
    return False


def _build_release_time_index(pypi_json: dict, *, exclude_yanked: bool = True) -> dict[str, datetime]:
    """Build version -> earliest upload time. When exclude_yanked=True, yanked versions are omitted."""
    releases = pypi_json.get("releases", {})
    index: dict[str, datetime] = {}
    for version, files in releases.items():
        if not files:
            continue
        if exclude_yanked and _is_version_yanked(releases, version):
            continue
        timestamps = []
        for file_meta in files:
            ts = file_meta.get("upload_time_iso_8601") or file_meta.get("upload_time")
            if not ts:
                continue
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                continue
        if timestamps:
            index[version] = min(timestamps)
    return index


def pick_latest_version_at_cutoff(release_times: dict[str, datetime], reference_date: str) -> str | None:
    cutoff = _parse_reference_date(reference_date)
    candidates: list[tuple[Version, str]] = []
    for version, release_time in release_times.items():
        if release_time > cutoff:
            continue
        try:
            parsed = Version(version)
        except InvalidVersion:
            continue
        candidates.append((parsed, version))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def pick_latest_satisfying_spec(
    release_times: dict[str, datetime],
    version_spec: str,
    reference_date: str,
) -> str | None:
    """Pick latest version at cutoff that satisfies the given specifier (e.g. >=1.0)."""
    try:
        spec = SpecifierSet(version_spec)
    except InvalidSpecifier:
        return None
    cutoff = _parse_reference_date(reference_date)
    candidates: list[tuple[Version, str]] = []
    for version, release_time in release_times.items():
        if release_time > cutoff:
            continue
        try:
            parsed = Version(version)
            if version in spec:
                candidates.append((parsed, version))
        except InvalidVersion:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def resolve_version(
    import_name: str,
    specified_version: str | None,
    version_spec: str | None,
    reference_date: str,
    pypi_info_dir: str | None = None,
    mapping: dict | None = None,
) -> VersionResolution:
    from paths import PYPI_INFO
    pypi_info_dir = pypi_info_dir or str(PYPI_INFO)
    if mapping is None:
        mapping, _ = load_mapping()

    pypi_name = get_pkg_name(import_name, mapping)
    pypi_json, fetch_error = _fetch_pypi_json(pypi_name, pypi_info_dir)  # single file
    if pypi_json is None:
        if fetch_error in ("pypi_not_found", "pypi_http_error", "pypi_json_invalid", "pypi_request_error"):
            method: ResolutionMethod = fetch_error
        else:
            method = "unavailable"
        return VersionResolution(
            import_name=import_name,
            pypi_name=pypi_name,
            specified_version=specified_version,
            version_spec=version_spec,
            resolved_version=None,
            version_exists=False if (specified_version or version_spec) else True,
            resolution_method=method,
            version_status=None,
        )

    releases = pypi_json.get("releases", {})
    if specified_version:
        in_releases = specified_version in releases
        is_yanked = in_releases and _is_version_yanked(releases, specified_version)
        exists = in_releases and not is_yanked
        status = "yanked" if is_yanked else None
        return VersionResolution(
            import_name=import_name,
            pypi_name=pypi_name,
            specified_version=specified_version,
            version_spec=version_spec,
            resolved_version=specified_version if exists else None,
            version_exists=exists,
            resolution_method="specified" if exists else "unavailable",
            version_status=status,
        )

    release_times = _build_release_time_index(pypi_json)
    if version_spec:
        latest = pick_latest_satisfying_spec(release_times, version_spec, reference_date)
        return VersionResolution(
            import_name=import_name,
            pypi_name=pypi_name,
            specified_version=None,
            version_spec=version_spec,
            resolved_version=latest,
            version_exists=latest is not None,
            resolution_method="version_spec_satisfied" if latest else "unavailable",
            version_status=None,
        )

    latest = pick_latest_version_at_cutoff(release_times, reference_date)
    return VersionResolution(
        import_name=import_name,
        pypi_name=pypi_name,
        specified_version=None,
        version_spec=None,
        resolved_version=latest,
        version_exists=True,
        resolution_method="latest_at_cutoff" if latest else "unavailable",
        version_status=None,
    )


def _default_resolution_paths():
    from paths import global_cache_path, outputs_path
    return str(global_cache_path("version_resolution_checkpoint.json")), str(outputs_path("resolved", "version_resolved_records.json"))


def _atomic_json_write(path: str, obj: object) -> None:
    """Write JSON atomically so readers never see a half-written file (crash / SIGINT safe)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".version_resolution_", suffix=".tmp.json", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_records_versions(
    all_records: list[dict],
    reference_date: str,
    *,
    pypi_info_dir: str | None = None,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    max_workers: int = 8,
) -> list[dict]:
    """Resolve versions for each extracted import with resume support."""
    from paths import PYPI_INFO
    pypi_info_dir = pypi_info_dir or str(PYPI_INFO)
    checkpoint_path = checkpoint_path or _default_resolution_paths()[0]
    output_path = output_path or _default_resolution_paths()[1]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mapping, _ = load_mapping()
    resolved_map: dict[str, dict] = {}
    done_task_ids: set[str] = set()

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except json.JSONDecodeError as e:
            backup = f"{checkpoint_path}.corrupt.{int(time.time() * 1000)}"
            logger.warning(
                f"Version resolution checkpoint is not valid JSON ({e}); "
                f"renaming to {backup} and starting M3 from scratch for this run."
            )
            try:
                os.replace(checkpoint_path, backup)
            except OSError:
                pass
            saved = {}
        for item in saved.get("records", []):
            resolved_map[item["task_id"]] = item
            done_task_ids.add(item["task_id"])
        if done_task_ids:
            logger.info(f"Resumed version resolution checkpoint: {len(done_task_ids)} done.")

    pending = [r for r in all_records if r.get("task_id") not in done_task_ids]

    def process_record(record: dict) -> dict:
        per_lib = []
        for item in record.get("extracted_versions", []):
            lib = item.get("library") or "unknown"
            try:
                resolution = resolve_version(
                    import_name=lib,
                    specified_version=item.get("version"),
                    version_spec=item.get("version_spec"),
                    reference_date=reference_date,
                    pypi_info_dir=pypi_info_dir,
                    mapping=mapping,
                )
            except Exception as e:
                logger.exception(
                    f"resolve_version unexpected error task_id={record.get('task_id')!r} "
                    f"library={lib!r}: {e}"
                )
                pypi_name = get_pkg_name(lib, mapping)
                resolution = VersionResolution(
                    import_name=lib,
                    pypi_name=pypi_name,
                    specified_version=item.get("version"),
                    version_spec=item.get("version_spec"),
                    resolved_version=None,
                    version_exists=False,
                    resolution_method="unavailable",
                    version_status=None,
                )
            payload = asdict(resolution)
            payload["import_line"] = item.get("import_line")
            per_lib.append(payload)
        result = dict(record)
        result["per_lib"] = per_lib
        return result

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {executor.submit(process_record, rec): rec.get("task_id") for rec in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resolving versions"):
            item = future.result()
            resolved_map[item["task_id"]] = item

            snapshot = {"records": list(resolved_map.values())}
            try:
                _atomic_json_write(checkpoint_path, snapshot)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
    finally:
        # On SIGINT, ContextManager shutdown(wait=True) blocks on worker threads stuck in network I/O.
        executor.shutdown(wait=False, cancel_futures=True)

    final_records = [resolved_map[r.get("task_id")] for r in all_records if r.get("task_id") in resolved_map]  # type: ignore
    _atomic_json_write(output_path, final_records)
    return final_records

