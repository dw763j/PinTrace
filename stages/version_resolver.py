"""Resolve TPL versions for security-oriented evaluation.

This module keeps version resolution independent from vulnerability and
compatibility checks, so each stage can be tested and reused separately.

Failure taxonomy: F0 (Spec/Resolution Failure) is managed here. When
version_exists=False or resolved_version=None for required libs, the
record is classified as F0 in stages/metrics.py.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import requests
from loguru import logger
from packaging.markers import InvalidMarker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from requests.exceptions import RequestException
from stdlib_list import stdlib_list
from tqdm import tqdm

from stages.utils import extract_code_from_content, get_pkg_name, load_mapping

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


@dataclass(frozen=True)
class UvResolverConfig:
    uv_bin: str = "uv"
    python_version: str = "3.12"
    python_platform: str = "x86_64-unknown-linux-gnu"
    exclude_newer: str = "2026-02-01T00:00:00Z"
    prerelease: str = "if-necessary-or-explicit"
    generate_hashes: bool = True

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_reference_date(reference_date: str) -> datetime:
    return datetime.fromisoformat(reference_date.replace("Z", "+00:00"))


def _load_pypi_json_from_cache(package: str, pypi_info_dir: str) -> dict | None:
    path = os.path.join(pypi_info_dir, f"pypi#{package}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
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
    if cached is not None:  # cached PyPI JSON for this package
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
    """Parse requirement declarations with :class:`packaging.Requirement`.

    The legacy ``library``/``version``/``version_spec`` keys remain available
    for old consumers.  Revision analyses should use ``raw_requirement``,
    ``declaration_source`` and ``constraint_kind`` instead.
    """
    items: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # pip directives (e.g. -r, --index-url) are not dependency advice.
        if line.startswith("-"):
            continue

        # PEP 508 comments start after whitespace.  A URL fragment is part of
        # the requirement and therefore must not be split at an arbitrary '#'.
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if not line:
            continue

        try:
            req = Requirement(line)
        except (InvalidRequirement, InvalidMarker) as exc:
            items.append(
                {
                    "library": line,
                    "version": None,
                    "version_spec": None,
                    "import_line": f"requirements.txt: {line}",
                    "raw_requirement": line,
                    "normalized_name": None,
                    "extras": [],
                    "marker": None,
                    "direct_url": None,
                    "declaration_source": "manifest",
                    "constraint_kind": "invalid",
                    "parse_error": str(exc),
                }
            )
            continue

        specifiers = list(req.specifier)
        exact = (
            len(specifiers) == 1
            and specifiers[0].operator in {"==", "==="}
            and "*" not in specifiers[0].version
        )
        if exact:
            version = specifiers[0].version
            version_spec = None
            constraint_kind = "exact"
        elif specifiers or req.url:
            version = None
            version_spec = str(req.specifier) or None
            constraint_kind = "range"
        else:
            version = None
            version_spec = None
            constraint_kind = "bare"

        items.append(
            {
                "library": req.name,
                "version": version,
                "version_spec": version_spec,
                "import_line": f"requirements.txt: {line}",
                "raw_requirement": line,
                "normalized_name": canonicalize_name(req.name),
                "extras": sorted(req.extras),
                "marker": str(req.marker) if req.marker else None,
                "direct_url": req.url,
                "declaration_source": "manifest",
                "constraint_kind": constraint_kind,
                "parse_error": None,
            }
        )
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
    stdlibs = set(stdlib_list(python_version))
    source_lines = content.splitlines()
    version_comment = re.compile(r"#\s*VERSION\s*=\s*([^\s#]+)", re.IGNORECASE)

    def make_record(lib_name: str, version: str | None, import_line: str) -> dict:
        return {
            "library": lib_name,
            "version": version,
            "version_spec": None,
            "import_line": import_line,
            "raw_requirement": f"{lib_name}=={version}" if version else lib_name,
            "declaration_source": "inline_annotation" if version else "code_import_only",
            "constraint_kind": "exact" if version else "bare",
        }

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Preserve best-effort extraction for malformed model output. The
        # line-oriented fallback intentionally handles only syntactically
        # complete single-module imports; valid code always takes the AST path.
        versions: list[dict] = []
        pattern1 = r"^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)(?:\s+as\s+\w+)?(?:\s*#\s*VERSION\s*=\s*([^\s\n]+))?$"
        pattern2 = r"^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+.+?(?:\s*#\s*VERSION\s*=\s*([^\s\n]+))?$"
        for raw_line in source_lines:
            line = raw_line.strip()
            match = re.match(pattern1, line) or re.match(pattern2, line)
            if not match:
                continue
            lib_name = match.group(1).split(".")[0]
            if lib_name in stdlibs:
                continue
            version = match.group(2).strip() if match.group(2) else None
            versions.append(make_record(lib_name, version, line))
        return versions

    versions = []
    import_nodes = sorted(
        (node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for node in import_nodes:
        if isinstance(node, ast.ImportFrom) and (node.level or not node.module):
            continue
        start = max(node.lineno - 1, 0)
        end = max(getattr(node, "end_lineno", node.lineno), node.lineno)
        import_line = "\n".join(source_lines[start:end]).strip()
        comment_match = version_comment.search(import_line)
        version = comment_match.group(1).strip() if comment_match else None
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        else:
            names = [str(node.module).split(".")[0]]
        for lib_name in names:
            if lib_name not in stdlibs:
                versions.append(make_record(lib_name, version, import_line))
    return versions


def _merge_requirements_and_code(
    req_items: list[dict],
    code_items: list[dict],
    python_version: str,
    mapping: dict,
) -> list[dict]:
    """Merge TPL from requirements.txt and code. Prefer req version/version_spec when both present."""
    STDLIBS = set(stdlib_list(python_version))
    req_by_pypi: dict[str, list[dict]] = defaultdict(list)
    for r in req_items:
        if r.get("constraint_kind") == "invalid":
            # Preserve invalid model declarations for incidence/validity metrics;
            # they cannot safely be matched to an import or sent to uv.
            continue
        key = canonicalize_name(r["library"])
        req_by_pypi[key].append(r)
    seen_pypi: set[str] = set()
    merged: list[dict] = []

    for item in code_items:
        lib = item["library"]
        if lib in STDLIBS:
            continue
        pypi = canonicalize_name(get_pkg_name(lib, mapping))
        reqs = req_by_pypi.pop(pypi, [])
        if reqs:
            first, *rest = reqs
            merged.append({**first, "library": lib, "pypi_name_hint": first["library"], "import_line": item.get("import_line", "")})
            merged.extend(rest)
        else:
            merged.append(item)
        seen_pypi.add(pypi)

    for lib, reqs in req_by_pypi.items():
        if lib not in seen_pypi:
            merged.extend(reqs)

    merged.extend(r for r in req_items if r.get("constraint_kind") == "invalid")

    return merged


def _requirement_text_for_uv(item: dict, mapping: dict) -> str | None:
    """Return a PEP 508 input line while preserving model constraints."""
    if item.get("constraint_kind") == "invalid":
        return None
    if item.get("declaration_source") == "manifest" and item.get("raw_requirement"):
        return str(item["raw_requirement"]).strip()

    import_name = str(item.get("library") or "").strip()
    if not import_name:
        return None
    package = get_pkg_name(import_name, mapping)
    if item.get("version"):
        return f"{package}=={item['version']}"
    if item.get("version_spec"):
        return f"{package}{item['version_spec']}"
    return package


def build_task_requirements(extracted_versions: list[dict], mapping: dict | None = None) -> list[str]:
    """Build deterministic task-level resolver input from direct dependencies."""
    if mapping is None:
        mapping, _ = load_mapping()
    lines: list[str] = []
    seen: set[str] = set()
    for item in extracted_versions:
        line = _requirement_text_for_uv(item, mapping)
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


_LOCK_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def parse_uv_lock_pins(lock_text: str) -> dict[str, str]:
    """Parse exact package pins from uv's hashed requirements.txt output."""
    pins: dict[str, str] = {}
    for line in lock_text.splitlines():
        match = _LOCK_PIN_RE.match(line.strip())
        if match:
            pins[canonicalize_name(match.group(1))] = match.group(2)
    return pins


def _safe_task_slug(task_id: object) -> str:
    raw = str(task_id or "unknown")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")[:80] or "task"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _subprocess_output_text(value: str | bytes | None, *, fallback: str = "") -> str:
    """Return captured subprocess output as JSON-safe text.

    ``subprocess.run(..., text=True, timeout=...)`` can still raise
    ``TimeoutExpired`` with raw ``bytes`` on Linux: CPython's ``communicate()``
    raises before decoding the captured buffers.
    """
    if value is None:
        return fallback
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    return text if text else fallback


def resolve_record_with_uv(
    record: dict,
    *,
    config: UvResolverConfig,
    lock_root: str | Path,
    mapping: dict | None = None,
    timeout_seconds: int = 600,
) -> dict:
    """Compile one task's complete environment and attach origin-aware pins."""
    if mapping is None:
        mapping, _ = load_mapping()
    task_dir = Path(lock_root) / _safe_task_slug(record.get("task_id"))
    task_dir.mkdir(parents=True, exist_ok=True)
    requirements_path = task_dir / "requirements.in"
    lock_path = task_dir / "resolved-20260201.txt"
    lines = build_task_requirements(record.get("extracted_versions", []), mapping)
    requirements_text = "\n".join(lines) + ("\n" if lines else "")
    requirements_path.write_text(requirements_text, encoding="utf-8")

    command = [
        config.uv_bin,
        "pip",
        "compile",
        str(requirements_path),
        "--python-version",
        config.python_version,
        "--python-platform",
        config.python_platform,
        "--exclude-newer",
        config.exclude_newer,
        "--prerelease",
        config.prerelease,
        "--output-file",
        str(lock_path),
    ]
    if config.generate_hashes:
        command.append("--generate-hashes")

    if not lines:
        lock_path.write_text("", encoding="utf-8")
        proc = subprocess.CompletedProcess(command, 0, "", "")
    else:
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(
                command,
                124,
                _subprocess_output_text(exc.stdout),
                _subprocess_output_text(exc.stderr, fallback="resolver timeout"),
            )
    lock_text = lock_path.read_text(encoding="utf-8") if lock_path.exists() else ""
    pins = parse_uv_lock_pins(lock_text)

    isolated_constraints: dict[int, tuple[bool, str | None]] = {}
    if proc.returncode != 0:
        # A bad bare dependency or cross-package conflict must not make every
        # otherwise valid model constraint look invalid in RQ1/RQ2. Validate
        # exact/range declarations independently while retaining task failure
        # for the advice-following environment.
        for index, item in enumerate(record.get("extracted_versions", [])):
            if item.get("declaration_source") not in {"manifest", "inline_annotation"}:
                continue
            if item.get("constraint_kind") not in {"exact", "range"}:
                continue
            line = _requirement_text_for_uv(item, mapping)
            if not line:
                continue
            constraint_in = task_dir / f"constraint-{index}.in"
            constraint_lock = task_dir / f"constraint-{index}.txt"
            constraint_in.write_text(line + "\n", encoding="utf-8")
            constraint_command = list(command)
            constraint_command[3] = str(constraint_in)
            constraint_command[constraint_command.index("--output-file") + 1] = str(constraint_lock)
            try:
                constraint_proc = subprocess.run(
                    constraint_command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                isolated_constraints[index] = (False, None)
                continue
            constraint_text = constraint_lock.read_text(encoding="utf-8") if constraint_lock.exists() else ""
            constraint_pins = parse_uv_lock_pins(constraint_text)
            package_hint = item.get("pypi_name_hint") or item.get("library") or ""
            package_name = canonicalize_name(get_pkg_name(str(package_hint), mapping))
            isolated_constraints[index] = (
                constraint_proc.returncode == 0 and package_name in constraint_pins,
                constraint_pins.get(package_name),
            )

    direct: list[dict] = []
    direct_names: set[str] = set()
    for index, item in enumerate(record.get("extracted_versions", [])):
        import_name = str(item.get("library") or "")
        package_hint = item.get("pypi_name_hint") or import_name
        pypi_name = get_pkg_name(str(package_hint), mapping) if package_hint else import_name
        normalized = canonicalize_name(pypi_name) if pypi_name else ""
        resolved = pins.get(normalized)
        kind = item.get("constraint_kind") or (
            "exact" if item.get("version") else "range" if item.get("version_spec") else "bare"
        )
        source = item.get("declaration_source") or (
            "inline_annotation" if item.get("version") else "code_import_only"
        )
        isolated_valid, isolated_version = isolated_constraints.get(index, (False, None))
        constraint_valid = bool(resolved) if kind in {"exact", "range"} else None
        constraint_resolved_version = resolved if kind in {"exact", "range"} else None
        if index in isolated_constraints:
            constraint_valid = isolated_valid
            constraint_resolved_version = isolated_version

        attributable_version = resolved or constraint_resolved_version
        if kind == "invalid" or not attributable_version:
            origin = "unresolved"
        elif kind == "exact" and source in {"manifest", "inline_annotation"}:
            origin = "model_exact"
        elif kind == "range" and source in {"manifest", "inline_annotation"}:
            origin = "model_constrained_resolver"
        else:
            origin = "unconstrained_resolver"
        payload = {
            **item,
            "import_name": import_name,
            "pypi_name": pypi_name,
            "specified_version": item.get("version"),
            "resolved_version": resolved,
            "version_exists": bool(resolved),
            "constraint_valid": constraint_valid,
            "constraint_resolved_version": constraint_resolved_version,
            "resolution_method": origin,
            "resolution_origin": origin,
            "is_direct": True,
            "dependency_scope": "test_harness" if source == "test_harness" else "direct",
            "version_status": None,
        }
        direct.append(payload)
        if normalized:
            direct_names.add(normalized)

    environment = [dict(item) for item in direct if item.get("resolved_version")]
    for normalized, version in sorted(pins.items()):
        if normalized in direct_names:
            continue
        environment.append(
            {
                "import_name": None,
                "pypi_name": normalized,
                "specified_version": None,
                "version_spec": None,
                "resolved_version": version,
                "version_exists": True,
                "declaration_source": "transitive",
                "constraint_kind": "bare",
                "resolution_origin": "transitive",
                "resolution_method": "transitive",
                "is_direct": False,
                "dependency_scope": "transitive",
                "version_status": None,
            }
        )

    return {
        **record,
        "per_lib": direct,
        "resolved_environment": environment,
        "resolver": {
            "status": "empty" if not lines else "resolved" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "config_hash": config.fingerprint(),
            "requirements_in": requirements_text,
            "lock_text": lock_text,
            "lock_sha256": hashlib.sha256(lock_text.encode("utf-8")).hexdigest() if lock_text else None,
            "command": command,
            "stdout": _subprocess_output_text(proc.stdout)[-4000:],
            "stderr": _subprocess_output_text(proc.stderr)[-8000:],
        },
    }


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
            version_exists=not (specified_version or version_spec),
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


def _json_default(obj: object) -> str:
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _atomic_json_write(path: str, obj: object) -> None:
    """Write JSON atomically so readers never see a half-written file (crash / SIGINT safe)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".version_resolution_", suffix=".tmp.json", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
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
    resolver: str = "legacy",
    uv_config: UvResolverConfig | None = None,
    lock_root: str | Path | None = None,
    resolver_timeout_seconds: int = 600,
) -> list[dict]:
    """Resolve versions for each extracted import with resume support."""
    from paths import PYPI_INFO
    pypi_info_dir = pypi_info_dir or str(PYPI_INFO)
    checkpoint_path = checkpoint_path or _default_resolution_paths()[0]
    output_path = output_path or _default_resolution_paths()[1]
    if resolver not in {"legacy", "uv"}:
        raise ValueError(f"Unsupported resolver: {resolver!r}")
    if resolver == "uv":
        uv_config = uv_config or UvResolverConfig(exclude_newer=reference_date)
        lock_root = Path(lock_root) if lock_root else Path(output_path).with_name("locks")
        config_hash = uv_config.fingerprint()
    else:
        config_hash = hashlib.sha256(
            json.dumps({"resolver": "legacy", "reference_date": reference_date}, sort_keys=True).encode("utf-8")
        ).hexdigest()
    input_hash = hashlib.sha256(
        json.dumps(all_records, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checkpoint_hash = hashlib.sha256(f"{config_hash}:{input_hash}".encode()).hexdigest()
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
        if saved.get("checkpoint_hash") != checkpoint_hash:
            logger.warning("Ignoring version-resolution checkpoint because resolver configuration or input changed.")
        else:
            for item in saved.get("records", []):
                resolved_map[item["task_id"]] = item
                done_task_ids.add(item["task_id"])
        if done_task_ids:
            logger.info(f"Resumed version resolution checkpoint: {len(done_task_ids)} done.")

    pending = [r for r in all_records if r.get("task_id") not in done_task_ids]

    def process_record(record: dict) -> dict:
        if resolver == "uv":
            assert uv_config is not None and lock_root is not None
            return resolve_record_with_uv(
                record,
                config=uv_config,
                lock_root=lock_root,
                mapping=mapping,
                timeout_seconds=resolver_timeout_seconds,
            )
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
            except Exception as e:  # noqa: BLE001 -- isolate one bad model record from the batch
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

            snapshot = {
                "config_hash": config_hash,
                "input_hash": input_hash,
                "checkpoint_hash": checkpoint_hash,
                "records": list(resolved_map.values()),
            }
            try:
                _atomic_json_write(checkpoint_path, snapshot)
            except OSError as e:
                logger.error(f"Failed to save checkpoint: {e}")
    finally:
        # On SIGINT, ContextManager shutdown(wait=True) blocks on worker threads stuck in network I/O.
        executor.shutdown(wait=False, cancel_futures=True)

    final_records = [resolved_map[r.get("task_id")] for r in all_records if r.get("task_id") in resolved_map]  # type: ignore
    _atomic_json_write(output_path, final_records)
    return final_records
