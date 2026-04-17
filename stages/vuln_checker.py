"""Vulnerability lookup and CVSS enrichment utilities.

Usage (import-only)::

    from stages.vuln_checker import check_vulnerability, load_osv_index
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)
from packaging.version import InvalidVersion, Version
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[override]
        return iterable

SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


@dataclass
class VulnResult:
    pypi_name: str
    version: str
    is_vulnerable: bool
    osv_ids: list[str]
    cve_ids: list[str]
    max_severity: str | None
    cvss_score: float | None


def _norm_severity(severity: str | None) -> str | None:
    if not severity:
        return None
    value = str(severity).strip().upper()
    return value if value in SEVERITY_ORDER else None


def _severity_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score <= 0:
        return "NONE"
    if score < 4.0:
        return "LOW"
    if score < 7.0:
        return "MEDIUM"
    if score < 9.0:
        return "HIGH"
    return "CRITICAL"


def _max_severity(severities: list[str], score: float | None = None) -> str | None:
    normalized = [_norm_severity(s) for s in severities]
    normalized = [s for s in normalized if s is not None]
    if score is not None:
        inferred = _severity_from_score(score)
        if inferred is not None:
            normalized.append(inferred)
    if not normalized:
        return None
    return max(normalized, key=lambda x: SEVERITY_ORDER.get(x, -1))


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


def _match_osv_range_events(version: str, events: list[dict[str, str]]) -> bool:
    """Match a version against OSV range events."""
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


def _load_json_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_osv_index(osv_dir: str) -> dict[str, dict[str, Any]]:
    """Build package-level vulnerability index from OSV JSON files.
    Aliases (id + aliases from same advisory) are deduplicated to one canonical id."""
    from stages.osv_utils import build_osv_alias_canonical_map

    canonical_map = build_osv_alias_canonical_map(osv_dir)
    packages: dict[str, dict[str, Any]] = {}
    for name in os.listdir(osv_dir):
        path = os.path.join(osv_dir, name)
        if not os.path.isfile(path) or not name.endswith(".json"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vuln_id = data.get("id")
        canonical_id = canonical_map.get(vuln_id, vuln_id) if vuln_id else None
        aliases = data.get("aliases", [])
        cve_ids = sorted({a for a in aliases if isinstance(a, str) and a.startswith("CVE-")})

        # try to get the package name from the affected item
        affected = data.get("affected", [])
        for item in affected:  # may contain multiple affected items
            pkg = ((item.get("package") or {}).get("name") or "").lower()
            if not pkg:
                continue
            if pkg not in packages:
                packages[pkg] = {"by_version": {}, "ranges": [], "vuln_ids": set()}

            versions = item.get("versions", []) or []
            for version in versions:
                bucket = packages[pkg]["by_version"].setdefault(version, {"osv_ids": set(), "cve_ids": set()})
                if canonical_id:
                    bucket["osv_ids"].add(canonical_id)
                    packages[pkg]["vuln_ids"].add(canonical_id)
                for cve in cve_ids:
                    bucket["cve_ids"].add(cve)

            for range_item in item.get("ranges", []) or []:
                if str(range_item.get("type", "")).upper() != "ECOSYSTEM":
                    continue
                events = range_item.get("events", []) or []
                if not events:
                    continue
                packages[pkg]["ranges"].append(
                    {
                        "osv_id": canonical_id,
                        "cve_ids": cve_ids,
                        "events": events,
                    }
                )
                if canonical_id:
                    packages[pkg]["vuln_ids"].add(canonical_id)

    for pkg in packages.values():
        pkg["vuln_ids"] = sorted(pkg["vuln_ids"])
        for ver, detail in pkg["by_version"].items():
            pkg["by_version"][ver] = {
                "osv_ids": sorted(detail["osv_ids"]),
                "cve_ids": sorted(detail["cve_ids"]),
            }
    return packages


def osv_range_mode_stats(osv_dir: str) -> dict[str, Any]:
    """Analyze how often OSV records rely on ranges/events matching."""
    total_advisories = 0
    advisories_with_ranges = 0
    advisories_with_versions = 0
    advisories_ranges_only = 0
    advisories_versions_only = 0
    advisories_with_both = 0

    affected_total = 0
    affected_with_ranges = 0
    affected_with_versions = 0
    range_event_counts = {"introduced": 0, "fixed": 0, "last_affected": 0, "limit": 0}

    for name in os.listdir(osv_dir):
        path = os.path.join(osv_dir, name)
        if not os.path.isfile(path) or not name.endswith(".json"):
            continue
        total_advisories += 1
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        advisory_has_ranges = False
        advisory_has_versions = False
        for item in data.get("affected", []) or []:
            affected_total += 1
            versions = item.get("versions", []) or []
            ranges = item.get("ranges", []) or []
            has_versions = len(versions) > 0
            has_ranges = len(ranges) > 0
            if has_versions:
                affected_with_versions += 1
                advisory_has_versions = True
            if has_ranges:
                affected_with_ranges += 1
                advisory_has_ranges = True
                for range_item in ranges:
                    for event in range_item.get("events", []) or []:
                        for key in range_event_counts:
                            if key in event:
                                range_event_counts[key] += 1

        if advisory_has_ranges:
            advisories_with_ranges += 1
        if advisory_has_versions:
            advisories_with_versions += 1
        if advisory_has_ranges and advisory_has_versions:
            advisories_with_both += 1
        elif advisory_has_ranges:
            advisories_ranges_only += 1
        elif advisory_has_versions:
            advisories_versions_only += 1

    return {
        "total_advisories": total_advisories,
        "advisories_with_ranges": advisories_with_ranges,
        "advisories_with_versions": advisories_with_versions,
        "advisories_ranges_only": advisories_ranges_only,
        "advisories_versions_only": advisories_versions_only,
        "advisories_with_both": advisories_with_both,
        "advisories_with_ranges_ratio": (advisories_with_ranges / total_advisories) if total_advisories else 0.0,
        "affected_total": affected_total,
        "affected_with_ranges": affected_with_ranges,
        "affected_with_versions": affected_with_versions,
        "affected_with_ranges_ratio": (affected_with_ranges / affected_total) if affected_total else 0.0,
        "range_event_counts": range_event_counts,
    }


def _collect_cve_ids_for_package_version(package_info: dict[str, Any], version: str) -> tuple[set[str], set[str]]:
    """Collect CVEs for a package version. Using the specified package_info and version or ranges to collect."""
    osv_ids: set[str] = set()
    cve_ids: set[str] = set()

    detail = (package_info.get("by_version") or {}).get(version)
    if detail:
        osv_ids.update(detail.get("osv_ids", []) or [])
        cve_ids.update(detail.get("cve_ids", []) or [])

    for entry in package_info.get("ranges", []) or []:
        events = entry.get("events", []) or []
        if _match_osv_range_events(version, events):
            osv_id = entry.get("osv_id")
            if osv_id:
                osv_ids.add(osv_id)
            cve_ids.update(entry.get("cve_ids", []) or [])

    return osv_ids, cve_ids


def _extract_cvss_from_metric(metric: dict[str, Any]) -> list[tuple[str | None, float | None]]:
    """Extract CVSS from NVD 5.x metric object. Handles cvssV3_1, cvssV3_0, cvssV2_0, cvssMetricV31, etc."""
    out: list[tuple[str | None, float | None]] = []
    for key, value in metric.items():
        key_lower = key.lower()
        if not (key_lower.startswith("cvssv") or key_lower.startswith("cvssmetricv")) or not isinstance(value, dict):
            continue
        sev = _norm_severity(value.get("baseSeverity"))
        score = value.get("baseScore")
        if score is None:
            score = value.get("impactScore")  # fallback for some formats
        out.append((sev, float(score) if isinstance(score, (int, float)) else None))
    return out


def _extract_cvss_from_cve_record(payload: dict[str, Any]) -> tuple[str | None, float | None]:
    candidates: list[tuple[str | None, float | None]] = []
    containers = payload.get("containers", {})

    cna = containers.get("cna", {})
    for metric in cna.get("metrics", []) or []:
        if isinstance(metric, dict):
            candidates.extend(_extract_cvss_from_metric(metric))

    for adp_item in containers.get("adp", []) or []:
        for metric in (adp_item.get("metrics", []) or []):
            if isinstance(metric, dict):
                candidates.extend(_extract_cvss_from_metric(metric))

    valid_scores = [score for _, score in candidates if score is not None]
    max_score = max(valid_scores) if valid_scores else None
    max_severity = _max_severity([sev for sev, _ in candidates if sev is not None], max_score)
    return max_severity, max_score


def _build_cve_file_index(cve_dump_dir: str, *, index_cache_path: str | None = None) -> dict[str, str]:
    # return mapping: cve_id -> file_path
    dump_dir = str(Path(cve_dump_dir).resolve())
    if index_cache_path:
        cached = _load_json_if_exists(index_cache_path)
        if cached.get("cve_dump_dir") == dump_dir and isinstance(cached.get("mapping"), dict):
            return cached["mapping"]

    mapping: dict[str, str] = {}
    for root, _, files in os.walk(dump_dir):
        for name in files:
            if not name.startswith("CVE-") or not name.endswith(".json"):
                continue
            cve_id = name[:-5]
            mapping[cve_id] = os.path.join(root, name)

    if index_cache_path:
        _save_json(index_cache_path, {"cve_dump_dir": dump_dir, "mapping": mapping})
    return mapping


def _default_global_cache(*parts: str) -> str:
    from paths import global_cache_path
    return str(global_cache_path(*parts))


def enrich_cves_with_local_dump(
    cve_ids: list[str],
    *,
    cve_dump_dir: str | None = None,
    cve_cache_path: str | None = None,
    cve_file_index_cache_path: str | None = None,
    ) -> dict[str, dict[str, Any]]:
    """Read CVSS/severity from local CVE list dump with cache."""
    from paths import CVE_DUMP
    cve_dump_dir = cve_dump_dir or str(CVE_DUMP)
    cve_cache_path = cve_cache_path or _default_global_cache("local_cve_cache.json")
    cve_file_index_cache_path = cve_file_index_cache_path or _default_global_cache("local_cve_file_index.json")
    cache = _load_json_if_exists(cve_cache_path)
    # Re-fetch CVEs with empty cache (e.g. file was not found before, or cache from wrong path)
    needed = [cve for cve in cve_ids if cve not in cache or not cache.get(cve)]
    if not needed:
        return cache

    file_index = _build_cve_file_index(cve_dump_dir, index_cache_path=cve_file_index_cache_path)
    for cve in tqdm(needed, desc="Load local CVE"):
        path = file_index.get(cve)
        if not path or not os.path.exists(path):
            cache[cve] = {}
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            severity, score = _extract_cvss_from_cve_record(payload)
            cache[cve] = {"severity": severity, "score": score}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed reading local CVE {cve} from {path}: {exc}")
            cache[cve] = {}
        _save_json(cve_cache_path, cache)
    return cache


def check_vulnerability(
    pypi_name: str,
    version: str,
    osv_index: dict[str, dict[str, Any]],
    cve_cache: dict[str, dict[str, Any]],
) -> VulnResult:
    """Check vulnerability for a package version. Using the osv_index and cve_cache to check."""

    package_info = osv_index.get(pypi_name, {})
    osv_ids, cve_ids = _collect_cve_ids_for_package_version(package_info, version)
    if not osv_ids and not cve_ids:
        return VulnResult(
            pypi_name=pypi_name,
            version=version,
            is_vulnerable=False,
            osv_ids=[],
            cve_ids=[],
            max_severity=None,
            cvss_score=None,
        )

    scores = []
    severities = []
    for cve in cve_ids:
        item = cve_cache.get(cve, {})
        if item.get("score") is not None:
            scores.append(float(item["score"]))
        if item.get("severity"):
            severities.append(str(item["severity"]))
    max_score = max(scores) if scores else None
    max_severity = _max_severity(severities, max_score)

    return VulnResult(
        pypi_name=pypi_name,
        version=version,
        is_vulnerable=True,
        osv_ids=sorted(osv_ids),
        cve_ids=sorted(cve_ids),
        max_severity=max_severity,
        cvss_score=max_score,
    )


def analyze_records_vulnerabilities(
    records: list[dict],
    *,
    osv_index: dict[str, dict[str, Any]],
    cve_cache_path: str | None = None,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    max_workers: int = 8,
    local_cve_dump_dir: str | None = None,
    cve_file_index_cache_path: str | None = None,
    nvd_cache_path: str | None = None,
) -> list[dict]:
    """Analyze vulnerability status for resolved records with resume support."""
    cve_cache_path = cve_cache_path or _default_global_cache("cve_cache.json")
    checkpoint_path = checkpoint_path or _default_global_cache("vuln_analysis_checkpoint.json")
    from paths import outputs_path
    output_path = output_path or str(outputs_path("vuln", "vulnerability_results.json"))
    cve_file_index_cache_path = cve_file_index_cache_path or _default_global_cache("local_cve_file_index.json")
    from paths import CVE_DUMP
    local_cve_dump_dir = local_cve_dump_dir or str(CVE_DUMP)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Backward compatibility for old callers.
    if nvd_cache_path:
        cve_cache_path = nvd_cache_path

    touched_cves = set()
    for record in records:
        for lib in record.get("per_lib", []):
            version = lib.get("resolved_version")
            if not version:
                continue
            pkg_name = str(lib.get("pypi_name", "")).replace("_", "-")
            package_info = osv_index.get(pkg_name, {})
            _, cve_ids = _collect_cve_ids_for_package_version(package_info, version)
            touched_cves.update(cve_ids)
    cve_cache = enrich_cves_with_local_dump(
        sorted(touched_cves),
        cve_dump_dir=local_cve_dump_dir,
        cve_cache_path=cve_cache_path,
        cve_file_index_cache_path=cve_file_index_cache_path,
    )

    processed_map: dict[str, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        processed_map = {item["task_id"]: item for item in saved.get("records", [])}
        logger.info(f"Resumed vulnerability checkpoint: {len(processed_map)} done.")

    pending = [r for r in records if r.get("task_id") not in processed_map]

    def process_record(record: dict) -> dict:
        seen: dict[tuple[str, str], dict] = {}
        for lib in record.get("per_lib", []):
            version = lib.get("resolved_version")
            if not version:
                continue
            pypi_name = lib["pypi_name"].replace("_", "-")
            key = (pypi_name, version)
            if key in seen:
                continue
            vuln = check_vulnerability(
                pypi_name=pypi_name,
                version=version,
                osv_index=osv_index,
                cve_cache=cve_cache,
            )
            seen[key] = vuln.__dict__
        result = dict(record)
        result["vuln_findings"] = list(seen.values())
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_record, rec): rec.get("task_id") for rec in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Vulnerability analyze"):
            item = future.result()
            processed_map[item["task_id"]] = item
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"records": list(processed_map.values())}, f, ensure_ascii=False, indent=2)

    final_records = [processed_map[r["task_id"]] for r in records if r["task_id"] in processed_map]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
    return final_records
