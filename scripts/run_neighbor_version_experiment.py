#!/usr/bin/env python3
"""D1/D2 neighbor-version experiment: observable lower bound on "safe but mis-specified" behavior.

Recommended invocation::

    python -m scripts.run_neighbor_version_experiment ...

Inputs (from a finished pipeline tree)::

    outputs/<dataset>/<mode>/<run_name>/<py_tag>/m5_compat_records.json

Two phases:
1) **Pick runs/models**: under the chosen ``py_tag``, rank models per mode (see ``--select-by``).
2) **Search neighborhood**: within the chosen run, run bounded version-space search on baseline tasks
   (see ``--baseline``), re-check ty (and optionally BCB on D1).

Baseline definitions (task-level; tasks must be "safe", i.e. no ``vuln_findings`` hits):
- ``safe_bcb_fail``: ``bcb_test.status == "fail"``
- ``safe_ty_incompat``: ``ty.is_compatible == False``
- ``safe_ty_install_error``: ``ty.errors`` contains ``rule == "ty-install-error"``

Notes:
- D2 has no BCB stage, so ``safe_bcb_fail`` is not supported on D2.

Neighborhood search (lower bound):
- Under bounded (N, K, L1-ordered, ``stop_policy``) search, "not observed" does not mean "impossible";
  metrics are a **lower bound on observable induction**.
- ``--focus-install-failed-tpl`` parses failing TPL names from ty-install-error logs and biases search
  toward those libraries (often with larger ``--focused-neighbor-n``).

Outputs:
- One JSON with run selection, configuration, per-task candidate traces, and summary aggregates.

Metric naming:
- Post-recovery metrics start with ``recover_``; denominator defaults to ``baseline_tasks``.
- Raw baseline counts use the ``baseline_*`` prefix.
"""

from __future__ import annotations
# import sys
import argparse
import os
import json
import re
from pathlib import Path

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable

from loguru import logger
from packaging.version import InvalidVersion, Version
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

from paths import BIGCODEBENCH, OUTPUTS, OSV_INDEX, PYPI_INFO, LOGS, XDG_CACHE_DIR
from stages.compact_checker import check_compatibility_for_pkg, load_bigcodebench_problems
from stages.vuln_checker import (
    _collect_cve_ids_for_package_version,
    load_osv_index,
)
from scripts.notifier_utils import send_experiment_notification


DATASET_CHOICES = ["d1", "d2"]
MODE_CHOICES = ["inline", "inline_no_vuln", "requirements.txt"]
RUN_SELECT_CHOICES = ["safe_bcb_fail", "safe_ty_incompat", "safe_ty_install_error"]
STOP_POLICY_CHOICES = ["none", "first_pass", "first_vuln_pass"]


def _run_stat_rank_key(select_by: str) -> Callable[[RunStat], tuple[int, int, int]]:
    rank_key_by_metric: dict[str, Callable[[RunStat], tuple[int, int, int]]] = {
        "safe_ty_install_error": lambda run: (
            run.safe_ty_install_error_count,
            run.safe_bcb_fail_count,
            run.total_tasks,
        ),
        "safe_ty_incompat": lambda run: (
            run.safe_ty_incompat_count,
            run.safe_bcb_fail_count,
            run.total_tasks,
        ),
        "safe_bcb_fail": lambda run: (
            run.safe_bcb_fail_count,
            run.safe_ty_incompat_count,
            run.total_tasks,
        ),
    }
    return rank_key_by_metric.get(select_by, rank_key_by_metric["safe_bcb_fail"])


def _is_baseline_record(record: dict[str, Any], baseline: str) -> bool:
    compat = record.get("compat_results", {}) if isinstance(record, dict) else {}
    ty = compat.get("ty", {}) if isinstance(compat, dict) else {}
    bcb = compat.get("bcb_test", {}) if isinstance(compat, dict) else {}

    ty_ok = bool(ty.get("is_compatible", False)) if isinstance(ty, dict) else False
    bcb_status = str(bcb.get("status", "")) if isinstance(bcb, dict) else ""
    is_safe = not _task_has_vuln(record)

    if baseline == "safe_ty_install_error":
        return is_safe and _has_ty_install_error(ty)
    if baseline == "safe_ty_incompat":
        return is_safe and (not ty_ok)
    return is_safe and (bcb_status == "fail")


def _parse_ref_date(reference_date: str) -> datetime:
    return datetime.fromisoformat(reference_date.replace("Z", "+00:00"))


def _safe_float_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _is_version_yanked(releases: dict[str, list[dict[str, Any]]], version: str) -> bool:
    files = releases.get(version, [])
    return any(meta.get("yanked") is True for meta in files)


def _normalize_pkg_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _infer_model_name(run_name: str) -> str:
    model = run_name
    for suffix in ("_inline_no_vuln", "_inline", "_requirements.txt", "_req", "-req"):
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            break
    return model


@dataclass
class RunStat:
    mode: str
    run_name: str
    model_name: str
    python_tag: str
    total_tasks: int
    safe_tasks: int
    safe_ty_incompat_count: int
    safe_ty_install_error_count: int
    safe_bcb_fail_count: int
    bcb_total: int


def _task_has_vuln(record: dict[str, Any]) -> bool:
    for f in record.get("vuln_findings", []) or []:
        if bool(f.get("is_vulnerable")):
            return True
    return False


def _has_ty_install_error(ty_result: dict[str, Any]) -> bool:
    """Return True iff ty diagnostics contain install-error signal."""
    if not isinstance(ty_result, dict):
        return False
    errors = ty_result.get("errors", [])
    if not isinstance(errors, list):
        return False
    for err in errors:
        if not isinstance(err, dict):
            continue
        rule = str(err.get("rule", "")).strip().lower()
        if rule == "ty-install-error":
            return True
    return False


def _extract_failed_tpl_from_ty_errors(
    ty_result: dict[str, Any],
    *,
    per_lib: list[dict[str, Any]] | None = None,
) -> set[str]:
    """Parse ty-install-error messages and extract failed package names.

    Sources:
    - ``Failed to build `numpy==1.24.3`` fragments
    - ``Building numpy==1.24.3`` fragments
    - Backtick-wrapped requirement snippets (e.g. ``numpy >1.24.3``)

    When ``per_lib`` is provided, keep names that also appear in the task dependency set to reduce noise.
    """
    if not isinstance(ty_result, dict):
        return set()
    errors = ty_result.get("errors", [])
    if not isinstance(errors, list):
        return set()

    task_pkgs: set[str] = set()
    for lib in per_lib or []:
        pkg = _normalize_pkg_name(str(lib.get("pypi_name", "")))
        if pkg:
            task_pkgs.add(pkg)

    def _parse_pkg_from_token(token: str) -> str | None:
        t = token.strip()
        # requirement-like token: pkg[extra] == 1.2.3 / pkg >= 1.0
        m = re.match(
            r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?(?:\s*(?:===|==|~=|!=|<=|>=|<|>).*)?$",
            t,
        )
        if not m:
            return None
        return _normalize_pkg_name(m.group(1))

    found: set[str] = set()
    for err in errors:
        if not isinstance(err, dict):
            continue
        if str(err.get("rule", "")).strip().lower() != "ty-install-error":
            continue
        msg = str(err.get("message", ""))
        if not msg:
            continue

        # Pattern A: Failed to build `pkg==x.y`
        for m in re.finditer(r"Failed to build `([^`]+)`", msg):
            pkg = _parse_pkg_from_token(m.group(1))
            if pkg:
                found.add(pkg)

        # Pattern B: Building pkg==x.y
        for m in re.finditer(r"Building\s+([A-Za-z0-9_.-]+)==", msg):
            pkg = _normalize_pkg_name(m.group(1))
            if pkg:
                found.add(pkg)

        # Pattern C: generic backtick tokens
        for m in re.finditer(r"`([^`]+)`", msg):
            pkg = _parse_pkg_from_token(m.group(1))
            if pkg:
                found.add(pkg)

        # Pattern D: "xxx==a.b depends on Python>=x"
        for m in re.finditer(r"([A-Za-z0-9_.-]+)==[^\s,]+\s+depends on", msg):
            pkg = _normalize_pkg_name(m.group(1))
            if pkg:
                found.add(pkg)

        # Pattern E: "and you require xxx==a.b" / "and you require xxx"
        for m in re.finditer(r"you require\s+([A-Za-z0-9_.-]+)(?:==[^\s,]+)?", msg):
            pkg = _normalize_pkg_name(m.group(1))
            if pkg:
                found.add(pkg)

        # Pattern F: "Because <pkg> was not found in the package registry"
        for m in re.finditer(r"Because\s+([A-Za-z0-9_.-]+)\s+was not found in the package registry", msg):
            pkg = _normalize_pkg_name(m.group(1))
            if pkg:
                found.add(pkg)

    if task_pkgs:
        narrowed = {p for p in found if p in task_pkgs}
        if narrowed:
            return narrowed
    return found


def _discover_m5_files(outputs_root: Path, python_tag: str, modes: list[str]) -> list[Path]:
    m5_files: list[Path] = []
    for mode in modes:
        mode_dir = outputs_root / mode
        if not mode_dir.is_dir():
            continue
        m5_files.extend(sorted(mode_dir.glob(f"*/{python_tag}/m5_compat_records.json")))
    return m5_files


def _make_task_key(record: dict[str, Any], index: int) -> str:
    task_id = str(record.get("task_id", "")).strip()
    if task_id:
        return f"task_id::{task_id}"
    return f"row_idx::{index}"


def _load_progress_records(progress_file: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    if not progress_file.exists():
        return loaded
    for raw_line in progress_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(row.get("task_key", "")).strip()
        result = row.get("result")
        if not key or not isinstance(result, dict):
            continue
        loaded[key] = result
    return loaded


def _append_progress_record(progress_file: Path, task_key: str, result: dict[str, Any]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "saved_at": datetime.now().isoformat(),
        "task_key": task_key,
        "task_id": str(result.get("task_id", "")),
        "result": result,
    }
    with progress_file.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_existing_task_results_from_output(output_json: Path) -> list[dict[str, Any]]:
    if not output_json.exists():
        return []
    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = data.get("task_results", []) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _collect_run_stat(m5_path: Path) -> RunStat:
    # .../outputs/d1/<mode>/<run_name>/<py_tag>/m5_compat_records.json
    mode = m5_path.parent.parent.parent.name
    run_name = m5_path.parent.parent.name
    python_tag = m5_path.parent.name
    model_name = _infer_model_name(run_name)

    records = json.loads(m5_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        records = []

    total = len(records)
    safe_tasks = 0
    safe_ty_incompat = 0
    safe_ty_install_error = 0
    safe_bcb_fail = 0
    bcb_total = 0

    for r in records:
        compat = r.get("compat_results", {}) if isinstance(r, dict) else {}
        ty = compat.get("ty", {}) if isinstance(compat, dict) else {}
        bcb = compat.get("bcb_test", {}) if isinstance(compat, dict) else {}
        ty_ok = bool(ty.get("is_compatible", False)) if isinstance(ty, dict) else False
        bcb_status = str(bcb.get("status", "")) if isinstance(bcb, dict) else ""

        if bcb_status:
            bcb_total += 1

        is_safe = not _task_has_vuln(r)
        if is_safe:
            safe_tasks += 1
            if _has_ty_install_error(ty):
                safe_ty_install_error += 1
            if not ty_ok:
                safe_ty_incompat += 1
            if bcb_status == "fail":
                safe_bcb_fail += 1

    return RunStat(
        mode=mode,
        run_name=run_name,
        model_name=model_name,
        python_tag=python_tag,
        total_tasks=total,
        safe_tasks=safe_tasks,
        safe_ty_incompat_count=safe_ty_incompat,
        safe_ty_install_error_count=safe_ty_install_error,
        safe_bcb_fail_count=safe_bcb_fail,
        bcb_total=bcb_total,
    )


def _pick_target_run(stats: list[RunStat], select_by: str, target_mode: str | None) -> RunStat:
    if not stats:
        raise ValueError("No run stats found for selection.")
    pool = [s for s in stats if (target_mode is None or s.mode == target_mode)]
    if not pool:
        raise ValueError(f"No runs found under target_mode={target_mode}")

    key_fn = _run_stat_rank_key(select_by)

    return sorted(pool, key=key_fn, reverse=True)[0]


def _load_pypi_json_from_dirs(package: str, pypi_dirs: list[Path]) -> dict[str, Any] | None:
    filename = f"pypi#{package}.json"
    for d in pypi_dirs:
        path = d / filename
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    return None


def _ordered_versions_at_cutoff(
    package: str,
    *,
    pypi_dirs: list[Path],
    reference_date: str,
    exclude_yanked: bool = True,
) -> list[str]:
    """Return comparable versions published on/before ``reference_date``, sorted ascending."""
    data = _load_pypi_json_from_dirs(package, pypi_dirs)
    if not data:
        return []

    releases = data.get("releases", {})
    if not isinstance(releases, dict):
        return []

    cutoff = _parse_ref_date(reference_date)
    candidates: list[tuple[Version, str]] = []
    for ver, files in releases.items():
        if not files:
            continue
        if exclude_yanked and _is_version_yanked(releases, ver):
            continue

        first_time: datetime | None = None
        for meta in files:
            ts = meta.get("upload_time_iso_8601") or meta.get("upload_time")
            if not ts:
                continue
            try:
                t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                continue
            if first_time is None or t < first_time:
                first_time = t
        if first_time is None or first_time > cutoff:
            continue

        try:
            pv = Version(ver)
        except InvalidVersion:
            continue
        candidates.append((pv, ver))

    candidates.sort(key=lambda x: x[0])
    return [v for _, v in candidates]


def _build_pkg_specs_from_per_lib(per_lib: list[dict[str, Any]]) -> list[str]:
    specs: list[str] = []
    for lib in per_lib:
        pypi_name = lib.get("pypi_name")
        if not pypi_name:
            continue
        resolved = lib.get("resolved_version")
        version_spec = lib.get("version_spec")

        # Prefer pinning to resolved_version so candidates stay measurable in the sandbox.
        if resolved:
            specs.append(f"{pypi_name}=={resolved}")
        elif version_spec:
            specs.append(f"{pypi_name}{version_spec}")
        else:
            specs.append(str(pypi_name))
    return specs


def _generate_offset_vectors(
    dims: int,
    neighbor_n: int,
    max_candidates_k: int,
) -> list[tuple[int, ...]]:
    """Generate bounded offset vectors sorted by L1 distance.

    We cannot enumerate the full ``(2N+1)^dims`` grid (memory blow-up); instead enumerate
    increasing L1 shells until ``max_candidates_k`` offsets are collected.
    """
    if dims <= 0:
        return []
    if neighbor_n <= 0 or max_candidates_k <= 0:
        return []

    max_l1 = dims * neighbor_n
    results: list[tuple[int, ...]] = []

    def build_at_l1(target_l1: int) -> None:
        cur = [0] * dims

        def dfs(i: int, remain_l1: int, nonzero_used: int) -> None:
            if len(results) >= max_candidates_k:
                return
            if i == dims:
                if remain_l1 == 0 and nonzero_used > 0:
                    results.append(tuple(cur))
                return

            # Prune: each remaining dim can contribute at most neighbor_n in absolute value
            rest_dims = dims - i
            if remain_l1 > rest_dims * neighbor_n:
                return

            # Try zero first so fewer libraries change early
            cur[i] = 0
            dfs(i + 1, remain_l1, nonzero_used)
            if len(results) >= max_candidates_k:
                return

            # Then non-zero offsets sorted by |v| with negatives first for stable ordering
            upper = min(neighbor_n, remain_l1)
            for abs_v in range(1, upper + 1):
                for v in (-abs_v, abs_v):
                    cur[i] = v
                    dfs(i + 1, remain_l1 - abs_v, nonzero_used + 1)
                    if len(results) >= max_candidates_k:
                        return

            cur[i] = 0

        dfs(0, target_l1, 0)

    for l1 in range(1, max_l1 + 1):
        build_at_l1(l1)
        if len(results) >= max_candidates_k:
            break
    return results


def _candidate_task_has_vuln(
    per_lib: list[dict[str, Any]],
    *,
    osv_index: dict[str, dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Return whether the candidate assignment is vulnerable plus per-library findings.

    For task-level concurrency we only consult the in-memory OSV index (no global CVE cache):
    - call ``_collect_cve_ids_for_package_version`` for each ``(pypi_name, version)``
    - any OSV/CVE hit marks ``is_vulnerable=True``
    - ``max_severity`` / ``cvss_score`` are always ``None`` in this experiment
    """
    findings: list[dict[str, Any]] = []
    has_vuln = False
    for lib in per_lib:
        pkg = _normalize_pkg_name(lib.get("pypi_name", ""))
        ver = lib.get("resolved_version")
        if not pkg or not ver:
            continue
        info = osv_index.get(pkg, {})
        osv_ids, cve_ids = _collect_cve_ids_for_package_version(info, str(ver))
        is_vul = bool(osv_ids or cve_ids)
        findings.append(
            {
                "pypi_name": pkg,
                "version": str(ver),
                "is_vulnerable": is_vul,
                "osv_ids": sorted(osv_ids),
                "cve_ids": sorted(cve_ids),
                "max_severity": None,
                "cvss_score": None,
            }
        )
        if is_vul:
            has_vuln = True
    return has_vuln, findings


def _resolve_neighbor_candidates_for_task(
    record: dict[str, Any],
    *,
    neighbor_n: int,
    max_candidates_k: int,
    reference_date: str,
    pypi_dirs: list[Path],
    specified_only: bool,
    max_searchable_libs: int,
    focused_pypi_names: set[str] | None = None,
    focused_neighbor_n: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build neighbor candidates for one task (no compatibility test) plus metadata."""
    base_per_lib = deepcopy(record.get("per_lib", []))
    if not isinstance(base_per_lib, list):
        base_per_lib = []

    focused_names = {_normalize_pkg_name(x) for x in (focused_pypi_names or set()) if str(x).strip()}
    using_focus = bool(focused_names)
    effective_neighbor_n = focused_neighbor_n if (using_focus and focused_neighbor_n and focused_neighbor_n > 0) else neighbor_n

    # Only libraries with explicit pins/constraints participate in neighborhood search.
    search_idx: list[int] = []
    for i, lib in enumerate(base_per_lib):
        if not lib.get("resolved_version"):
            continue
        if using_focus:
            pkg = _normalize_pkg_name(lib.get("pypi_name", ""))
            if pkg not in focused_names:
                continue
        if specified_only:
            if lib.get("specified_version") is None and lib.get("version_spec") is None:
                continue
        search_idx.append(i)

    # Fallback: if strict filtering removes everything, use any row with resolved_version.
    if not search_idx:
        for i, lib in enumerate(base_per_lib):
            if lib.get("resolved_version"):
                search_idx.append(i)

    truncated_libs = False
    if max_searchable_libs > 0 and len(search_idx) > max_searchable_libs:
        # Cap participating libraries to avoid combinatorial blow-ups.
        search_idx = search_idx[:max_searchable_libs]
        truncated_libs = True

    libs_meta: list[dict[str, Any]] = []
    libs_versions: list[list[str]] = []
    base_indices: list[int] = []
    for i in search_idx:
        lib = base_per_lib[i]
        pkg = _normalize_pkg_name(lib.get("pypi_name", ""))
        base_ver = str(lib.get("resolved_version"))
        ordered = _ordered_versions_at_cutoff(
            pkg,
            pypi_dirs=pypi_dirs,
            reference_date=reference_date,
            exclude_yanked=True,
        )
        if not ordered or base_ver not in ordered:
            continue
        base_idx = ordered.index(base_ver)
        libs_meta.append(
            {
                "per_lib_index": i,
                "pypi_name": pkg,
                "base_version": base_ver,
                "ordered_versions_size": len(ordered),
            }
        )
        libs_versions.append(ordered)
        base_indices.append(base_idx)

    if not libs_meta:
        return [], {
            "reason": "no_searchable_libs",
            "searchable_lib_count": 0,
            "searchable_libs": [],
        }

    vectors = _generate_offset_vectors(
        dims=len(libs_meta),
        neighbor_n=effective_neighbor_n,
        max_candidates_k=max_candidates_k * 4,  # oversample before downstream pruning to K
    )

    candidates: list[dict[str, Any]] = []
    seen_signature: set[tuple[tuple[str, str], ...]] = set()

    for vec in vectors:
        cand_per_lib = deepcopy(base_per_lib)
        valid = True
        per_lib_moves: list[dict[str, Any]] = []
        for dim, offset in enumerate(vec):
            meta = libs_meta[dim]
            idx_base = base_indices[dim]
            idx_new = idx_base + offset
            ordered = libs_versions[dim]
            if idx_new < 0 or idx_new >= len(ordered):
                valid = False
                break
            target_ver = ordered[idx_new]
            pl_i = int(meta["per_lib_index"])
            cand_per_lib[pl_i]["resolved_version"] = target_ver
            per_lib_moves.append(
                {
                    "pypi_name": meta["pypi_name"],
                    "base_version": meta["base_version"],
                    "candidate_version": target_ver,
                    "offset": offset,
                    "base_index": idx_base,
                    "candidate_index": idx_new,
                }
            )
        if not valid:
            continue

        signature = tuple(
            sorted(
                (
                    _normalize_pkg_name(lib.get("pypi_name", "")),
                    str(lib.get("resolved_version", "")),
                )
                for lib in cand_per_lib
                if lib.get("pypi_name") and lib.get("resolved_version")
            )
        )
        if signature in seen_signature:
            continue
        seen_signature.add(signature)

        l1_distance = sum(abs(int(x)) for x in vec)
        candidates.append(
            {
                "offset_vector": list(vec),
                "l1_distance": l1_distance,
                "changed_lib_count": sum(1 for x in vec if x != 0),
                "per_lib_moves": per_lib_moves,
                "per_lib": cand_per_lib,
            }
        )
        if len(candidates) >= max_candidates_k:
            break

    candidates.sort(key=lambda x: (x["l1_distance"], x["changed_lib_count"]))
    return candidates, {
        "reason": "ok",
        "focus_enabled": using_focus,
        "focused_pypi_names": sorted(focused_names),
        "effective_neighbor_n": effective_neighbor_n,
        "max_searchable_libs": max_searchable_libs,
        "searchable_libs_truncated": truncated_libs,
        "searchable_lib_count": len(libs_meta),
        "searchable_libs": libs_meta,
        "generated_candidate_count": len(candidates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neighbor-version experiment on existing D1/D2 outputs.")
    add = parser.add_argument

    value_args: list[tuple[str, dict[str, Any]]] = [
        ("--dataset",       {"type": str, "default": "d1", "choices": DATASET_CHOICES, "help": "Target dataset."}),
        ("--outputs-d1",    {"type": Path, "default": OUTPUTS / "d1", "help": "Root dir for D1 outputs."}),
        ("--outputs-d2",    {"type": Path, "default": OUTPUTS / "d2", "help": "Root dir for D2 outputs."}),
        ("--modes",         {"nargs": "+", "default": None, "choices": MODE_CHOICES, "help": "Modes used to discover candidate runs. If unset, use dataset defaults."}),
        ("--python-version",{"type": str, "default": "3.12", "help": "e.g. 3.12"}),
        ("--target-mode",   {"type": str, "default": None, "help": "If set, only pick target run within this mode."}),
        ("--target-run-name",       {"type": str, "default": None, "help": "If set, skip auto-pick and use this run_name directly (within --target-mode if provided)."}),
        ("--select-by",     {"type": str, "default": "safe_ty_incompat", "choices": RUN_SELECT_CHOICES, "help": "Auto-pick metric for choosing target run."}),
        ("--baseline",      {"type": str, "default": "safe_ty_incompat", "choices": RUN_SELECT_CHOICES, "help": "Baseline task definition for neighbor search."}),
        ("--neighbor-n",    {"type": int, "default": 3, "help": "Per-lib neighborhood radius N."}),
        ("--max-candidates-k",      {"type": int, "default": 40, "help": "Max tested candidates K per task."}),
        ("--max-searchable-libs",   {"type": int, "default": 12, "help": "Max number of libs to include in neighbor search per task (resource safety guard)."}),
        ("--stop-policy",   {"type": str, "default": "first_pass", "choices": STOP_POLICY_CHOICES, "help": "Early-stop policy within each task's candidate scan."}),
        ("--focused-neighbor-n",    {"type": int, "default": 10, "help": "Neighborhood radius N used for focused failed-TPL search."}),
        ("--max-tasks",     {"type": int, "default": 0, "help": "Limit baseline tasks (0 means all)."}),
        ("--reference-date",{"type": str, "default": "2026-02-01T00:00:00+00:00"}),
        ("--bcb-timeout",   {"type": int, "default": 180}),
        ("--uv-install-timeout",    {"type": int, "default": 600, "help": "Timeout seconds for each uv pip install call in compatibility checking."}),
        ("--parallel",      {"type": int, "default": 1, "help": "Number of baseline tasks to process in parallel."}),
        ("--parallel-models",{"type": int, "default": 1, "help": "Number of models to process in parallel when --all-models is enabled."}),
        ("--output-json",   {"type": Path, "default": None, "help": "Default: outputs/<dataset>/neighbor_experiments/<baseline>_<mode>_<run>_py<ver>.json"}),
        ("--progress-dir",  {"type": Path, "default": None, "help": "Directory for per-task progress jsonl files. Default: outputs/<dataset>/neighbor_experiments/_progress"}),
        ("--notify-experiment-name",{"type": str, "default": None, "help": "Optional notification title. Default is auto-generated from dataset/baseline/run."}),
    ]
    flag_args: list[tuple[str, str]] = [
        ("--specified-only", "Search only libs with specified_version/version_spec (recommended for this question)."),
        ("--focus-install-failed-tpl", "When ty-install-error exists, parse failed TPL from logs and focus neighborhood search on those packages."),
        ("--notify", "Try sending experiment-result email via scripts/notifier_utils.py."),
        ("--store-candidates", "Store full candidates_tested in output JSON (larger memory/IO)."),
        ("--all-models", "Run neighbor experiment for every discovered model (instead of one selected run)."),
        ("--resume", "Resume from per-task progress file and skip finished tasks."),
    ]

    for option_name, kwargs in value_args:
        add(option_name, **kwargs)
    for option_name, help_text in flag_args:
        add(option_name, action="store_true", help=help_text)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = str(args.dataset).strip().lower()
    if dataset not in {"d1", "d2"}:
        raise SystemExit(f"Unsupported dataset={args.dataset}")
    outputs_root = args.outputs_d1 if dataset == "d1" else args.outputs_d2
    if not outputs_root.is_dir():
        raise SystemExit(f"outputs root not found: {outputs_root}")

    default_modes = MODE_CHOICES
    modes = list(args.modes) if args.modes else default_modes

    supports_bcb = dataset == "d1"
    if not supports_bcb and args.baseline == "safe_bcb_fail":
        raise SystemExit("D2 has no bcb_test process; baseline=safe_bcb_fail is not supported.")
    if not supports_bcb and args.select_by == "safe_bcb_fail":
        raise SystemExit("D2 has no bcb_test process; select-by=safe_bcb_fail is not supported.")

    project_root = Path(__file__).resolve().parents[1]
    py_tag = f"py{str(args.python_version).replace('.', '')}"
    pypi_dirs = [PYPI_INFO.resolve(), (project_root / "pypi_info").resolve()]
    logger.add(LOGS / "neighbor_version_experiment" / f"{dataset}_{args.baseline}_{args.target_mode}_py{args.python_version}.log")
    
    runtime_cache_dir = XDG_CACHE_DIR
    runtime_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_cache_dir))

    logger.info(f"XDG_CACHE_HOME: {os.environ.get('XDG_CACHE_HOME')}")
    default_progress_dir = outputs_root / "neighbor_experiments" / "_progress"
    progress_root = args.progress_dir or default_progress_dir

    # 1) Collect baseline-related counts for every discovered run
    m5_files = _discover_m5_files(outputs_root, py_tag, modes)
    if not m5_files:
        raise SystemExit(f"No m5_compat_records.json found under {outputs_root} with tag={py_tag}")
    all_stats = [_collect_run_stat(p) for p in m5_files]

    # Rank within each mode (paper-style leaderboard)
    rank_key_fn = _run_stat_rank_key(args.select_by)
    per_mode_rank: dict[str, list[dict[str, Any]]] = {}
    for mode in sorted(set(s.mode for s in all_stats)):
        bucket = [s for s in all_stats if s.mode == mode]
        bucket.sort(key=rank_key_fn, reverse=True)
        per_mode_rank[mode] = [asdict(x) for x in bucket]

    if args.all_models and args.target_run_name:
        raise SystemExit("--all-models cannot be used together with --target-run-name.")

    # 2) Load external corpora (D1: BCB + OSV; D2: ty-only + OSV)
    bcb_problems: dict[str, dict[str, Any]] = {}
    if supports_bcb:
        bcb_problems = load_bigcodebench_problems(str(BIGCODEBENCH))
    else:
        logger.info("dataset=d2, skip loading BigCodeBench problems (ty-only compatibility).")
    osv_index = load_osv_index(str(OSV_INDEX))

    logger.info(f"Loaded bcb_problems: {len(bcb_problems)}")
    logger.info(f"Loaded osv_index: {len(osv_index)}")

    def _run_for_one_task(rec: dict[str, Any], target: RunStat) -> dict[str, Any]:
        """Run neighborhood search + compatibility / vuln checks for one baseline task."""
        task_id = str(rec.get("task_id", ""))
        code = str(rec.get("code", ""))
        bcb_problem = bcb_problems.get(task_id) if supports_bcb else None
        compat = rec.get("compat_results", {}) if isinstance(rec, dict) else {}
        ty_base = compat.get("ty", {}) if isinstance(compat, dict) else {}
        failed_tpl_from_logs = (
            _extract_failed_tpl_from_ty_errors(ty_base, per_lib=rec.get("per_lib", []))
            if args.focus_install_failed_tpl
            else set()
        )

        candidates, cand_meta = _resolve_neighbor_candidates_for_task(
            rec,
            neighbor_n=args.neighbor_n,
            max_candidates_k=args.max_candidates_k,
            reference_date=args.reference_date,
            pypi_dirs=pypi_dirs,
            specified_only=bool(args.specified_only),
            max_searchable_libs=args.max_searchable_libs,
            focused_pypi_names=failed_tpl_from_logs,
            focused_neighbor_n=args.focused_neighbor_n,
        )

        first_ty_ok: dict[str, Any] | None = None
        first_ty_ok_vuln: dict[str, Any] | None = None  # ty clean but vulnerable deps
        first_no_ty_install_error: dict[str, Any] | None = None  # install succeeds (no ty-install-error)
        first_no_ty_install_error_vuln: dict[str, Any] | None = None  # install succeeds yet vulnerable
        first_bcb_pass: dict[str, Any] | None = None
        first_bcb_pass_vuln: dict[str, Any] | None = None
        tested_rows: list[dict[str, Any]] = []
        tested_count = 0

        for c in candidates:
            candidate_per_lib = c["per_lib"]
            pkg_specs = _build_pkg_specs_from_per_lib(candidate_per_lib)
            ty_out = check_compatibility_for_pkg(
                code,
                pkg_specs=pkg_specs,
                python_version=args.python_version,
                bcb_problem=bcb_problem,
                bcb_timeout_seconds=args.bcb_timeout,
                uv_install_timeout_seconds=args.uv_install_timeout,
                project_root=str(project_root),
            )

            ty_ok = bool(ty_out.get("is_compatible", False))
            ty_install_error = _has_ty_install_error(ty_out)
            bcb = ty_out.get("bcb_test") if isinstance(ty_out, dict) else None
            bcb_status = str((bcb or {}).get("status", "")) if isinstance(bcb, dict) else ""

            has_vuln, vuln_findings = _candidate_task_has_vuln(
                candidate_per_lib,
                osv_index=osv_index,
            )

            row = {
                "l1_distance": c["l1_distance"],
                "changed_lib_count": c["changed_lib_count"],
                "offset_vector": c["offset_vector"],
                "per_lib_moves": c["per_lib_moves"],
                "pkg_specs": pkg_specs,
                "ty_compatible": ty_ok,
                "ty_install_error": ty_install_error,
                "bcb_status": bcb_status,
                "task_has_vuln": has_vuln,
                "vuln_findings": vuln_findings,
            }
            tested_count += 1
            if args.store_candidates:
                tested_rows.append(row)

            if first_no_ty_install_error is None and (not ty_install_error):
                first_no_ty_install_error = row
            if first_no_ty_install_error_vuln is None and (not ty_install_error) and has_vuln:
                first_no_ty_install_error_vuln = row
            if first_ty_ok is None and ty_ok:
                first_ty_ok = row
            if first_ty_ok_vuln is None and ty_ok and has_vuln:
                first_ty_ok_vuln = row
            if supports_bcb:
                if first_bcb_pass is None and bcb_status == "pass":
                    first_bcb_pass = row
                if first_bcb_pass_vuln is None and bcb_status == "pass" and has_vuln:
                    first_bcb_pass_vuln = row

            # stop policy:
            # - D1: first_pass / first_vuln_pass follow BCB pass semantics
            # - D2: same flags fall back to ty-only pass semantics
            if args.stop_policy == "first_pass":
                if supports_bcb:
                    if first_bcb_pass is not None:
                        break
                else:
                    if first_ty_ok is not None:
                        break
            if args.stop_policy == "first_vuln_pass":
                if supports_bcb:
                    if first_bcb_pass_vuln is not None:
                        break
                else:
                    if first_ty_ok_vuln is not None:
                        break

        return {
            "task_id": task_id,
            "baseline_type": args.baseline,
            "failed_tpl_from_ty_install_error": sorted(failed_tpl_from_logs),
            "candidate_generation_meta": cand_meta,
            "tested_candidate_count": tested_count,
            "first_no_ty_install_error": first_no_ty_install_error,
            "first_no_ty_install_error_vulnerable": first_no_ty_install_error_vuln,
            "first_ty_compatible": first_ty_ok,
            "first_ty_compatible_vulnerable": first_ty_ok_vuln,
            "first_bcb_pass": first_bcb_pass,
            "first_bcb_pass_vulnerable": first_bcb_pass_vuln,
            "candidates_tested": tested_rows if args.store_candidates else [],
            "generated_candidates": len(candidates),
        }

    def _run_one_selected_target(target: RunStat, output_path: Path) -> dict[str, Any]:
        target_m5 = outputs_root / target.mode / target.run_name / target.python_tag / "m5_compat_records.json"
        records = json.loads(target_m5.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            records = []

        # Keep only tasks matching the requested baseline phenotype
        baseline_records = [r for r in records if _is_baseline_record(r, args.baseline)]
        if args.max_tasks and args.max_tasks > 0:
            baseline_records = baseline_records[: args.max_tasks]
        baseline_task_keys = [_make_task_key(rec, idx) for idx, rec in enumerate(baseline_records)]

        progress_file = progress_root / (
            f"{dataset}_{args.baseline}_{target.mode}_{target.run_name}_{target.python_tag}.progress.jsonl"
        )
        loaded_progress = _load_progress_records(progress_file) if args.resume else {}
        loaded_from_output = 0
        if args.resume:
            existing_rows = _load_existing_task_results_from_output(output_path)
            existing_by_task_id = {
                str(row.get("task_id", "")).strip(): row
                for row in existing_rows
                if str(row.get("task_id", "")).strip()
            }
            for key, rec in zip(baseline_task_keys, baseline_records):
                if key in loaded_progress:
                    continue
                task_id = str(rec.get("task_id", "")).strip()
                if not task_id:
                    continue
                hit = existing_by_task_id.get(task_id)
                if isinstance(hit, dict):
                    loaded_progress[key] = hit
                    loaded_from_output += 1

        resumed_task_count = sum(1 for key in baseline_task_keys if key in loaded_progress)
        pending_pairs: list[tuple[str, dict[str, Any]]] = [
            (key, rec) for key, rec in zip(baseline_task_keys, baseline_records) if key not in loaded_progress
        ]

        # Baseline counts: installs without ty-install-error and how many are vulnerable
        baseline_ty_install_ok = 0
        baseline_ty_install_ok_vuln = 0
        for r in baseline_records:
            ty = ((r.get("compat_results") or {}).get("ty") or {})
            if not _has_ty_install_error(ty):
                baseline_ty_install_ok += 1
                if _task_has_vuln(r):
                    baseline_ty_install_ok_vuln += 1

        logger.info(
            f"[neighbor-exp] selected={target.mode}/{target.run_name}/{target.python_tag}, "
            f"baseline={args.baseline}, tasks={len(baseline_records)}, "
            f"resume={bool(args.resume)}, resumed={resumed_task_count}, "
            f"loaded_from_output={loaded_from_output}, pending={len(pending_pairs)}"
        )

        # Execute neighbor search concurrently per task
        task_results_map: dict[str, dict[str, Any]] = dict(loaded_progress) if args.resume else {}
        recover_ty = 0
        recover_bcb = 0
        recover_bcb_vuln = 0
        total_candidates_tested = 0
        total_candidates_generated = 0
        total_pending = len(pending_pairs)

        if total_pending > 0:
            if args.parallel <= 1:
                for task_key, rec in tqdm(pending_pairs, desc=f"[neighbor-exp] {target.model_name} tasks", unit="task"):
                    res = _run_for_one_task(rec, target)
                    task_results_map[task_key] = res
                    _append_progress_record(progress_file, task_key, res)
            else:
                with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                    futures = {ex.submit(_run_for_one_task, rec, target): task_key for task_key, rec in pending_pairs}
                    for fut in tqdm(
                        as_completed(futures),
                        total=total_pending,
                        desc=f"[neighbor-exp] {target.model_name} tasks",
                        unit="task",
                    ):
                        task_key = futures[fut]
                        res = fut.result()
                        task_results_map[task_key] = res
                        _append_progress_record(progress_file, task_key, res)

        task_results: list[dict[str, Any]] = [task_results_map[key] for key in baseline_task_keys if key in task_results_map]

        # Aggregate summary metrics
        recover_ty_vuln = 0
        recover_no_ty_install_error = 0
        recover_no_ty_install_error_vuln = 0
        for res in task_results:
            if res.get("first_no_ty_install_error") is not None:
                recover_no_ty_install_error += 1
            if res.get("first_no_ty_install_error_vulnerable") is not None:
                recover_no_ty_install_error_vuln += 1
            if res["first_ty_compatible"] is not None:
                recover_ty += 1
            if res.get("first_ty_compatible_vulnerable") is not None:
                recover_ty_vuln += 1
            if res["first_bcb_pass"] is not None:
                recover_bcb += 1
            if res["first_bcb_pass_vulnerable"] is not None:
                recover_bcb_vuln += 1
            total_candidates_generated += int(res.get("generated_candidates", 0))
            total_candidates_tested += int(res.get("tested_candidate_count", 0))

        baseline_n = len(baseline_records)
        summary: dict[str, Any] = {
            "formulas": {
                "recover_no_ty_install_error_rate": "recover_no_ty_install_error_tasks / baseline_tasks (found neighbor without ty-install-error)",
                "recover_no_ty_install_error_vuln_rate": "recover_no_ty_install_error_vuln_tasks / baseline_tasks (no ty-install-error but vulnerable)",
                "recover_ty_rate": "recover_ty_tasks / baseline_tasks",
                "recover_ty_vuln_rate": "recover_ty_vuln_tasks / baseline_tasks (ty clean but vulnerable)",
                "baseline_ty_install_ok": "baseline tasks without ty-install-error",
                "baseline_ty_install_ok_vuln": "baseline tasks without ty-install-error yet vulnerable",
            },
            "lower_bound_note": (
                "Observed rates are lower bounds under configured strategy "
                "(neighbor_n, max_candidates_k, ranking-by-l1, stop_policy)."
            ),
            "baseline_tasks": baseline_n,
            "baseline_ty_install_ok": baseline_ty_install_ok,
            "baseline_ty_install_ok_vuln": baseline_ty_install_ok_vuln,
            "recover_no_ty_install_error_tasks": recover_no_ty_install_error,
            "recover_no_ty_install_error_vuln_tasks": recover_no_ty_install_error_vuln,
            "recover_ty_tasks": recover_ty,
            "recover_ty_vuln_tasks": recover_ty_vuln,
            "recover_no_ty_install_error_rate": _safe_float_div(recover_no_ty_install_error, baseline_n),
            "recover_no_ty_install_error_vuln_rate": _safe_float_div(recover_no_ty_install_error_vuln, baseline_n),
            "recover_ty_rate": _safe_float_div(recover_ty, baseline_n),
            "recover_ty_vuln_rate": _safe_float_div(recover_ty_vuln, baseline_n),
            "avg_generated_candidates_per_task": _safe_float_div(total_candidates_generated, baseline_n),
            "avg_tested_candidates_per_task": _safe_float_div(total_candidates_tested, baseline_n),
            "total_generated_candidates": total_candidates_generated,
            "total_tested_candidates": total_candidates_tested,
        }
        if supports_bcb:
            summary["formulas"]["recover_bcb_rate"] = "recover_bcb_tasks / baseline_tasks"
            summary["formulas"]["recover_vuln_bcb_pass_rate"] = "recover_bcb_vuln_tasks / baseline_tasks"
            summary["recover_bcb_tasks"] = recover_bcb
            summary["recover_bcb_vuln_tasks"] = recover_bcb_vuln
            summary["recover_bcb_rate"] = _safe_float_div(recover_bcb, baseline_n)
            summary["recover_vuln_bcb_pass_rate"] = _safe_float_div(recover_bcb_vuln, baseline_n)

        result = {
            "run_selection": {
                "python_tag": py_tag,
                "select_by": args.select_by,
                "target_mode": args.target_mode,
                "target_run_name_arg": args.target_run_name,
                "selected_run": asdict(target),
                "per_mode_rank": per_mode_rank,
            },
            "experiment_config": {
                "dataset": dataset,
                "supports_bcb": supports_bcb,
                "baseline": args.baseline,
                "neighbor_n": args.neighbor_n,
                "max_candidates_k": args.max_candidates_k,
                "max_searchable_libs": args.max_searchable_libs,
                "stop_policy": args.stop_policy,
                "specified_only": bool(args.specified_only),
                "focus_install_failed_tpl": bool(args.focus_install_failed_tpl),
                "focused_neighbor_n": args.focused_neighbor_n,
                "store_candidates": bool(args.store_candidates),
                "reference_date": args.reference_date,
                "python_version": args.python_version,
                "bcb_timeout": args.bcb_timeout,
                "uv_install_timeout": args.uv_install_timeout,
                "modes": modes,
                "outputs_root": str(outputs_root),
                "pypi_dirs": [str(x) for x in pypi_dirs],
                "osv_index_path": str(OSV_INDEX),
            },
            "summary": summary,
            "progress": {
                "enabled": True,
                "resume": bool(args.resume),
                "progress_file": str(progress_file),
                "baseline_tasks_total": len(baseline_records),
                "resumed_tasks": resumed_task_count,
                "loaded_from_output_tasks": loaded_from_output,
                "newly_finished_tasks": len(pending_pairs),
                "task_results_count": len(task_results),
            },
            "task_results": task_results,
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success(f"[neighbor-exp] done, output={output_path}")
        return {"output_json": str(output_path), "selected_run": asdict(target), "summary": summary}

    # 3) Select target run(s) and execute (single model or --all-models)
    if args.all_models:
        by_model: dict[str, list[RunStat]] = {}
        for stat in all_stats:
            by_model.setdefault(stat.model_name, []).append(stat)
        selected_targets: list[RunStat] = []
        skipped_models: list[dict[str, Any]] = []
        for model_name in sorted(by_model):
            pool = by_model[model_name]
            try:
                selected_targets.append(_pick_target_run(pool, args.select_by, args.target_mode))
            except ValueError as exc:
                skipped_models.append({"model_name": model_name, "reason": str(exc)})
                logger.warning(f"[neighbor-exp] skip model={model_name}, reason={exc}")
        if not selected_targets:
            raise SystemExit(
                f"No selectable runs found for --all-models under target_mode={args.target_mode}, py_tag={py_tag}"
            )
    elif args.target_run_name:
        skipped_models = []
        matched = [
            s
            for s in all_stats
            if s.run_name == args.target_run_name and (args.target_mode is None or s.mode == args.target_mode)
        ]
        if not matched:
            raise SystemExit(
                f"target_run_name={args.target_run_name} not found under target_mode={args.target_mode}, py_tag={py_tag}"
            )
        selected_targets = [matched[0]]
    else:
        skipped_models = []
        selected_targets = [_pick_target_run(all_stats, args.select_by, args.target_mode)]

    def _resolve_output_json_path(target: RunStat) -> Path:
        default_output = (
            outputs_root / "neighbor_experiments" / f"{args.baseline}_{target.mode}_{target.run_name}_{target.python_tag}.json"
        )
        if args.output_json and not args.all_models:
            return args.output_json
        if args.output_json and args.all_models:
            if args.output_json.suffix.lower() == ".json":
                raise SystemExit("--output-json must be a directory path when used with --all-models.")
            return args.output_json / default_output.name
        return default_output

    model_runs: list[dict[str, Any]] = []
    if args.all_models and args.parallel_models > 1:
        with ThreadPoolExecutor(max_workers=args.parallel_models) as ex:
            futures = {
                ex.submit(_run_one_selected_target, target, _resolve_output_json_path(target)): target
                for target in selected_targets
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[neighbor-exp] models",
                unit="model",
            ):
                model_runs.append(fut.result())
    else:
        for target in selected_targets:
            run_result = _run_one_selected_target(target, _resolve_output_json_path(target))
            model_runs.append(run_result)

    if args.all_models:
        manifest = {
            "dataset": dataset,
            "baseline": args.baseline,
            "python_tag": py_tag,
            "target_mode": args.target_mode,
            "select_by": args.select_by,
            "run_count": len(model_runs),
            "skipped_models": skipped_models,
            "runs": model_runs,
        }
        manifest_output = (
            (args.output_json / f"all_models_{args.baseline}_{py_tag}.json")
            if (args.output_json and args.output_json.suffix.lower() != ".json")
            else (outputs_root / "neighbor_experiments" / f"all_models_{args.baseline}_{py_tag}.json")
        )
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success(f"[neighbor-exp] all-models manifest output={manifest_output}")

    if args.notify and model_runs:
        first_run = model_runs[0]
        first_selected = first_run.get("selected_run", {})
        first_summary = first_run.get("summary", {})
        notify_name = args.notify_experiment_name or f"neighbor-exp-{dataset}-{args.baseline}"
        notify_summary = {
            "dataset": dataset,
            "baseline": args.baseline,
            "target_mode": str(first_selected.get("mode", "")),
            "target_run_name": str(first_selected.get("run_name", "")),
            "python_version": args.python_version,
            "runs_executed": len(model_runs),
            "baseline_tasks": first_summary.get("baseline_tasks", 0),
            "recover_ty_rate": first_summary.get("recover_ty_rate"),
            "recover_ty_vuln_rate": first_summary.get("recover_ty_vuln_rate"),
            "avg_tested_candidates_per_task": first_summary.get("avg_tested_candidates_per_task"),
            "first_output_json": first_run.get("output_json"),
        }
        if supports_bcb:
            notify_summary["recover_bcb_rate"] = first_summary.get("recover_bcb_rate")
            notify_summary["recover_vuln_bcb_pass_rate"] = first_summary.get("recover_vuln_bcb_pass_rate")

        sent, msg = send_experiment_notification(
            experiment_name=notify_name,
            status="SUCCESS",
            summary=notify_summary,
        )
        if sent:
            logger.success(f"[neighbor-exp] notification sent: {msg}")
        else:
            logger.warning(f"[neighbor-exp] notification skipped/failed: {msg}")


if __name__ == "__main__":
    main()
