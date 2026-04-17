"""Aggregate paper-facing metrics from resolved records.

Usage (import-only)::

    from stages.metrics import aggregate_metrics
"""

from __future__ import annotations

# import csv
import json
import os
from collections import Counter

from stages.failure_classifier import (
    classify_ty_error,
    classify_bcb_error_type,
)


def aggregate_metrics(records: list[dict], model_name: str) -> dict:
    total_tasks = len(records)

    # lib-level
    total_lib = 0
    specified_lib = 0
    valid_specified = 0
    vuln_lib = 0
    version_status_counter: Counter = Counter()
    severity_counter: Counter = Counter()

    # task-level vulnerability
    task_has_vuln = 0

    # task-level TY compatibility
    ty_task_compatible = 0
    ty_task_incompatible = 0

    # task-level security × TY compatibility
    task_safe_ty_compat = 0
    task_unsafe_ty_compat = 0
    task_safe_ty_incompat = 0
    task_unsafe_ty_incompat = 0

    # BCB task-level
    bcb_task_pass = 0
    bcb_task_fail = 0
    bcb_task_error_like = 0
    bcb_task_total = 0
    safe_bcb_pass = 0
    safe_bcb_fail = 0
    unsafe_bcb_pass = 0
    unsafe_bcb_fail = 0

    # taxonomies
    ty_error_type_counter: Counter = Counter()
    ty_rule_counter: Counter = Counter()
    ty_install_category_counter: Counter = Counter()
    bcb_fail_error_type_counter: Counter = Counter()
    bcb_from_ty_install = 0
    bcb_from_ty_env = 0

    for rec in records:
        vuln_findings = rec.get("vuln_findings", [])
        findings_by_key = {
            (str(f.get("pypi_name") or "").replace("_", "-").lower(), f.get("version")): f
            for f in vuln_findings
            if isinstance(f, dict)
        }
        has_vuln = any(bool(f.get("is_vulnerable")) for f in vuln_findings if isinstance(f, dict))
        if has_vuln:
            task_has_vuln += 1

        compat_results = rec.get("compat_results", {})
        ty_result = compat_results.get("ty", {}) if isinstance(compat_results, dict) else {}
        is_task_ty_compatible = bool(ty_result.get("is_compatible", False))
        if is_task_ty_compatible:
            ty_task_compatible += 1
            ty_rule_counter["ty-compatible"] += 1
        else:
            ty_task_incompatible += 1

        if has_vuln and is_task_ty_compatible:
            task_unsafe_ty_compat += 1
        elif has_vuln and not is_task_ty_compatible:
            task_unsafe_ty_incompat += 1
        elif (not has_vuln) and is_task_ty_compatible:
            task_safe_ty_compat += 1
        else:
            task_safe_ty_incompat += 1

        ty_errors = ty_result.get("errors") if isinstance(ty_result, dict) else None
        if isinstance(ty_errors, list):
            for err in ty_errors:
                if not isinstance(err, dict):
                    continue
                rule = str(err.get("rule") or "").strip()
                ty_rule_counter[rule or "unknown-rule"] += 1

        ty_error_type, ty_install_category = classify_ty_error(ty_result)
        if ty_error_type != "none":
            ty_error_type_counter[ty_error_type] += 1
            if ty_error_type == "install_error" and ty_install_category:
                ty_install_category_counter[ty_install_category] += 1

        bcb = compat_results.get("bcb_test") if isinstance(compat_results, dict) else None
        bcb_status = str(bcb.get("status", "")) if isinstance(bcb, dict) else ""
        if bcb_status:
            bcb_task_total += 1
            if bcb_status == "pass":
                bcb_task_pass += 1
                if has_vuln:
                    unsafe_bcb_pass += 1
                else:
                    safe_bcb_pass += 1
            elif bcb_status == "fail":
                bcb_task_fail += 1
                if has_vuln:
                    unsafe_bcb_fail += 1
                else:
                    safe_bcb_fail += 1
            else:
                bcb_task_error_like += 1
                if has_vuln:
                    unsafe_bcb_fail += 1
                else:
                    safe_bcb_fail += 1

        bcb_err_type = classify_bcb_error_type(bcb if isinstance(bcb, dict) else None)
        if bcb_err_type != "none":
            if ty_error_type == "install_error":
                bcb_from_ty_install += 1
            elif ty_error_type == "env_error":
                bcb_from_ty_env += 1
            elif bcb_status == "fail":
                bcb_fail_error_type_counter[bcb_err_type] += 1

        for lib in rec.get("per_lib", []):
            total_lib += 1
            specified = lib.get("specified_version")
            version_spec = lib.get("version_spec")
            resolved = lib.get("resolved_version")
            status = lib.get("version_status")
            if status:
                version_status_counter[status] += 1

            has_spec = specified is not None or (version_spec is not None and lib.get("version_exists"))
            if has_spec:
                specified_lib += 1
                if lib.get("version_exists"):
                    valid_specified += 1

            lib_key = (str(lib.get("pypi_name") or "").replace("_", "-").lower(), resolved)
            vul = findings_by_key.get(lib_key)
            if bool(vul and vul.get("is_vulnerable")):
                vuln_lib += 1
                sev = vul.get("max_severity")
                if sev:
                    severity_counter[sev] += 1

    version_results = {
        "total_lib": total_lib,
        "specified_lib": specified_lib,
        "lib_valid_specified": valid_specified,
        "lib_version_spec_rate": specified_lib / total_lib if total_lib else 0.0,
        "lib_version_validity_rate": valid_specified / specified_lib if specified_lib else 0.0,
    }

    vulnerability = {
        "lib_vuln_rate": vuln_lib / total_lib if total_lib else 0.0,
        "task_vuln_exposure": task_has_vuln / total_tasks if total_tasks else 0.0,
        "severity_dist": dict(severity_counter),
    }
    compat_ty = {
        "task_compatible": ty_task_compatible,
        "task_incompatible": ty_task_incompatible,
        "task_compat_rate": ty_task_compatible / total_tasks if total_tasks else 0.0,
        "task_rule_dist": dict(ty_rule_counter),
        "ty_error": {
            "task_dist": dict(ty_error_type_counter),
            "install_error_clusters": dict(ty_install_category_counter),
        },
    }
    compat_security = {
        "task_safe_ty_compat": task_safe_ty_compat,
        "task_unsafe_ty_compat": task_unsafe_ty_compat,
        "task_safe_ty_incompat": task_safe_ty_incompat,
        "task_unsafe_ty_incompat": task_unsafe_ty_incompat,
        "task_total": total_tasks,
    }
    compat_bcb = {
        "task_pass": bcb_task_pass,
        "task_fail": bcb_task_fail,
        "task_error_like": bcb_task_error_like,
        "task_total": bcb_task_total,
        "pass_rate": bcb_task_pass / bcb_task_total if bcb_task_total else None,
        "safe_bcb_pass": safe_bcb_pass,
        "safe_bcb_fail": safe_bcb_fail,
        "unsafe_bcb_pass": unsafe_bcb_pass,
        "unsafe_bcb_fail": unsafe_bcb_fail,
        "bcb_error": {
            "from_ty_install_error": bcb_from_ty_install,
            "from_ty_env_error": bcb_from_ty_env,
            "fail_error_type_dist": dict(bcb_fail_error_type_counter),
        },
    }

    return {
        "model_name": model_name,
        "total_tasks": total_tasks,
        "version_results": version_results,
        "vulnerability": vulnerability,
        "version_status_dist": dict(version_status_counter),
        "compat_ty": compat_ty,
        "compat_security": compat_security,
        "compat_bcb": compat_bcb
    }


def save_metrics(summary: dict, *, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
