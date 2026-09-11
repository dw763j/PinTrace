"""Build origin-aware canonical JSONL tables from revision pipeline outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from paths import EVALUATION
from stages.failure_classifier import classify_compatibility_layer, classify_ty_error


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_records(run_dir: Path) -> tuple[Path, list[dict]]:
    for name in ("m5_compat_records.json", "m4_vuln_records.json", "m3_resolved_records.json"):
        path = run_dir / name
        if path.is_file():
            data = _load_json(path)
            if isinstance(data, list):
                return path, data
    raise FileNotFoundError(f"No M3-M5 records in {run_dir}")


def _vulnerability_join_key(dependency: dict) -> tuple[str, object, object, object]:
    """Canonical key shared by direct dependencies and vulnerability findings."""
    version = dependency.get("version")
    if version is None:
        version = dependency.get("resolved_version") or dependency.get("constraint_resolved_version")
    scope = dependency.get("dependency_scope") or "direct"
    return (
        str(dependency.get("pypi_name") or "").lower().replace("_", "-"),
        version,
        dependency.get("resolution_origin"),
        scope,
    )


def _has_valid_constraint(dependencies: list[dict]) -> bool:
    return any(
        dependency.get("constraint_valid")
        if dependency.get("constraint_valid") is not None
        else dependency.get("version_exists")
        for dependency in dependencies
    )


def _has_model_cve(findings: list[dict], kind: str) -> bool:
    expected_origin = "model_exact" if kind == "exact" else "model_constrained_resolver"
    return any(
        finding.get("is_vulnerable")
        and finding.get("constraint_kind") == kind
        and finding.get("resolution_origin") == expected_origin
        and finding.get("dependency_scope") == "direct"
        for finding in findings
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--tracks", nargs="+", default=["d1", "d2"])
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--expected-models", type=int, default=None)
    parser.add_argument("--expected-pintrace-tasks", type=int, default=None)
    parser.add_argument("--expected-bcb-tasks", type=int, default=None)
    parser.add_argument(
        "--record-overlays",
        type=Path,
        nargs="*",
        default=[],
        help="JSONL corrections keyed by track/mode/model/task_id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVALUATION,
        help="M6 canonical tables directory (outputs/evaluation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlays: dict[tuple[str, str, str, str], dict] = {}
    for overlay_path in args.record_overlays:
        with overlay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = tuple(str(item[name]) for name in ("track", "mode", "model", "task_id"))
                if key in overlays:
                    raise ValueError(f"Duplicate correction overlay key: {key}")
                overlays[key] = item["record"]
    used_overlays: set[tuple[str, str, str, str]] = set()
    py_tag = f"py{args.python_version.replace('.', '')}"
    manifests = sorted(args.outputs_root.glob(f"**/{py_tag}/paper/run_manifest.json"))
    dependency_rows: list[dict] = []
    task_rows: list[dict] = []
    audit_rows: list[dict] = []
    group_models: dict[tuple[str, str], set[str]] = {}
    group_task_ids: dict[tuple[str, str], dict[str, frozenset[str]]] = {}

    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        track = str(manifest.get("track") or "")
        all_args = manifest.get("all_the_args") or {}
        mode = str(all_args.get("prompt_mode") if track == "d1" else all_args.get("pinning_mode"))
        if mode not in args.modes:
            continue
        run_dir = manifest_path.parents[1]
        source_path, records = _latest_records(run_dir)
        metrics_path = run_dir / "metrics_summary.json"
        metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
        model = str((metrics or {}).get("model_name") or manifest.get("run_name") or run_dir.parent.name)
        group = (track, mode)
        if model in group_models.setdefault(group, set()):
            raise ValueError(f"Duplicate canonical run for {group} model={model}: {run_dir}")
        group_models[group].add(model)
        expected_tasks = args.expected_bcb_tasks if track == "d1" else args.expected_pintrace_tasks
        task_ids = [str(record.get("task_id")) for record in records]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"Duplicate task ids in {source_path}")
        if expected_tasks is not None and len(task_ids) != expected_tasks:
            raise ValueError(f"{source_path}: expected {expected_tasks} tasks, found {len(task_ids)}")
        group_task_ids.setdefault(group, {})[model] = frozenset(task_ids)

        for record in records:
            task_id = str(record.get("task_id"))
            overlay_key = (track, mode, model, task_id)
            if overlay_key in overlays:
                record = overlays[overlay_key]
                used_overlays.add(overlay_key)
            findings = record.get("vuln_findings") or []
            vulnerable_findings = {
                _vulnerability_join_key(f): f
                for f in findings
                if f.get("is_vulnerable")
            }
            direct = record.get("per_lib") or []
            for ordinal, dep in enumerate(direct):
                key = _vulnerability_join_key(dep)
                finding = vulnerable_findings.get(key) or {}
                dependency_rows.append(
                    {
                        "track": track,
                        "mode": mode,
                        "model": model,
                        "python_version": args.python_version,
                        "task_id": task_id,
                        "ordinal": ordinal,
                        **dep,
                        "is_cve_associated": bool(finding),
                        "cve_ids": finding.get("cve_ids") or [],
                        "osv_ids": finding.get("osv_ids") or [],
                        "max_severity": finding.get("max_severity"),
                        "cvss_score": finding.get("cvss_score"),
                    }
                )
            model_constraints = [
                dep for dep in direct
                if dep.get("declaration_source") in {"manifest", "inline_annotation"}
                and dep.get("constraint_kind") in {"exact", "range"}
            ]
            constraints_by_kind = {
                kind: [dep for dep in model_constraints if dep.get("constraint_kind") == kind]
                for kind in ("exact", "range")
            }

            compat = record.get("compat_results") or {}
            ty_result = compat.get("ty") or {}
            bcb_result = compat.get("bcb_test") or None
            _, ty_install_category = classify_ty_error(ty_result)
            task_rows.append(
                {
                    "track": track,
                    "mode": mode,
                    "model": model,
                    "python_version": args.python_version,
                    "task_id": task_id,
                    "has_model_constraint": bool(model_constraints),
                    "has_valid_model_constraint": any(
                        dep.get("constraint_valid")
                        if dep.get("constraint_valid") is not None
                        else dep.get("version_exists")
                        for dep in model_constraints
                    ),
                    "has_model_cve": any(
                        f.get("is_vulnerable")
                        and f.get("resolution_origin") in {"model_exact", "model_constrained_resolver"}
                        and f.get("dependency_scope") == "direct"
                        for f in findings
                    ),
                    "has_model_exact_constraint": bool(constraints_by_kind["exact"]),
                    "has_valid_model_exact_constraint": _has_valid_constraint(constraints_by_kind["exact"]),
                    "has_model_exact_cve": _has_model_cve(findings, "exact"),
                    "has_model_range_constraint": bool(constraints_by_kind["range"]),
                    "has_valid_model_range_constraint": _has_valid_constraint(constraints_by_kind["range"]),
                    "has_model_range_cve": _has_model_cve(findings, "range"),
                    "security_status": "available" if "vuln_findings" in record else "unavailable",
                    "resolver_status": (record.get("resolver") or {}).get("status"),
                    "ty_compatible": ty_result.get("is_compatible"),
                    "bcb_status": (bcb_result or {}).get("status"),
                    "compatibility_layer": classify_compatibility_layer(ty_result, bcb_result),
                    "ty_install_category": ty_install_category,
                }
            )
        audit_path = run_dir / "paper" / "run_audit.json"
        audit_rows.append(_load_json(audit_path) if audit_path.is_file() else {
            "track": track, "mode": mode, "model": model, "run_dir": str(run_dir), "audit_missing": True
        })

    required_groups = {(track, mode) for track in args.tracks for mode in args.modes}
    missing_groups = sorted(required_groups - set(group_models))
    if missing_groups:
        raise ValueError(f"Missing canonical track/mode groups: {missing_groups}")
    if args.expected_models is not None:
        bad = {key: len(group_models[key]) for key in required_groups if len(group_models[key]) != args.expected_models}
        if bad:
            raise ValueError(f"Unexpected model counts: {bad}")
    for group in sorted(required_groups):
        populations = group_task_ids[group]
        baseline_model, baseline_ids = next(iter(populations.items()))
        mismatched = [model for model, ids in populations.items() if ids != baseline_ids]
        if mismatched:
            raise ValueError(
                f"Task-id population mismatch in {group}; baseline={baseline_model}, mismatched={mismatched[:3]}"
            )
    for track in args.tracks:
        track_populations = [
            next(iter(group_task_ids[(track, mode)].values()))
            for mode in args.modes
        ]
        if any(ids != track_populations[0] for ids in track_populations[1:]):
            raise ValueError(f"Task-id populations differ across modes for track={track}")
    unused_overlays = sorted(set(overlays) - used_overlays)
    if unused_overlays:
        raise ValueError(f"Correction overlays did not match canonical records: {unused_overlays[:3]}")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {
        "direct_dependencies": _write_jsonl(args.output / "direct_dependencies.jsonl", dependency_rows),
        "task_outcomes": _write_jsonl(args.output / "task_outcomes.jsonl", task_rows),
        "run_audits": _write_jsonl(args.output / "run_audits.jsonl", audit_rows),
    }
    (args.output / "canonical_config.json").write_text(
        json.dumps({"python_version": args.python_version, "tracks": args.tracks, "modes": args.modes, "groups": {f"{k[0]}:{k[1]}": sorted(v) for k, v in group_models.items()}, "record_overlays": [str(path.resolve()) for path in args.record_overlays], "overlay_count": len(used_overlays), "counts": counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
