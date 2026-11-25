"""Project corrected source declarations and attach attributable resolver evidence.

This module never executes generated code, invokes a resolver, or maps an import.
Its inputs are the corrected prepared corpus and verified evidence indexes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion


TARGET = {
    "python_version": "3.12", "sys_platform": "linux",
    "platform_system": "Linux", "os_name": "posix",
    "platform_machine": "x86_64", "platform_python_implementation": "CPython",
    "implementation_name": "cpython", "extra": "",
}
CUTOFF = "2026-02-01T00:00:00Z"
MODEL_KINDS = {"exact", "range"}
UNRESOLVED_IDENTITIES = {"uncertain", "identity_unresolved"}


def digest(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def response_key(row):
    return (row["track"], row["mode"], row["model"].lower(), str(row["task_id"]))


def provenance(row):
    return dict(zip(("track", "mode", "model", "task_id"), response_key(row)),
                response_sha256=row["response_sha256"])


def requirement_for(record):
    """Use manifest distribution identity directly and the prepared import identity."""
    if record["constraint_kind"] == "invalid" or not record["identity_assessable"]:
        return None
    if record["declaration_source"] == "manifest":
        return str(Requirement(record["raw_requirement"]))
    name = record["pypi_name"]
    version = record.get("specified_version")
    specifier = record.get("version_spec")
    return name + ("==" + version if version else str(specifier or ""))


def canonical_requirement(requirement):
    parsed = Requirement(requirement)
    extras = "[" + ",".join(sorted(parsed.extras)) + "]" if parsed.extras else ""
    result = canonicalize_name(parsed.name) + extras
    result += " @ " + parsed.url if parsed.url else str(parsed.specifier)
    if parsed.marker:
        result += "; " + str(parsed.marker)
    return result


def marker_active(marker):
    if not marker:
        return True
    # The frozen target does not specify a patch release or host kernel.
    unsupported = ("python_full_version", "implementation_version", "platform_release", "platform_version")
    if any(variable in marker for variable in unsupported):
        raise ValueError("Marker requires an unspecified frozen-target field: " + marker)
    return Marker(marker).evaluate(environment=TARGET)


def source_record(prepared, original, source_kind, source_ordinal):
    row = dict(original)
    row.update(provenance(prepared))
    row["source_kind"] = source_kind
    row["source_ordinal"] = source_ordinal
    row["declaration_id"] = digest([*response_key(prepared), prepared["response_sha256"], source_kind, source_ordinal])
    row["specified_version"] = original.get("specified_version") or original.get("version")
    distribution = original.get("pypi_name") or original.get("pypi_name_hint")
    if source_kind == "manifest" and original["constraint_kind"] != "invalid":
        distribution = canonicalize_name(Requirement(original["raw_requirement"]).name)
    row["pypi_name"] = canonicalize_name(distribution) if distribution else None
    row["identity_assessable"] = bool(distribution) and original.get("identity_classification") not in UNRESOLVED_IDENTITIES
    row["marker_active"] = marker_active(original.get("marker"))
    row["is_direct"] = True
    row["dependency_scope"] = "direct"
    row["python_version"] = TARGET["python_version"]
    row["constraint_valid"] = False if original["constraint_kind"] == "invalid" else None
    row["constraint_resolved_version"] = None
    row["resolved_version"] = None
    row["resolution_origin"] = "unresolved"
    row["resolution_method"] = None
    row["constraint_id"] = None
    row["frozen_assessment_eligible"] = bool(
        row["identity_assessable"] and row["marker_active"] and row["constraint_kind"] in MODEL_KINDS
    )
    row["all_constraints_assessment_eligible"] = row["frozen_assessment_eligible"] or row["constraint_kind"] == "invalid"
    row["resolver_requirement"] = requirement_for(row)
    if row["resolver_requirement"]:
        try:
            row["resolver_requirement"] = canonical_requirement(row["resolver_requirement"])
        except InvalidRequirement as error:
            # The source reader classifies literal VERSION text as exact advice.
            # Preserve that source category when its content is unparseable.
            row["resolver_parse_error"] = str(error)
    return row


def enumerate_declarations(prepared):
    """One manifest occurrence plus each unmatched import occurrence forms merged input."""
    dependencies = prepared["dependencies"]
    manifests = [source_record(prepared, item, "manifest", index)
                 for index, item in enumerate(dependencies["manifest_records"])]
    imports = [source_record(prepared, item, "import", index)
               for index, item in enumerate(dependencies["import_records"])]
    by_distribution = defaultdict(list)
    for row in manifests:
        if row["constraint_kind"] != "invalid":
            by_distribution[row["pypi_name"]].append(row["declaration_id"])

    direct = list(manifests)
    rq1 = list(manifests) if prepared["mode"] == "requirements.txt" else []
    excluded = []
    for row in imports:
        classification = row.get("identity_classification")
        if classification not in {"external", *UNRESOLVED_IDENTITIES}:
            excluded.append(row)
            continue
        matches = by_distribution.get(row["pypi_name"], []) if row["identity_assessable"] else []
        row["matched_manifest_declaration_ids"] = matches
        row["merged_declaration_ids"] = matches or [row["declaration_id"]]
        if not matches:
            direct.append(row)
        if prepared["mode"] == "inline":
            rq1.append(row)
    for ordinal, row in enumerate(direct):
        row["ordinal"] = ordinal
    for row in rq1:
        row["rq1_carrier"] = "manifest" if prepared["mode"] == "requirements.txt" else "code_import"
        row["rq1_eligible"] = prepared["track"] == "d2"
    return direct, rq1, excluded


def accepts_version(requirement, version):
    try:
        parsed = Requirement(requirement)
        return bool(version) and parsed.specifier.contains(str(version), prereleases=True)
    except (InvalidRequirement, InvalidVersion):
        return False


def apply_witness(row, version, method, evidence):
    row["constraint_valid"] = True
    row["constraint_resolved_version"] = version
    row["resolution_origin"] = "model_exact" if row["constraint_kind"] == "exact" else "model_constrained_resolver"
    row["resolution_method"] = method
    row["validity_status"] = "satisfiable"
    row["validity_evidence"] = evidence


def assess_declaration(row, joint, results):
    """Return a unique independent check request when no attributable witness exists."""
    kind = row["constraint_kind"]
    if kind == "invalid":
        row["validity_status"] = "invalid_parse"
        return None
    if not row["identity_assessable"]:
        row["validity_status"] = "identity_unresolved"
        return None
    if not row["marker_active"]:
        row["validity_status"] = "marker_inactive"
        return None
    if row.get("resolver_parse_error"):
        row["constraint_valid"] = False
        row["validity_status"] = "invalid_requirement_syntax"
        return None
    requirement = row["resolver_requirement"]
    pin = joint.get("pins", {}).get(row["pypi_name"]) if joint.get("joint_status") == "resolved" else None
    if kind == "bare":
        row["validity_status"] = "bare_declaration"
        if pin:
            row["resolved_version"] = pin
            row["resolution_origin"] = "unconstrained_resolver"
            row["resolution_method"] = "joint_lock"
        return None
    parsed = Requirement(requirement)
    # A same-name/version pin does not prove that a direct URL supplied it.
    reference_matches = not parsed.url or joint.get("direct_references", {}).get(row["pypi_name"]) == parsed.url
    if pin and reference_matches and accepts_version(requirement, pin):
        row["resolved_version"] = pin
        apply_witness(row, pin, "joint_lock", {name: joint.get(name) for name in ("lock_path", "lock_sha256", "evidence_source")})
        return None
    request = {
        "requirement": requirement, "package": row["pypi_name"], "kind": kind,
        "target": TARGET, "cutoff": CUTOFF,
    }
    constraint_id = digest(request)
    request["constraint_id"] = constraint_id
    row["constraint_id"] = constraint_id
    result = results.get(constraint_id)
    if result:
        assert result["requirement"] == requirement and result["package"] == row["pypi_name"]
        evidence = {name: result.get(name) for name in (
            "constraint_id", "input_fingerprint", "attempt_directory", "lock_sha256", "status",
            "selected_version", "classification_revision",
        )}
        evidence["result_record_sha256"] = digest(result)
        row["validity_evidence"] = evidence
        row["validity_status"] = result["status"]
        row["constraint_valid"] = result["valid"]
        if result["valid"]:
            version = result["selected_version"]
            assert accepts_version(requirement, version), (requirement, version)
            apply_witness(row, version, "independent_constraint", evidence)
        return None
    row["validity_status"] = "pending_independent_check"
    return request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--joint-index", type=Path)
    parser.add_argument("--constraint-results", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    joint_index = {response_key(row): row for row in read_rows(args.joint_index)} if args.joint_index else {}
    results = {row["constraint_id"]: row for row in read_rows(args.constraint_results)} if args.constraint_results else {}
    requests = {}
    counts = Counter()
    populations = defaultdict(Counter)
    paths = {name: args.out / (name + ".jsonl") for name in (
        "direct_dependencies", "source_carrier_declarations", "excluded_imports", "harness_dependencies",
    )}
    files = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    seen = set()
    try:
        for prepared in read_rows(args.prepared):
            key = response_key(prepared)
            assert key not in seen, key
            seen.add(key)
            counts["responses"] += 1
            group = populations[prepared["track"] + "/" + prepared["mode"]]
            group["responses"] += 1
            joint = joint_index.get(key, {})
            if args.joint_index:
                assert joint, ("missing_joint_index_record", key)
                assert joint["response_sha256"] == prepared["response_sha256"], key
            direct, rq1, excluded = enumerate_declarations(prepared)
            for row in direct:
                request = assess_declaration(row, joint, results)
                if request:
                    queued = requests.setdefault(request["constraint_id"], dict(request, declaration_ids=[]))
                    queued["declaration_ids"].append(row["declaration_id"])
                counts["direct_kind/" + row["constraint_kind"]] += 1
                counts["validity_status/" + row["validity_status"]] += 1
                group["direct_kind/" + row["constraint_kind"]] += 1
                group["validity_status/" + row["validity_status"]] += 1
            harness = [dict(item, **provenance(prepared), dependency_scope="test_harness", is_direct=False)
                       for item in prepared.get("harness", [])]
            outputs = {"direct_dependencies": direct, "source_carrier_declarations": rq1,
                       "excluded_imports": excluded, "harness_dependencies": harness}
            for name, rows in outputs.items():
                counts[name] += len(rows)
                group[name] += len(rows)
                for row in rows:
                    files[name].write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in rq1:
                counts["rq1_kind/" + row["constraint_kind"]] += 1
                group["rq1_kind/" + row["constraint_kind"]] += 1
    finally:
        for output in files.values():
            output.close()
    queue_path = args.out / "constraint_validation_queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as output:
        for constraint_id in sorted(requests):
            output.write(json.dumps(requests[constraint_id], ensure_ascii=False) + "\n")
    counts["unique_independent_checks"] = len(requests)
    summary = {
        "prepared_path": str(args.prepared), "prepared_sha256": file_digest(args.prepared),
        "joint_index_path": str(args.joint_index) if args.joint_index else None,
        "joint_index_sha256": file_digest(args.joint_index) if args.joint_index else None,
        "constraint_results_path": str(args.constraint_results) if args.constraint_results else None,
        "constraint_results_sha256": file_digest(args.constraint_results) if args.constraint_results else None,
        "target": TARGET, "cutoff": CUTOFF, "counts": dict(counts),
        "by_population": {name: dict(values) for name, values in populations.items()},
        "output_sha256": {name: file_digest(path) for name, path in paths.items()},
        "constraint_queue_sha256": file_digest(queue_path),
    }
    (args.out / "declaration_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
