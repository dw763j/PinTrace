"""Prepare source-linked corrected evaluation inputs from saved responses."""

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical_requirements(lines):
    values = set()
    constrained_unconditionally = set()
    bare = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
            requirement.name = canonicalize_name(requirement.name)
            value = str(requirement)
            values.add(value)
            if requirement.marker is None:
                if not requirement.extras and not requirement.specifier and not requirement.url:
                    bare[requirement.name] = value
                else:
                    constrained_unconditionally.add(requirement.name)
        except InvalidRequirement:
            values.add(line)
    for name, value in bare.items():
        if name in constrained_unconditionally:
            values.remove(value)
    return sorted(values)


def key(record):
    return record["track"], record["mode"], record["model"].lower(), str(record["task_id"])


def load_overrides(path):
    if path is None:
        return {}
    records = json.loads(path.read_text())
    result = {key(record): record for record in records}
    assert len(result) == len(records), "Duplicate source override"
    return result


def apply_source_overrides(row, extraction, source_overrides):
    response_key = key(row)
    source = source_overrides.get(response_key)
    if source:
        assert source["response_sha256"] == row["response_sha256"], "Source override hash mismatch"
        extraction = dict(source["result_override"])
    return extraction


def apply_identity_decisions(row, dependencies, decisions):
    """Apply source-specific author decisions without changing model text."""
    for decision in decisions.get(key(row), []):
        match = decision["source_match"]
        assert match["response_sha256"] == row["response_sha256"]
        matches = [
            item for item in dependencies["identities"]
            if dependencies["import_records"][item["import_ordinal"]]["import_line"].strip()
            == match["import_line"].strip()
        ]
        assert len(matches) == 1, "Author identity decision must match one import"
        identity = matches[0]
        record = dependencies["import_records"][identity["import_ordinal"]]
        action = decision["resolver_action"]
        provider = action["provider_distribution"]
        identity.update(
            needs_human_review=False, author_review_complete=True,
            author_case_id=decision["case_id"],
            distribution=provider,
            classification="external" if provider else "uncertain",
            rule="author_verified_namespace_provider" if provider else "author_unresolved_identity",
        )
        record.update(
            identity_classification=identity["classification"], identity_rule=identity["rule"],
            pypi_name_hint=provider, pypi_name=provider,
            model_distribution=decision["model_advice"]["model_distribution"],
            distribution_identity=decision["model_advice"]["distribution_identity"],
            provider_origin=action["provider_origin"], author_case_id=decision["case_id"],
        )
        if provider:
            record["resolver_requirement"] = action["requirement"]
            dependencies["resolver_records"].append(dict(record))
            dependencies["requirements"].append(action["requirement"])
    dependencies["requirements"] = list(dict.fromkeys(dependencies["requirements"]))
    dependencies["non_external_imports"] = [
        record for record in dependencies["import_records"]
        if record["identity_classification"] != "external"
    ]
    dependencies["review_required"] = any(item["needs_human_review"] for item in dependencies["identities"])
    dependencies["complete_dependency_input"] = not any(
        item["classification"] == "uncertain" for item in dependencies["identities"]
    )
    return dependencies


def requirement_names(lines):
    names = set()
    for line in lines:
        try:
            names.add(canonicalize_name(Requirement(line).name))
        except InvalidRequirement:
            match = re.match(r"[A-Za-z0-9_.-]+", line)
            if match:
                names.add(canonicalize_name(match.group()))
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PINTRACE_ENGINE_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--manual-cases", type=Path, required=True)
    parser.add_argument("--source-overrides", type=Path)
    parser.add_argument("--identity-decisions", type=Path)
    parser.add_argument("--structural-outcomes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root))
    os.environ["EXPERIMENT_NOTIFY_ENABLED"] = "0"
    from stages.version_resolver import _extract_tpl_versions_from_code, _parse_requirements_txt, build_task_requirements
    from manual_response_extraction import extract_response
    from manual_dependency_identity import prepare_dependency_records

    mapping = json.loads(args.mapping.read_text())["mapping"]
    mapping["ckeditor_uploader"] = "django-ckeditor"
    manual = {
        key(record): record["case_id"]
        for record in json.loads(args.manual_cases.read_text())
        if record["kind"] == "dependency"
    }
    problems = {
        str(record["task_id"]): record
        for line in (args.root / "resources/BigCodeBench-v0.1.4-local.jsonl").open()
        if line.strip() for record in [json.loads(line)]
    }
    counts = collections.Counter()
    pending = []
    source_overrides = load_overrides(args.source_overrides)
    structural_outcomes = load_overrides(args.structural_outcomes)
    identity_decisions = collections.defaultdict(list)
    if args.identity_decisions:
        for decision in json.loads(args.identity_decisions.read_text())["decisions"]:
            identity_decisions[key(decision["source_match"])].append(decision)
    with args.corpus.open() as source, (args.out / "prepared.jsonl").open("w") as output:
        for line in source:
            row = json.loads(line)
            extraction = extract_response(row["raw_response"], legacy_code=row["selected_code"])
            extraction = apply_source_overrides(row, extraction, source_overrides)
            structural = structural_outcomes.get(key(row))
            if structural:
                assert structural["response_sha256"] == row["response_sha256"]
                extraction.update(
                    code="", files={}, primary_filename=None, manifest_text="",
                    manifest_sources=[], review_required=False,
                    reasons=[structural["reason_code"]],
                )
            imports = []
            for filename, code in extraction["files"].items():
                imports.extend(dict(item, source_file=filename) for item in _extract_tpl_versions_from_code(code, "3.12"))
            manifests = _parse_requirements_txt(extraction["manifest_text"])
            dependencies = prepare_dependency_records(imports, manifests, code_files=extraction["files"], mapping=mapping)
            dependencies = apply_identity_decisions(row, dependencies, identity_decisions)
            requirements = list(dependencies["requirements"])
            harness_records = []
            harness_review = False
            if row["track"] == "d1" and extraction["code"].strip():
                test = problems[row["task_id"]]["test"]
                harness_imports = _extract_tpl_versions_from_code(test, "3.12")
                harness = prepare_dependency_records(harness_imports, requirements, code_files={"test.py": test, **extraction["files"]}, mapping=mapping)
                harness_review = harness["review_required"]
                for item in harness["identities"]:
                    if item["classification"] == "external" and item["distribution"] not in requirement_names(requirements):
                        requirements.append(item["distribution"])
                        harness_records.append(dict(item, declaration_source="test_harness"))
            old_requirements = (row.get("requirements_in") or "").splitlines()
            if row.get("requirements_in") is None:
                old_requirements = build_task_requirements(row["stored_extraction"], mapping=mapping)
            code_changed = extraction["code"] != row["selected_code"] or len(extraction["files"]) > 1
            requirements_changed = canonical_requirements(requirements) != canonical_requirements(old_requirements)
            old_manifests = {
                item.get("raw_requirement") for item in row["stored_extraction"]
                if item.get("declaration_source") == "manifest"
            }
            new_manifests = {item.get("raw_requirement") for item in manifests}
            review_required = extraction["review_required"] or dependencies["review_required"] or harness_review
            changed = code_changed or requirements_changed
            complete_input = dependencies["complete_dependency_input"] and not structural
            response_status = "unusable_response" if structural else "identity_incomplete" if not complete_input else "usable"
            response_key = key(row)
            result = {
                "request_id": sha(json.dumps(response_key))[:24],
                **{name: row[name] for name in ["track", "mode", "model", "task_id", "response_sha256"]},
                "manual_case_id": manual.get(response_key),
                "code": extraction["code"], "files": extraction["files"],
                "primary_filename": extraction["primary_filename"],
                "requirements": requirements, "old_requirements": old_requirements,
                "environment_eligible": bool(requirements and extraction["code"].strip()),
                "complete_dependency_input": bool(complete_input),
                "evaluation_eligible": bool(complete_input and not review_required),
                "response_status": response_status,
                "routine_dependency_analysis_applicable": not bool(structural),
                "author_structural_case_id": structural["case_id"] if structural else None,
                "extraction": {name: value for name, value in extraction.items() if name not in ["code", "files", "blocks"]},
                "dependencies": dependencies, "harness": harness_records,
                "code_changed": code_changed, "requirements_changed": requirements_changed,
                "manifest_changed": old_manifests != new_manifests,
                "evaluation_changed": changed, "review_required": review_required,
                "old_outcome": row["old_outcome"],
                "measurement_path": row["measurement_path"],
                "measurement_index": row["measurement_index"],
                "source_selection_override": key(row) in source_overrides,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts["responses"] += 1
            for name in ["code_changed", "requirements_changed", "manifest_changed", "evaluation_changed", "review_required"]:
                counts[name] += bool(result[name])
            counts["response_status_" + response_status] += 1
            if changed and result["evaluation_eligible"]:
                counts["automatic_evaluation_candidates"] += 1
            if review_required:
                pending.append({name: result[name] for name in ["request_id", "track", "mode", "model", "task_id", "manual_case_id", "code_changed", "requirements_changed"]})
    assert counts["responses"] == 36260
    (args.out / "summary.json").write_text(json.dumps(dict(counts), indent=2))
    (args.out / "review_pending.json").write_text(json.dumps(pending, indent=2))
    provenance = {"arguments": {name: str(value) for name, value in vars(args).items()}}
    provenance["input_sha256"] = {
        name: file_sha(path)
        for name, path in [
            ("corpus", args.corpus), ("mapping", args.mapping),
            ("source_overrides", args.source_overrides),
            ("identity_decisions", args.identity_decisions),
            ("structural_outcomes", args.structural_outcomes),
        ] if path is not None
    }
    provenance["script_sha256"] = {
        name: file_sha(Path(__file__).parent / name)
        for name in ["prepare_manual_corrections_remote.py", "manual_response_extraction.py", "manual_dependency_identity.py"]
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
