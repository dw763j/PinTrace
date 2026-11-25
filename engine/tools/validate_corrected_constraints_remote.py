"""Independently resolve corrected literal constraints under the study protocol.

Each input JSONL record supplies constraint_id, requirement, package, and kind.
The script preserves distribution names and records tri-state validity.  uv may
prepare package metadata, while generated task code is never executed.
"""

import argparse
import concurrent.futures
import contextlib
import datetime
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion

from evaluation_support_remote import run


REVISION = "corrected-constraint-validator-v1"
CLASSIFICATION_REVISION = "constraint-log-classification-v4"
TERMINAL = {"resolved", "resolved_direct_url", "unsatisfiable", "invalid_parse", "marker_inactive", "metadata_build_failure", "resolver_policy_failure"}
NETWORK_PATTERNS = (
    r"failed to (?:fetch|download|connect)", r"error (?:sending|requesting)",
    r"connection (?:refused|reset|closed|error)", r"network (?:is unreachable|connectivity is disabled)",
    r"(?:dns|certificate|tls|ssl) (?:error|failure|failed|verify)", r"timed? out",
    r"\bHTTP(?:/\d(?:\.\d)?)?\s+(?:(?:status|error|response)(?:\s+code)?\s*[:=]?\s*)?(?:401|403|408|429|5\d\d)\b",
    r"\bstatus code\s*[:=]?\s*(?:401|403|408|429|5\d\d)\b",
    r"not found in (?:the )?cache", r"offline mode", r"temporary failure in name resolution",
)
METADATA_PATTERNS = (
    r"failed to (?:build|prepare|extract|read|parse)", r"build backend", r"build_backend",
    r"invalid metadata", r"metadata (?:is invalid|generation failed)",
    r"failed to resolve (?:requirements|dependencies) from (?:the )?build", r"subprocess-exited-with-error",
    r"error building wheel", r"invalid wheel", r"hash mismatch", r"failed to verify",
)
FROZEN_BUILD_FAILURES = (
    ("python2_only_build_source", r"Missing parentheses in call to 'print'|multiple exception types must be parenthesized|NameError: name 'file' is not defined"),
    ("removed_runtime_or_build_api", r"No module named '(?:imp|ConfigParser)'|has no attribute 'SafeConfigParser'|use_2to3 is invalid|build_py_2to3 not found"),
    ("required_native_configuration_unavailable", r"'apxs' command appears not to be installed|Could not run curl-config|Could not find gdal-config"),
    ("required_file_absent_from_source_distribution", r"FileNotFoundError:.*(?:requirements_extra_data_sources\.txt|README\.rst)"),
    ("distribution_metadata_rejected", r"has invalid metadata"),
    ("package_metadata_symlink_operation_failure", r"create_library_symlinks.*FileExistsError:.*libarrow"),
    ("undeclared_metadata_build_dependency", r"ModuleNotFoundError: No module named 'torch'.*doesn't declare it as a build dependency"),
)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                stream.write(b"0")
                stream.flush()
                stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ResolverConfig:
    uv: str = "/tmp/cyse-20261109-tools/bin/uv"
    cache_dir: str = "/data/mine/llm_tpl/.cache/uv"
    timeout: int = 600

    def protocol(self):
        return {
            "validator_revision": REVISION, "python_version": "3.12",
            "python_platform": "x86_64-unknown-linux-gnu", "uv_version": "0.10.7",
            "cutoff": "2026-02-01T00:00:00Z", "prerelease": "if-necessary-or-explicit",
            "generate_hashes": True, "index_url": "https://pypi.org/simple",
        }

    def environment(self):
        allowed = {"PATH", "HOME", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"}
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.update(UV_CACHE_DIR=self.cache_dir, UV_NO_PROGRESS="1", UV_CONCURRENT_BUILDS="1", UV_CONCURRENT_DOWNLOADS="8")
        return environment


def normalized_item(item):
    required = ("constraint_id", "requirement", "package", "kind")
    if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
        raise ValueError("Each constraint needs nonempty string constraint_id, requirement, package, and kind")
    return {key: item[key] for key in required}


def target_marker_environment():
    environment = default_environment()
    environment.update(
        python_version="3.12", python_full_version="3.12.0", implementation_version="3.12.0",
        implementation_name="cpython", platform_python_implementation="CPython",
        os_name="posix", sys_platform="linux", platform_system="Linux", platform_machine="x86_64", extra="",
    )
    return environment


@lru_cache(maxsize=None)
def verify_uv_version(executable):
    version = subprocess.check_output([executable, "--version"], text=True, timeout=30).strip()
    if not re.fullmatch(r"uv 0\.10\.7(?:\s+.*)?", version):
        raise RuntimeError(f"Frozen protocol requires uv 0.10.7, observed {version}")
    return version


def parse_lock(lock_text):
    pins = {}
    direct_urls = {}
    for line in lock_text.splitlines():
        if not line or line[0].isspace() or line.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(line.rstrip(" \\"))
        except InvalidRequirement:
            continue
        package = canonicalize_name(requirement.name)
        exact = [item.version for item in requirement.specifier if item.operator in {"==", "==="}]
        if len(exact) == 1:
            if package in pins and pins[package] != exact[0]:
                raise ValueError(f"Multiple locked versions for {package}")
            pins[package] = exact[0]
        if requirement.url:
            direct_urls[package] = requirement.url
    return pins, direct_urls


def classify_resolution(process, *, requirement, package, lock_text, marker_active):
    """Classify a completed compile with infrastructure failures kept tri-state."""
    result = {"status": "resolver_error", "valid": None, "selected_version": None, "pins": {}, "locked_direct_urls": {}, "classification_rationale": None}
    if process.get("timeout") or process.get("returncode") == 124:
        return {**result, "status": "timeout"}
    if process.get("returncode") == 0:
        if lock_text is None:
            return {**result, "status": "missing_lock"}
        try:
            pins, direct_urls = parse_lock(lock_text)
        except ValueError as error:
            return {**result, "status": "lock_inconsistent", "classification_detail": str(error)}
        result.update(pins=pins, locked_direct_urls=direct_urls, selected_version=pins.get(package))
        if marker_active is False and package not in pins and package not in direct_urls:
            return {**result, "status": "marker_inactive"}
        if marker_active is False:
            return {**result, "status": "marker_mismatch"}
        if requirement.url and package in direct_urls:
            return {**result, "status": "resolved_direct_url", "valid": True}
        version = pins.get(package)
        try:
            if version and requirement.specifier.contains(version, prereleases=True):
                return {**result, "status": "resolved", "valid": True}
        except InvalidVersion as error:
            return {**result, "status": "lock_inconsistent", "classification_detail": str(error)}
        return {**result, "status": "missing_or_mismatched_root_pin"}
    log = "\n".join([process.get("stdout", ""), process.get("stderr", "")])
    if "Couldn't parse requirement" in log and "Expected direct URL" in log and "supported file extension" in log:
        return {**result, "status": "invalid_parse", "valid": False, "classification_rationale": "Frozen resolver rejects an HTML index as a direct distribution artifact"}
    if any(re.search(pattern, log, re.I) for pattern in NETWORK_PATTERNS):
        return {**result, "status": "network_error"}
    compact_log = " ".join(log.split())
    if "URL dependencies must be expressed as direct requirements" in compact_log:
        return {**result, "status": "resolver_policy_failure", "valid": False, "classification_rationale": "Frozen resolver rejects a transitive direct-reference dependency under the original input"}
    for reason, pattern in FROZEN_BUILD_FAILURES:
        if re.search(pattern, compact_log, re.I):
            return {**result, "status": "metadata_build_failure", "valid": False, "classification_rationale": reason}
    if any(re.search(pattern, log, re.I) for pattern in METADATA_PATTERNS):
        return {**result, "status": "metadata_error"}
    if re.search(r"no solution found", log, re.I) and re.search(r"unsatisfiable", log, re.I):
        return {**result, "status": "unsatisfiable", "valid": False}
    return result


def _perform_attempt(item, config, attempt, input_fingerprint):
    requirement_path = attempt / "requirements.in"
    lock_path = attempt / "resolved.txt"
    requirement_path.write_text(item["requirement"] + "\n", encoding="utf-8")
    protocol = config.protocol()
    command = [
        config.uv, "pip", "compile", str(requirement_path), "--python-version", protocol["python_version"],
        "--python-platform", protocol["python_platform"], "--exclude-newer", protocol["cutoff"],
        "--prerelease", protocol["prerelease"], "--index-url", protocol["index_url"],
        "--generate-hashes", "--output-file", str(lock_path),
    ]
    result = {
        **item, "input_fingerprint": input_fingerprint, "protocol": protocol,
        "started_at_utc": utc(), "attempt_directory": str(attempt), "command": command,
        "resolver_executed": False, "status": "invalid_parse", "valid": False,
        "selected_version": None, "pins": {}, "locked_direct_urls": {}, "lock_text": None, "lock_sha256": None,
        "marker_active": None, "marker_environment": target_marker_environment(),
        "returncode": None, "stdout": "", "stderr": "", "timeout_seconds": config.timeout,
        "uv_version_output": verify_uv_version(config.uv),
        "classification_revision": CLASSIFICATION_REVISION,
    }
    atomic_json(attempt / "input.json", {**item, "input_fingerprint": input_fingerprint, "protocol": protocol})
    atomic_json(attempt / "command.json", command)
    try:
        if "\n" in item["requirement"] or "\r" in item["requirement"]:
            raise InvalidRequirement("A constraint must contain exactly one requirement line")
        requirement = Requirement(item["requirement"])
    except InvalidRequirement as error:
        result["parse_error"] = str(error)
    else:
        package = canonicalize_name(item["package"])
        if canonicalize_name(requirement.name) != package:
            result.update(status="input_package_mismatch", valid=None, parsed_package=requirement.name)
        else:
            try:
                result["marker_active"] = requirement.marker.evaluate(result["marker_environment"]) if requirement.marker else True
            except (ValueError, KeyError) as error:
                result["marker_evaluation_error"] = str(error)
            try:
                process = run(command, env=config.environment(), cwd=str(attempt), timeout=config.timeout)
            except (OSError, UnicodeError) as error:
                process = {"command": command, "returncode": None, "stdout": "", "stderr": str(error), "exception_type": type(error).__name__}
            result.update(process)
            result["resolver_executed"] = process.get("returncode") is not None
            lock_text = lock_path.read_text(encoding="utf-8") if lock_path.exists() else None
            result.update(classify_resolution(process, requirement=requirement, package=package, lock_text=lock_text, marker_active=result["marker_active"]))
            result["lock_text"] = lock_text
            result["lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_text is not None else None
    result["completed_at_utc"] = utc()
    (attempt / "stdout.txt").write_text(result["stdout"], encoding="utf-8")
    (attempt / "stderr.txt").write_text(result["stderr"], encoding="utf-8")
    atomic_json(attempt / "result.json", result)
    return result


def validate_constraint(item, *, out, config, retry_indeterminate=False):
    """Return one resumable, source-preserving independent constraint result."""
    item = normalized_item(item)
    case = Path(out).resolve() / "cases" / hashlib.sha256(item["constraint_id"].encode()).hexdigest()
    case.mkdir(parents=True, exist_ok=True)
    expected = fingerprint({"input": item, "protocol": config.protocol()})
    with exclusive_lock(case / "case.lock"):
        result_path = case / "result.json"
        existing_attempts = sorted(
            (path for path in (case / "attempts").glob("*") if path.is_dir() and path.name.isdecimal()),
            key=lambda path: int(path.name),
        )
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous["input_fingerprint"] != expected:
                raise ValueError(f"Constraint fingerprint changed for {item['constraint_id']}")
        latest_completed = None
        for attempt in existing_attempts:
            for path in (attempt / "input.json", attempt / "result.json"):
                if not path.exists():
                    continue
                prior = json.loads(path.read_text(encoding="utf-8"))
                if prior["input_fingerprint"] != expected:
                    raise ValueError(f"Constraint fingerprint changed for {item['constraint_id']}")
                if path.name == "result.json":
                    latest_completed = prior
        if latest_completed is not None:
            atomic_json(result_path, latest_completed)
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("resolver_executed") and previous.get("classification_revision") != CLASSIFICATION_REVISION:
                previous.update(classify_resolution(
                    previous, requirement=Requirement(previous["requirement"]),
                    package=canonicalize_name(previous["package"]), lock_text=previous.get("lock_text"),
                    marker_active=previous.get("marker_active"),
                ))
                previous["classification_revision"] = CLASSIFICATION_REVISION
                atomic_json(result_path, previous)
            if previous["status"] in TERMINAL or not retry_indeterminate:
                return previous
        attempt_number = max([int(path.name) for path in existing_attempts], default=0) + 1
        attempt = case / "attempts" / f"{attempt_number:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        result = _perform_attempt(item, config, attempt, expected)
        atomic_json(result_path, result)
        return result


def load_inputs(path):
    unique = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = normalized_item(json.loads(line))
        key = item["constraint_id"]
        if key in unique and unique[key] != item:
            raise ValueError(f"Conflicting duplicate constraint_id on line {number}")
        unique[key] = item
    return [unique[key] for key in sorted(unique)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-indeterminate", action="store_true")
    parser.add_argument("--uv", default=ResolverConfig.uv)
    parser.add_argument("--cache-dir", default=ResolverConfig.cache_dir)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    config = ResolverConfig(args.uv, args.cache_dir, args.timeout)
    version = verify_uv_version(config.uv)
    inputs = load_inputs(args.input)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.out / "batch.lock"):
        manifest = {
            "input_fingerprint": fingerprint({"inputs": inputs, "protocol": config.protocol()}),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "unique_constraints": len(inputs), "protocol": config.protocol(), "uv_version_output": version,
        }
        manifest_path = args.out / "manifest.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior["input_fingerprint"] != manifest["input_fingerprint"]:
                raise ValueError("Batch input or frozen protocol changed; use a separate output directory")
        else:
            atomic_json(manifest_path, manifest)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(validate_constraint, item, out=args.out, config=config, retry_indeterminate=args.retry_indeterminate) for item in inputs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"completed": len(results), "planned": len(inputs), "constraint_id": result["constraint_id"], "status": result["status"], "valid": result["valid"]}), flush=True)
        results.sort(key=lambda item: item["constraint_id"])
        temporary = args.out / "results.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            for result in results:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
        temporary.replace(args.out / "results.jsonl")
        summary = {
            "constraints": len(results), "status_counts": dict(Counter(item["status"] for item in results)),
            "valid": sum(item["valid"] is True for item in results), "invalid": sum(item["valid"] is False for item in results),
            "indeterminate_or_inactive": sum(item["valid"] is None for item in results),
        }
        atomic_json(args.out / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
