"""Compatibility checker using ty type checker and optional BigCodeBench test execution.

Usage (import-only)::

    from stages.compact_checker import check_compatibility_for_pkg
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from stages.utils import get_pkg_name
from stages.uv_runtime import enforce_uv_cache_limit


@dataclass
class CompatError:
    call_site: object | None
    level: str
    message: str
    rule: str | None = None


@dataclass
class CompatResult:
    is_compatible: bool
    level_reached: str
    errors: list[CompatError]


_BCB_RUNNER_SCRIPT = (
    "import builtins, json, os, sys, time, traceback, types, unittest\n"
    "from pathlib import Path\n"
    "payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
    "module_name='__test__'\n"
    "mod=types.ModuleType(module_name)\n"
    "mod.__dict__.update({'__builtins__': builtins, '__file__': module_name + '.py', '__package__': None, '__doc__': None, 'sys': sys, 'os': os, 'environ': os.environ})\n"
    "result={'status':'fail','details':{},'error':None}\n"
    "try:\n"
    "    full_code = payload['solution'] + '\\n' + payload['test']\n"
    "    exec(compile(full_code, module_name + '.py', 'exec'), mod.__dict__)\n"
    "    tc = getattr(mod, 'TestCases')\n"
    "    suite = unittest.TestLoader().loadTestsFromTestCase(tc)\n"
    "    tr = unittest.TestResult()\n"
    "    suite.run(tr)\n"
    "    issues = tr.failures + tr.errors\n"
    "    if issues:\n"
    "        for test, trace in issues:\n"
    "            result['details'][test.id().split('.')[-1]] = trace\n"
    "        result['status'] = 'fail'\n"
    "    else:\n"
    "        result['status'] = 'pass'\n"
    "except BaseException:\n"
    "    result['status'] = 'error'\n"
    "    result['error'] = traceback.format_exc()[-3000:]\n"
    "print(json.dumps(result, ensure_ascii=False))\n"
)


def _normalize_pkg_name(name: str) -> str:
    """Normalize package/module name for set comparison."""
    return name.strip().lower().replace("_", "-")


def _extract_pkg_name_from_spec(spec: str) -> str:
    """Extract package name from a pip spec like 'a==1.0', 'a>=1', 'a[b]'."""
    s = spec.strip()
    if not s:
        return ""
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "==="):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    if "[" in s:
        s = s.split("[", 1)[0]
    return _normalize_pkg_name(s)


def _extract_imported_top_modules(code: str) -> set[str]:
    """Extract top-level imported modules from Python source code."""
    modules: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root:
                    modules.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                root = node.module.split(".", 1)[0]
                if root:
                    modules.add(root)
    return modules


def _is_stdlib_module(module_name: str) -> bool:
    """Check if a top-level module belongs to Python stdlib."""
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        return False
    return module_name in stdlib


def _infer_test_utilized_tpl(test_code: str, pkg_specs: list[str]) -> list[str]:
    """
    Infer third-party modules imported by test code but not covered by pkg_specs.

    Returns package-like names to install (best effort using import root names).
    """
    imported_modules = _extract_imported_top_modules(test_code or "")
    specified = {
        _extract_pkg_name_from_spec(spec)
        for spec in pkg_specs
        if _extract_pkg_name_from_spec(spec)
    }
    extra: list[str] = []
    for mod in sorted(imported_modules):
        normalized = _normalize_pkg_name(mod)
        if not normalized:
            continue
        if _is_stdlib_module(mod):
            continue
        if normalized in specified:
            continue
        extra.append(mod)
    return extra


def _run_bcb_test_in_venv(
    venv_python: str,
    code: str,
    problem: dict[str, Any],
    tmpdir: str,
    timeout_seconds: int,
    project_root: str | None,
) -> dict[str, Any]:
    """Run BigCodeBench unittest in an existing venv. Returns {status, details, error}."""
    payload = {
        "solution": code,
        "test": problem.get("test", ""),
        "entry_point": problem.get("entry_point", ""),
        "timeout_seconds": timeout_seconds,
    }
    runner_path = Path(tmpdir) / "runner.py"
    runner_path.write_text(_BCB_RUNNER_SCRIPT, encoding="utf-8")
    payload_path = Path(tmpdir) / "payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    if project_root:
        env["PYTHONPATH"] = project_root
    # Match BigCodeBench: run in isolated temp dir (create_tempdir/chdir) so cwd is clean
    proc = subprocess.run(
        [venv_python, str(runner_path), str(payload_path)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds + 20,
        env=env,
        cwd=tmpdir,
    )
    if proc.returncode != 0:
        return {"status": "runner_error", "details": {}, "error": (proc.stderr or proc.stdout)[-2000:]}
    try:
        output = json.loads(proc.stdout.strip().splitlines()[-1])
        return output
    except (json.JSONDecodeError, IndexError):
        return {"status": "runner_parse_error", "details": {}, "error": proc.stdout[-2000:]}


def _capture_installed_packages(venv_path: str) -> list[dict] | None:
    """Run uv pip list --format json in venv and return [{name, version}, ...]."""
    proc = subprocess.run(
        ["uv", "pip", "list", "--format", "json"],
        env={
            **os.environ,
            "VIRTUAL_ENV": venv_path,
            "PATH": f"{venv_path}/bin:{os.environ.get('PATH', '')}",
        },
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        out = json.loads(proc.stdout)
        return out if isinstance(out, list) else None
    except json.JSONDecodeError:
        return None


def run_ty_check(
    code: str,
    pkg_specs: list[str],
    python_version: str = "3.12",
    *,
    uv_install_timeout_seconds: int = 600,
) -> tuple[list[dict], list[dict] | None]:
    """Run ty in an isolated venv and parse json diagnostics.

    Creates a fresh virtual environment, installs dependencies, and runs type checking
    to avoid dependency resolution issues with the current environment.

    Args:
        code: User code to check.
        pkg_specs: Package specifications (e.g., ["requests==2.28.0", "loguru", "tqdm"]).

    Returns:
        list of dicts with "message" and "rule" fields.
    """
    enforce_uv_cache_limit(max_gb=300)
    with tempfile.TemporaryDirectory(prefix="tpl_ty_") as tmpdir:
        code_path = os.path.join(tmpdir, "snippet.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        venv_path = os.path.join(tmpdir, "venv")
        logger.info(f"[ty] Creating isolated venv at {venv_path}")
        proc = subprocess.run(
            ["uv", "venv", "--python", python_version, venv_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            logger.warning(f"[ty] Failed to create venv: {proc.stderr}")
            return [{"message": f"Failed to create venv: {proc.stderr}", "rule": "ty-venv-error"}], None

        venv_python = os.path.join(venv_path, "bin", "python")
        all_packages = ["ty"] + pkg_specs
        install_cmd = ["uv", "pip", "install", "--python", venv_python] + all_packages
        logger.info(f"[ty] Installing packages: {all_packages} with python version: {python_version}")

        try:
            proc = subprocess.run(
                install_cmd,
                env={**os.environ, "VIRTUAL_ENV": venv_path, "PATH": f"{venv_path}/bin:{os.environ.get('PATH', '')}"},
                capture_output=True,
                text=True,
                timeout=uv_install_timeout_seconds if uv_install_timeout_seconds and uv_install_timeout_seconds > 0 else None,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"[ty] uv install timed out after {uv_install_timeout_seconds}s, python={python_version}, specs_count={len(pkg_specs)}"
            )
            return [
                {
                    "message": f"uv install timed out after {uv_install_timeout_seconds}s",
                    "rule": "ty-install-timeout",
                }
            ], None

        if proc.returncode != 0:
            logger.warning(f"[ty] Failed to install packages: {proc.stderr}")
            return [{"message": f"Failed to install packages: {proc.stderr}", "rule": "ty-install-error"}], None

        installed_packages = _capture_installed_packages(venv_path)
        ty_path = os.path.join(venv_path, "bin", "ty")
        cmd = [ty_path, "check", "--output-format", "gitlab", code_path]
        logger.info(f"[ty] Running type check with specs={pkg_specs}")
        proc = subprocess.run(
            cmd,
            env={**os.environ, "VIRTUAL_ENV": venv_path, "PATH": f"{venv_path}/bin:{os.environ.get('PATH', '')}"},
            capture_output=True,
            text=True,
        )
        logger.info(f"[ty] Completed with returncode={proc.returncode}")

        if not proc.stdout.strip() and not proc.stderr.strip():
            return [], installed_packages
        if proc.stderr.strip():
            logger.warning(f"[ty] stderr={proc.stderr.strip()[:500]}")
        if not proc.stdout.strip():
            return [{"message": "ty produced no diagnostics output", "rule": "ty-empty-output"}], installed_packages
        try:
            raw = json.loads(proc.stdout)
            if not isinstance(raw, list):
                logger.warning("[ty] Unexpected output payload type; wrapping raw text.")
                return [{"message": proc.stdout.strip(), "rule": "ty-unexpected-payload"}], installed_packages
            mapped = []
            for item in raw:
                check_name = item.get("check_name")
                description = item.get("description") or ""
                mapped.append(
                    {
                        "rule": check_name,
                        "message": description,
                        "raw": item,
                    }
                )
            logger.info(f"[ty] Parsed diagnostics entries={len(mapped)}")
            return mapped, installed_packages
        except json.JSONDecodeError:
            logger.warning("[ty] Failed to parse JSON output, returning raw diagnostic wrapper.")
            return [{"message": proc.stdout.strip(), "rule": "ty-output-parse-failed"}], installed_packages
        finally:
            enforce_uv_cache_limit(max_gb=300)


def run_ty_and_bcb_in_isolated_venv(
    code: str,
    pkg_specs: list[str],
    *,
    python_version: str = "3.12",
    bcb_problem: dict[str, Any] | None = None,
    bcb_timeout_seconds: int = 180,
    uv_install_timeout_seconds: int = 600,
    project_root: str | None = None,
) -> tuple[list[dict], dict[str, Any] | None, list[dict] | None, dict[str, Any] | None]:
    """Run ty static check then optionally BigCodeBench test in a single uv venv.

    Creates one venv, installs ty + pkg_specs, runs ty first, then runs BCB unittest
    if bcb_problem is provided. This avoids building the environment twice.

    Returns:
        (ty_diags, bcb_result, installed_packages, test_utilized_tpl).
        ty_diags is list of dicts with message/rule.
        bcb_result is None if bcb_problem is None, else {status, details, error}.
        installed_packages is uv pip list --format json output [{name, version}, ...] or None.
        test_utilized_tpl records extra test-only dependencies and install status.
    """
    enforce_uv_cache_limit(max_gb=300)
    bcb_result: dict[str, Any] | None = None
    test_utilized_tpl: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="tpl_ty_bcb_") as tmpdir:
        code_path = os.path.join(tmpdir, "snippet.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        venv_path = os.path.join(tmpdir, "venv")
        logger.info(f"[ty+bcb] Creating isolated venv at {venv_path} (python={python_version})")
        proc = subprocess.run(
            ["uv", "venv", "--python", python_version, venv_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            msg = f"Failed to create venv: {proc.stderr}"
            logger.warning(f"[ty+bcb] {msg}")
            bcb_err: dict[str, Any] | None = (
                {"status": "error", "error": f"venv_failed: {msg[:1000]}"}
                if bcb_problem else None
            )
            return [{"message": msg, "rule": "ty-venv-error"}], bcb_err, None, None

        venv_python = os.path.join(venv_path, "bin", "python")
        all_packages = ["ty"] + pkg_specs
        install_cmd = ["uv", "pip", "install", "--python", venv_python] + all_packages
        logger.info(f"[ty+bcb] Installing packages: {all_packages}")
        try:
            proc = subprocess.run(
                install_cmd,
                env={**os.environ, "VIRTUAL_ENV": venv_path, "PATH": f"{venv_path}/bin:{os.environ.get('PATH', '')}"},
                capture_output=True,
                text=True,
                timeout=uv_install_timeout_seconds if uv_install_timeout_seconds and uv_install_timeout_seconds > 0 else None,
            )
        except subprocess.TimeoutExpired:
            msg = f"uv install timed out after {uv_install_timeout_seconds}s"
            logger.warning(f"[ty+bcb] {msg}")
            bcb_err = (
                {"status": "error", "error": f"install_timeout: {msg}"}
                if bcb_problem else None
            )
            return [{"message": msg, "rule": "ty-install-timeout"}], bcb_err, None, None
        if proc.returncode != 0:
            msg = f"Failed to install packages: {proc.stderr}"
            logger.warning(f"[ty+bcb] {msg}")
            bcb_err = (
                {"status": "error", "error": f"install_failed: {msg[:1000]}"}
                if bcb_problem else None
            )
            return [{"message": msg, "rule": "ty-install-error"}], bcb_err, None, None

        # Capture installed packages after installing specified_tpl (includes transitive deps).
        installed_packages = _capture_installed_packages(venv_path)

        # Extra install step: test-only third-party imports not in specified pkg_specs.
        # Only install those NOT already present via transitive deps.
        if bcb_problem and bcb_problem.get("test"):
            test_code = str(bcb_problem.get("test") or "")
            detected = _infer_test_utilized_tpl(test_code, pkg_specs)
            # convert detected to pypi package names
            detected = [get_pkg_name(m) for m in detected]

            installed_norm: set[str] = set()
            if installed_packages:
                for p in installed_packages:
                    if isinstance(p, dict) and p.get("name"):
                        installed_norm.add(_normalize_pkg_name(str(p["name"])))

            already_present = sorted([m for m in detected if _normalize_pkg_name(m) in installed_norm])
            to_install = sorted([m for m in detected if _normalize_pkg_name(m) not in installed_norm])

            test_utilized_tpl = {
                "detected": detected,
                "already_present": already_present,
                "to_install": to_install,
                "installed": [],
                "install_failed": [],
            }
            if to_install:
                logger.info(f"[ty+bcb] Installing test_utilized_tpl packages (not preinstalled): {to_install}")
                try:
                    extra_proc = subprocess.run(
                        ["uv", "pip", "install", "--python", venv_python] + to_install,
                        env={**os.environ, "VIRTUAL_ENV": venv_path, "PATH": f"{venv_path}/bin:{os.environ.get('PATH', '')}"},
                        capture_output=True,
                        text=True,
                        timeout=uv_install_timeout_seconds if uv_install_timeout_seconds and uv_install_timeout_seconds > 0 else None,
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"[ty+bcb] test_utilized_tpl install timed out after {uv_install_timeout_seconds}s"
                    )
                    test_utilized_tpl["install_failed"] = list(to_install)
                    test_utilized_tpl["error"] = f"uv install timed out after {uv_install_timeout_seconds}s"
                    extra_proc = None
                if extra_proc is None:
                    pass
                elif extra_proc.returncode == 0:
                    test_utilized_tpl["installed"] = list(to_install)
                else:
                    logger.warning(
                        f"[ty+bcb] Failed to install test_utilized_tpl packages: {extra_proc.stderr[:500]}"
                    )
                    test_utilized_tpl["install_failed"] = list(to_install)
                    test_utilized_tpl["error"] = (extra_proc.stderr or extra_proc.stdout)[-2000:]

        ty_path = os.path.join(venv_path, "bin", "ty")
        logger.info(f"[ty+bcb] Running ty check with specs={pkg_specs}")
        proc = subprocess.run(
            [ty_path, "check", "--output-format", "gitlab", code_path],
            env={**os.environ, "VIRTUAL_ENV": venv_path, "PATH": f"{venv_path}/bin:{os.environ.get('PATH', '')}"},
            capture_output=True,
            text=True,
        )
        logger.info(f"[ty+bcb] ty completed with returncode={proc.returncode}")

        ty_diags: list[dict] = []
        if proc.stdout.strip() or proc.stderr.strip():
            if proc.stderr.strip():
                logger.warning(f"[ty+bcb] ty stderr={proc.stderr.strip()[:500]}")
            if proc.stdout.strip():
                try:
                    raw = json.loads(proc.stdout)
                    if isinstance(raw, list):
                        for item in raw:
                            ty_diags.append({
                                "rule": item.get("check_name"),
                                "message": item.get("description") or "",
                                "raw": item,
                            })
                    else:
                        ty_diags.append({"message": proc.stdout.strip(), "rule": "ty-unexpected-payload"})
                except json.JSONDecodeError:
                    ty_diags.append({"message": proc.stdout.strip(), "rule": "ty-output-parse-failed"})

        if bcb_problem and bcb_problem.get("test"): # all the tasks should have bcb problem and test
            logger.info("[ty+bcb] Running BigCodeBench test in same venv")
            bcb_result = _run_bcb_test_in_venv(
                venv_python=venv_python,
                code=code,
                problem=bcb_problem,
                tmpdir=tmpdir,
                timeout_seconds=bcb_timeout_seconds,
                project_root=project_root,
            )
            logger.info(f"[ty+bcb] BCB result: status={bcb_result.get('status')}")

        enforce_uv_cache_limit(max_gb=300)
    return ty_diags, bcb_result, installed_packages, test_utilized_tpl


def check_compatibility(
    code: str,
    *,
    pkg_specs: list[str] | None = None,
    python_version: str = "3.12",
) -> CompatResult:
    """Check compatibility via ty type checker.

    Args:
        code: User code to check.
        pkg_specs: Package specifications (e.g., ["requests==2.28.0"]).
        python_version: Python version for the isolated venv.

    Returns:
        CompatResult with compatibility status and errors.
    """
    logger.info(f"[compat] Start ty check, pkg_specs={pkg_specs or []}")

    errors: list[CompatError] = []
    ty_diags, _ = run_ty_check(code, pkg_specs or [], python_version=python_version)
    for diag in ty_diags:
        errors.append(
            CompatError(
                call_site=None,
                level="ty",
                message=diag.get("message", "ty diagnostic"),
                rule=diag.get("rule"),
            )
        )
    result = CompatResult(is_compatible=not errors, level_reached="ty", errors=errors)
    logger.info(f"[compat] Finished ty check, errors={len(result.errors)}")
    return result


def check_compatibility_for_pkg(
    code: str,
    *,
    pkg_specs: list[str] | None = None,
    python_version: str = "3.12",
    bcb_problem: dict[str, Any] | None = None,
    bcb_timeout_seconds: int = 180,
    uv_install_timeout_seconds: int = 600,
    project_root: str | None = None,
) -> dict:
    """Run ty compatibility check and optionally BigCodeBench test in one venv.

    When bcb_problem is provided (with 'test' and 'entry_point'), runs ty first
    then BCB test in the same uv environment.

    Returns:
        dict with is_compatible, level_reached, errors, and optionally bcb_test.
    """
    specs = pkg_specs or []
    if not specs and not bcb_problem:
        result = check_compatibility(code, pkg_specs=[], python_version=python_version)
        out = {
            "is_compatible": result.is_compatible,
            "level_reached": result.level_reached,
            "errors": [
                {"level": err.level, "message": err.message, "rule": err.rule, "call_site": None}
                for err in result.errors
            ],
        }
        return out

    if bcb_problem and bcb_problem.get("test"):  # all the tasks should have bcb problem and test
        ty_diags, bcb_result, installed_packages, test_utilized_tpl = run_ty_and_bcb_in_isolated_venv(
            code,
            specs,
            python_version=python_version,
            bcb_problem=bcb_problem,
            bcb_timeout_seconds=bcb_timeout_seconds,
            uv_install_timeout_seconds=uv_install_timeout_seconds,
            project_root=project_root,
        )
        errors = [
            CompatError(call_site=None, level="ty", message=d.get("message", ""), rule=d.get("rule"))
            for d in ty_diags
        ]
        out = {
            "is_compatible": not errors,
            "level_reached": "ty",
            "errors": [{"level": e.level, "message": e.message, "rule": e.rule, "call_site": None} for e in errors],
            "bcb_test": bcb_result,
        }
        if installed_packages is not None:
            out["installed_packages"] = installed_packages
        if test_utilized_tpl is not None:
            out["test_utilized_tpl"] = test_utilized_tpl
        return out

    # D2 has no testcase/bcb_problem by design; run ty-only compatibility check.
    logger.info("[compat] no bcb problem provided, running ty-only compatibility check")
    ty_diags, installed_packages = run_ty_check(
        code,
        specs,
        python_version=python_version,
        uv_install_timeout_seconds=uv_install_timeout_seconds,
    )
    errors = [
        CompatError(call_site=None, level="ty", message=d.get("message", ""), rule=d.get("rule"))
        for d in ty_diags
    ]
    out = {
        "is_compatible": not errors,
        "level_reached": "ty",
        "errors": [{"level": e.level, "message": e.message, "rule": e.rule, "call_site": None} for e in errors],
    }
    if installed_packages is not None:
        out["installed_packages"] = installed_packages
    return out


def load_bigcodebench_problems(path: str) -> dict[str, dict[str, Any]]:
    """Load BigCodeBench JSONL into task_id -> {test, entry_point}."""
    out: dict[str, dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row.get("task_id")
                if isinstance(task_id, str) and ("test" in row or "entry_point" in row):
                    out[task_id] = {"test": row.get("test", ""), "entry_point": row.get("entry_point", "")}
            except json.JSONDecodeError:
                continue
    return out


def check_records_compatibility(
    records: list[dict],
    python_version: str = "3.12",
    *,
    bcb_problems: dict[str, dict[str, Any]] | None = None,
    bcb_timeout_seconds: int = 180,
    project_root: str | None = None,
) -> list[dict]:
    """Check compatibility for all records: ty static check, then BigCodeBench test if available.

    For each record: creates ONE uv venv, installs deps + ty, runs ty first, then
    optionally runs BigCodeBench unittest in the same venv (avoids rebuilding).

    Args:
        records: List of records with 'code', 'per_lib', 'task_id' fields.
        python_version: Python version for the venv.
        bcb_problems: Optional task_id -> {test, entry_point}. When provided and
            task_id matches, runs BCB test after ty in the same venv.
        bcb_timeout_seconds: Timeout for BCB test execution.
        project_root: PYTHONPATH for BCB runner (required for some test imports).
    Returns:
        List of records with added 'compat_results' (ty + optional bcb_test).
    """
    import time as time_module
    from tqdm import tqdm

    if project_root is None:
        project_root = str(Path(__file__).resolve().parents[1])

    results = []
    for record in tqdm(records, desc="Compatibility analyze"):
        code = record.get("code", "")

        # resolve version specs to pkg_specs
        pkg_specs = []
        for lib in record.get("per_lib", []):
            pypi_name = lib.get("pypi_name")
            if not pypi_name:
                continue
            version_spec = lib.get("version_spec")
            resolved_version = lib.get("resolved_version")
            if version_spec:
                pkg_specs.append(f"{pypi_name}{version_spec}")
            elif resolved_version:
                pkg_specs.append(f"{pypi_name}=={resolved_version}")
            else:
                pkg_specs.append(pypi_name)

        task_id = record.get("task_id", "")
        bcb_problem = (bcb_problems or {}).get(task_id) if task_id else None

        if not pkg_specs and not bcb_problem:
            # No deps and no bcb problem: skip compat entirely.
            enriched = dict(record)
            enriched["compat_results"] = {
                "ty": {"is_compatible": True, "level_reached": "ty", "errors": []},
                "pkg_specs": [],
                "compat_meta": {"duration_ms": 0, "note": "no deps", "python_version": python_version},
            }
            results.append(enriched)
            continue
        # When pkg_specs is empty but bcb_problem exists (no-dep tasks that still have BCB tests),
        # fall through to the try block so BCB is still executed.

        try:
            logger.info(f"[compat] task={task_id} pkg_specs={pkg_specs} bcb={bcb_problem is not None}")
            # run ty compatibility check and optionally BigCodeBench test in one venv
            start_ts = time_module.perf_counter()
            ty_result = check_compatibility_for_pkg(
                code,
                pkg_specs=pkg_specs or [],
                python_version=python_version,
                bcb_problem=bcb_problem,
                bcb_timeout_seconds=bcb_timeout_seconds,
                project_root=project_root,
            )
            duration_ms = round((time_module.perf_counter() - start_ts) * 1000, 2)

            enriched = dict(record)
            compat_ty = {k: v for k, v in ty_result.items() if k not in ("bcb_test", "installed_packages")}
            bcb_test = ty_result.get("bcb_test") if ty_result else None
            installed_packages = ty_result.get("installed_packages") if ty_result else None
            compat_results = {
                "ty": compat_ty,
                "pkg_specs": pkg_specs,
                "compat_meta": {"duration_ms": duration_ms, "python_version": python_version},
            }
            # Always record bcb_test when a bcb_problem was provided, even if result is None
            # (None means install failed before BCB could run).
            if bcb_problem is not None:
                compat_results["bcb_test"] = bcb_test if bcb_test is not None else {
                    "status": "error", "error": "bcb_not_executed"
                }
            if installed_packages is not None:
                compat_results["installed_packages"] = installed_packages
            enriched["compat_results"] = compat_results
            results.append(enriched)
        except Exception as exc:
            logger.exception(f"[compat] failed task={task_id}: {exc}")
            error_result = {
                "is_compatible": False,
                "level_reached": "ty",
                "errors": [{"level": "ty", "message": str(exc), "rule": "compat-runtime-error"}],
            }
            enriched = dict(record)
            compat_results = {
                "ty": error_result,
                "pkg_specs": pkg_specs,
                "compat_meta": {"duration_ms": None, "python_version": python_version},
            }
            if bcb_problem is not None:
                compat_results["bcb_test"] = {"status": "error", "error": str(exc)}
            enriched["compat_results"] = compat_results
            results.append(enriched)

    return results