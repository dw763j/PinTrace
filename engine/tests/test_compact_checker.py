"""Test compact_checker: verify installation and ty check with special version specifiers."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.compact_checker import run_ty_check, check_compatibility, check_records_compatibility


def test_run_ty_check_with_version_specifier():
    """Verify run_ty_check can install TPL with >= specifier and complete type check."""
    code = """
import numpy as np

def foo(x: np.ndarray) -> float:
    return float(np.mean(x))
"""
    pkg_specs = ["numpy>=1.20"]
    diags, _ = run_ty_check(code, pkg_specs, python_version="3.12")
    assert isinstance(diags, list)
    # No install error
    install_errors = [d for d in diags if d.get("rule") in ("ty-venv-error", "ty-install-error")]
    assert not install_errors, f"Install failed: {install_errors}"


def test_check_compatibility_with_mixed_specs():
    """Verify check_compatibility works with exact version, version_spec, and unpinned."""
    code = """
import requests
import numpy as np

def fetch(url: str) -> requests.Response:
    return requests.get(url)

def mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))
"""
    # Mix: exact (requests), version_spec (numpy>=1.20)
    pkg_specs = ["requests==2.28.0", "numpy>=1.20,<2.0"]
    result = check_compatibility(code, pkg_specs=pkg_specs, python_version="3.12")
    assert result is not None
    assert result.level_reached == "ty"
    # Should complete without install/runtime crash
    crash_errors = [e for e in result.errors if e.rule in ("ty-venv-error", "ty-install-error", "compat-runtime-error")]
    assert not crash_errors, f"Check crashed: {[e.message for e in crash_errors]}"


def test_check_records_compatibility_uses_m3_resolved_version():
    """M5 must not resolve a range again after M3 chose an exact version."""
    records = [
        {
            "task_id": "test/1",
            "code": "import numpy as np\nx: np.ndarray = np.array([1, 2, 3])\nprint(np.mean(x))",
            "per_lib": [
                {
                    "pypi_name": "numpy",
                    "resolved_version": "1.26.0",
                    "version_spec": ">=1.20",
                    "specified_version": None,
                },
            ],
        },
    ]
    with patch("stages.compact_checker.check_compatibility_for_pkg") as mocked:
        mocked.return_value = {
            "is_compatible": True,
            "level_reached": "ty",
            "errors": [],
            "installed_packages": [{"name": "numpy", "version": "1.26.0"}],
        }
        results = check_records_compatibility(records, python_version="3.12")
    assert len(results) == 1
    r = results[0]
    assert "compat_results" in r
    ty_result = r["compat_results"].get("ty", {})
    pkg_specs = r["compat_results"].get("pkg_specs", [])
    assert "numpy==1.26.0" in pkg_specs
    assert "is_compatible" in ty_result
    # installed_packages from uv pip list --format json
    installed = r["compat_results"].get("installed_packages")
    assert installed is not None and isinstance(installed, list)
    pkg_names = [p.get("name") for p in installed]
    assert "numpy" in pkg_names
    # No install error
    errors = ty_result.get("errors", [])
    install_errors = [e for e in errors if isinstance(e, dict) and e.get("rule") in ("ty-venv-error", "ty-install-error")]
    assert not install_errors, f"Install failed: {install_errors}"


def test_check_records_passes_m3_lock_to_m5():
    lock_text = "requests==2.31.0 --hash=sha256:abc\n"
    records = [
        {
            "task_id": "test/locked",
            "code": "import requests",
            "per_lib": [{"pypi_name": "requests", "resolved_version": "2.31.0"}],
            "resolver": {"lock_text": lock_text},
        }
    ]
    with patch("stages.compact_checker.check_compatibility_for_pkg") as mocked:
        mocked.return_value = {"is_compatible": True, "level_reached": "ty", "errors": []}
        check_records_compatibility(records, python_version="3.12", ty_version="0.0.23")
    kwargs = mocked.call_args.kwargs
    assert kwargs["lock_text"] == lock_text
    assert kwargs["pkg_specs"] == ["requests==2.31.0"]
    assert kwargs["ty_version"] == "0.0.23"


def test_check_records_does_not_run_m5_after_resolver_failure():
    records = [
        {
            "task_id": "test/failed-lock",
            "code": "import requests",
            "per_lib": [{"pypi_name": "requests", "resolved_version": None}],
            "resolver": {"status": "failed", "lock_text": "", "stderr": "no solution"},
        }
    ]
    with patch("stages.compact_checker.check_compatibility_for_pkg") as mocked:
        result = check_records_compatibility(records, python_version="3.12")[0]
    mocked.assert_not_called()
    error = result["compat_results"]["ty"]["errors"][0]
    assert error["rule"] == "ty-install-error"
    assert "no solution" in error["message"]


if __name__ == "__main__":
    test_run_ty_check_with_version_specifier()
    print("test_run_ty_check_with_version_specifier: OK")
    test_check_compatibility_with_mixed_specs()
    print("test_check_compatibility_with_mixed_specs: OK")
    test_check_records_compatibility_uses_m3_resolved_version()
    print("test_check_records_compatibility_uses_m3_resolved_version: OK")
    print("All compact_checker tests passed.")
