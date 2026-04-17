"""Classification helpers for ty / BCB compatibility results.

Usage (import-only)::

    from stages.failure_classifier import classify_ty_error, classify_bcb_error_type

This module previously exposed a failure-layer taxonomy (F0–F5). We now move to
more descriptive, phase-specific labels:

- ty_error_type:
  - "none": ty reports is_compatible=True (or no result)
  - "env_error": environment creation / venv error (e.g., ty-venv-error)
  - "install_error": dependency installation error (ty-install-error)
  - "typecheck_error": regular ty diagnostics (unresolved-attribute, invalid-argument-type, etc.)
  - "infra_error": infrastructure issues surfaced via ty (compat-*) or similar
  - "other_error": anything not fitting above buckets

- ty_install_category: second-level classification for "install_error", aligned
  with stages/cluster_ty_install_errors.py and summarize_compat_error_rules.py
  (e.g., "python-version-constraint", "dependency-resolve-no-solution", etc.).

- bcb_error_type:
  - "none": no BCB result or status == "pass"
  - "semantic_fail": BCB status == "fail" (tests failed)
  - "import_error": import/initialization errors when running BCB tests
  - "infra_error": infrastructure / timeout / permission / disk errors
  - "other_error": any other non-pass, non-fail BCB status
"""

from __future__ import annotations
from loguru import logger
import re
from typing import Literal, Tuple

TyErrorType = Literal[
    "none",
    "env_error",
    "install_error",
    "typecheck_error",
    "infra_error",
    "other_error",
]

TyInstallCategory = str  # descriptive cluster label from cluster_ty_install_errors

BcbErrorType = Literal[
    "none",
    "semantic_fail",
    "import_error",
    "infra_error",
    "other_error",
]

# Patterns for install/env/infra classification
_F1A_PATTERNS = [
    r"Requires-Python",
    r"python_version",
    r"not supported",
    r"No matching distribution",
    r"unsupported platform",
    r"abi.*incompatible",
    r"platform.*incompatible",
    r"no wheel",
    r"no built distribution",
]

_F1B_PATTERNS = [
    r"dependency conflict",
    r"ResolutionImpossible",
    r"Could not find a version",
    r"conflicting dependencies",
    r"Incompatible",
    r"version conflict",
    r"constraint",
]

_F1C_PATTERNS = [
    r"Building wheel",
    r"error: command",
    r"subprocess",
    r"gcc|clang|cc ",
    r"rustc",
    r"openssl",
    r"building failed",
    r"setup\.py",
    r"error: legacy-install-failure",
]

# Patterns for F2 (import/init) vs F5 (infra)

_F2_MODULE_PATTERNS = [
    r"ModuleNotFoundError",
    r"No module named",
]

_F2_BINARY_PATTERNS = [
    r"ImportError.*DLL load",
    r"ImportError.*lib.*not found",
    r"OSError.*lib.*not found",
    r"symbol not found",
]

_F2_RUNTIME_PATTERNS = [
    r"ImportError",
    r"OSError.*import",
]

_F5_PATTERNS = [
    r"timeout",
    r"TimeoutError",
    r"timed out",
    r"NetworkError",
    r"ConnectionError",
    r"Permission denied",
    r"PermissionError",
    r"disk full",
    r"No space left",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    for p in patterns:
        if re.search(p, lower, re.IGNORECASE):
            return True
    return False


# Cluster labels from cluster_ty_install_errors → human-readable categories
_INSTALL_CLUSTER_LABELS: dict[str, TyInstallCategory] = {
    "missing-distutils-py312": "missing_distutils",
    "python-version-constraint": "python_version_constraint",
    "no-matching-distribution": "no_matching_distribution",
    "dependency-resolve-no-solution": "dependency_resolve_no_solution",
    "dependency-conflict": "dependency_conflict",
    "build-backend-failure": "build_backend_failure",
    "python-package-build-mismatch-pkg_resources": "pkg_resources_missing_build_dep",
    "python-dev-headers-missing": "python_dev_headers_missing",
    "c-lib-dev-missing-libxml2-libxslt": "libxml2_libxslt_dev_missing",
    "python-stdlib-compat-configparser": "stdlib_configparser_compat",
    "network-io-error": "network_io_error",
    "permission-error": "permission_error",
}


def classify_ty_install_error_message(message: str) -> Tuple[TyErrorType, TyInstallCategory | None]:
    """Classify ty-install-error by message.

    Returns (ty_error_type, ty_install_category). ty_error_type is always "install_error"
    on success; category is a descriptive cluster label or None.
    """
    if not message:
        return "install_error", None
    try:
        from stages.cluster_ty_install_errors import (
            normalize_msg as _norm_msg,
            classify_install_message as _classify_install,
        )
        norm = _norm_msg(message)
        label = _classify_install(norm)
        category = _INSTALL_CLUSTER_LABELS.get(label, label or "other")
        return "install_error", category
    except Exception:
        logger.exception("Error classifying ty-install-error message: %s", message)
        # Fallback: still mark as install_error but with unknown category.
        return "install_error", "other"


def classify_venv_failure(stderr: str) -> TyErrorType:
    """Classify uv venv creation failure as an env_error (venv/py mismatch) or infra_error."""
    if not stderr:
        return "env_error"
    s = stderr[:2000]
    if _match_any(s, _F1A_PATTERNS) or "python" in s.lower() and "not found" in s.lower():
        return "env_error"
    if _match_any(s, _F5_PATTERNS):
        return "infra_error"
    return "env_error"


def classify_bcb_error(error: str) -> BcbErrorType:
    """Classify BCB runner error into import_error / infra_error / other_error."""
    if not error:
        return "other_error"
    s = error[:4000]

    if _match_any(s, _F5_PATTERNS):
        return "infra_error"
    if _match_any(s, _F2_MODULE_PATTERNS):
        return "import_error"
    if _match_any(s, _F2_BINARY_PATTERNS):
        return "import_error"
    if _match_any(s, _F2_RUNTIME_PATTERNS):
        return "import_error"

    return "other_error"


# ty rules → F3-L1 (Missing Symbol): missing API surface / unresolved symbols
# Aligned with compat_error_rule_summary + https://docs.astral.sh/ty/reference/rules/
_TY_F3_L1_RULES = frozenset({
    "unresolved-attribute",
    "unresolved-import",
    "unresolved-reference",
    "unresolved-global",
    "possibly-missing-attribute",
    "possibly-missing-import",
    "possibly-unresolved-reference",
    "possibly-missing-submodule",
})

# ty rules → F3-L2 (Signature Mismatch): symbol exists but signature incompatible
_TY_F3_L2_RULES = frozenset({
    "missing-argument",
    "too-many-positional-arguments",
    "unknown-argument",
    "parameter-already-assigned",
    "positional-only-parameter-as-kwarg",
    "no-matching-overload",
    "invalid-method-override",
})

# ty rules → F3-L3 (Type-level / syntax / deprecated): typing, syntax, deprecation, etc.
# All other ty rules that appear in compat_error_rule_summary go here so we never invent a phantom tier.
_TY_F3_L3_RULES = frozenset({
    "invalid-argument-type",
    "invalid-assignment",
    "invalid-return-type",
    "invalid-type-form",
    "not-subscriptable",
    "not-iterable",
    "deprecated",
    "invalid-syntax",
    "call-non-callable",
    "call-top-callable",  # typo variant seen in data
    "unsupported-operator",
    "unused-type-ignore-comment",
})


def classify_ty_rule(rule: str | None, message: str) -> str:
    """Map ty diagnostic rule to a coarse typecheck category (currently unused, kept for potential future detail)."""
    if not rule:
        return "typecheck_error"
    r = rule.strip().lower()
    if r in _TY_F3_L1_RULES:
        return "typecheck_error"
    if r in _TY_F3_L2_RULES:
        return "typecheck_error"
    if r in _TY_F3_L3_RULES:
        return "typecheck_error"
    return "typecheck_error"


def classify_ty_error(ty_result: dict | None) -> Tuple[TyErrorType, TyInstallCategory | None]:
    """Classify ty_result into (ty_error_type, ty_install_category).

    ty_install_category is only meaningful when ty_error_type == "install_error".
    """
    ty = ty_result or {}
    if ty.get("is_compatible"):
        return "none", None

    errors = ty.get("errors") or []
    if not errors:
        return "other_error", None

    first_err = errors[0] if isinstance(errors[0], dict) else {}
    rule = first_err.get("rule")
    message = str(first_err.get("message", ""))

    if rule == "ty-venv-error":
        return classify_venv_failure(message), None
    if rule == "ty-install-error":
        return classify_ty_install_error_message(message)
    if rule and str(rule).startswith("compat-"):
        return "infra_error", None

    # Regular ty diagnostics → typecheck_error
    return "typecheck_error", None


def classify_bcb_error_type(bcb_result: dict | None) -> BcbErrorType:
    """Classify bcb_test result into a coarse error type."""
    if not isinstance(bcb_result, dict):
        return "none"
    status = str(bcb_result.get("status", "") or "")
    if not status or status == "pass":
        return "none"
    if status == "fail":
        return "semantic_fail"
    # error / runner_error / runner_parse_error etc.
    return classify_bcb_error(str(bcb_result.get("error", "") or ""))