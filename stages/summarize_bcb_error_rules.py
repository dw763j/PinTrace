#!/usr/bin/env python3
"""
Summarize compat_results.bcb_test error rules across models and runs.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m stages.summarize_bcb_error_rules --help

Scans outputs/d1/**/m5_compat_records.json and summarizes:
- bcb_test.status distribution
- task-level primary bcb rule distribution

Note:
- For status == "fail", bcb_test.details usually contains multiple test_case traces.
  We infer one primary rule per task (first parsable traceback), avoiding repeated
  counting of the same root cause across test cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_D1 = PROJECT_ROOT / "outputs" / "d1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BcbRuleSummary:
    total_tasks: int
    bcb_present_tasks: int
    bcb_fail_tasks: int
    fail_rate_among_present: float | None
    pass_tasks: int
    bcb_error_tasks: int
    runner_error_tasks: int
    ty_install_error_tasks: int
    non_ty_install_error_tasks: int
    other_status_tasks: int
    coverage_check_total: int
    total_rule_entries: int
    rule_counts: Dict[str, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize compat_results.bcb_test statuses and inferred rules."
    )
    p.add_argument("--outputs-d1", type=Path, default=OUTPUTS_D1, help="Root outputs/d1 directory.")
    p.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUTS_D1 / "bcb_error_rule_summary.json",
        help="Where to write the JSON summary.",
    )
    p.add_argument(
        "--python-version",
        type=str,
        default=None,
        help=(
            "Python version to summarize, e.g. 3.12. "
            "When set, only py<ver>/m5_compat_records.json are scanned."
        ),
    )
    return p.parse_args()


def infer_mode_and_model(path: Path) -> tuple[str, str]:
    """Infer (prompt_mode, model_key) from outputs/d1 file path."""
    parts = path.parts
    try:
        idx = parts.index("d1")
    except ValueError:
        return "unknown", "unknown"

    mode = parts[idx + 1] if len(parts) > idx + 1 else "unknown"
    run_name = parts[idx + 2] if len(parts) > idx + 2 else "unknown"

    model = run_name
    for suffix in (
        "_inline",
        "_inline_no_vuln",
        "_requirements.txt",
        "_req",
        "-req",
    ):
        if model.endswith(suffix):
            model = model[: -len(suffix)]
    return mode, model


def _normalize_text(msg: str) -> str:
    s = re.sub(r"\s+", " ", msg).strip()
    s = re.sub(r"/tmp/[^\s]+", "<TMP_DIR>", s)
    s = re.sub(r"\.cache/uv/[^\s]+", "<UV_CACHE>", s)
    return s


def _extract_exception_token(text: str) -> str | None:
    """
    Extract final exception token from traceback-like text.
    Examples: ValueError, AssertionError, ModuleNotFoundError, ...
    """
    if not text:
        return None
    candidates = re.findall(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:", text)
    if candidates:
        return candidates[-1]
    # Bare exception symbols without trailing ":" (e.g. "raise effect ValueError", "StopIteration")
    bare = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Iteration|NotFound))\b",
        text,
    )
    if bare:
        return bare[-1]
    # Some frameworks raise named exceptions not ending with Error/Exception.
    if "templatenotfound" in text.lower():
        return "TemplateNotFound"
    return None


def _classify_known_bucket(text: str) -> str | None:
    """Heuristic fallback buckets when no explicit exception token exists."""
    low = text.lower()
    if not low:
        return "empty-message"
    if "failed to acquire lock" in low and ("timeout" in low or "timed out" in low):
        return "uv-cache-lock-timeout"
    if "no solution found when resolving dependencies" in low or "unsatisfiable" in low:
        return "dependency-unsatisfiable"
    if "requires-python" in low or "unsupported python version" in low:
        return "python-version-constraint"
    if "distutils" in low and "removed from the standard library" in low:
        return "py312-distutils-removed"
    if "failed to build" in low or "build backend returned an error" in low:
        return "build-backend-failure"
    if "qt.qpa.xcb" in low or "could not connect to display" in low:
        return "headless-display-qt"
    if "numpy.dtype size changed" in low or "binary incompatibility" in low:
        return "binary-incompatibility"
    if "cannot import name" in low:
        return "import-name-error"
    if "modulenotfounderror" in low or "no module named" in low:
        return "module-not-found"
    if "template not found" in low or "templatenotfound" in low:
        return "template-not-found"
    if "permission denied" in low or "permissionerror" in low:
        return "permission"
    if "no space left" in low or "disk full" in low:
        return "disk"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "connectionerror" in low or "network is unreachable" in low:
        return "network"
    return None


def infer_bcb_rule_from_test(bcb_test: dict[str, Any]) -> str:
    """
    Infer one task-level primary BCB rule.

    Rule format: bcb-fail::<exception-or-bucket>
    """
    details = bcb_test.get("details")
    if isinstance(details, dict):
        for _, v in sorted(details.items()):
            txt = _normalize_text(str(v or ""))
            if not txt:
                continue
            exc = _extract_exception_token(txt)
            if exc:
                return f"bcb-fail::{exc}"
            bucket = _classify_known_bucket(txt)
            if bucket:
                return f"bcb-fail::{bucket}"
    return "bcb-fail::unknown"


def summarize_file(
    path: Path,
    agg_rules: dict[tuple[str, str], Counter],
    agg_total_tasks: dict[tuple[str, str], int],
    agg_bcb_present: dict[tuple[str, str], int],
    agg_bcb_fail: dict[tuple[str, str], int],
    agg_pass: dict[tuple[str, str], int],
    agg_bcb_error: dict[tuple[str, str], int],
    agg_runner_error: dict[tuple[str, str], int],
    agg_ty_install_error: dict[tuple[str, str], int],
    agg_non_ty_install_error: dict[tuple[str, str], int],
    agg_other_status: dict[tuple[str, str], int],
) -> None:
    mode, model = infer_mode_and_model(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return

    key = (model, mode)
    if key not in agg_rules:
        agg_rules[key] = Counter()
    agg_total_tasks[key] = agg_total_tasks.get(key, 0) + len(data)

    for rec in data:
        compat = rec.get("compat_results") or {}
        ty = compat.get("ty") if isinstance(compat, dict) else None
        has_ty_install_error = False
        ty_errors = ty.get("errors") if isinstance(ty, dict) else None
        if isinstance(ty_errors, list):
            has_ty_install_error = any(
                isinstance(err, dict) and str(err.get("rule") or "") == "ty-install-error"
                for err in ty_errors
            )

        bcb = compat.get("bcb_test") if isinstance(compat, dict) else None
        if not isinstance(bcb, dict):
            continue

        agg_bcb_present[key] = agg_bcb_present.get(key, 0) + 1
        status = str(bcb.get("status") or "").strip().lower() or "unknown"
        if status == "pass":
            agg_pass[key] = agg_pass.get(key, 0) + 1
            continue
        if status == "error":
            agg_bcb_error[key] = agg_bcb_error.get(key, 0) + 1
            if has_ty_install_error:
                agg_ty_install_error[key] = agg_ty_install_error.get(key, 0) + 1
            else:
                agg_non_ty_install_error[key] = agg_non_ty_install_error.get(key, 0) + 1
            continue
        if status == "runner_error":
            agg_runner_error[key] = agg_runner_error.get(key, 0) + 1
            continue
        if status != "fail":
            agg_other_status[key] = agg_other_status.get(key, 0) + 1
            continue

        agg_bcb_fail[key] = agg_bcb_fail.get(key, 0) + 1
        rule = infer_bcb_rule_from_test(bcb)
        agg_rules[key][rule] += 1


def main() -> None:
    args = parse_args()
    outputs_d1: Path = args.outputs_d1
    if not outputs_d1.is_dir():
        raise SystemExit(f"outputs/d1 not found: {outputs_d1}")

    if args.python_version:
        py_tag = f"py{args.python_version.replace('.', '')}"
        m5_files = sorted(outputs_d1.glob(f"**/{py_tag}/m5_compat_records.json"))
        if not m5_files:
            m5_files = sorted(outputs_d1.glob("**/m5_compat_records.json"))
    else:
        m5_files = sorted(outputs_d1.glob("**/m5_compat_records.json"))
    if not m5_files:
        raise SystemExit(f"No m5_compat_records.json found under {outputs_d1}")

    agg_rules: dict[tuple[str, str], Counter] = {}
    agg_total_tasks: dict[tuple[str, str], int] = {}
    agg_bcb_present: dict[tuple[str, str], int] = {}
    agg_bcb_fail: dict[tuple[str, str], int] = {}
    agg_pass: dict[tuple[str, str], int] = {}
    agg_bcb_error: dict[tuple[str, str], int] = {}
    agg_runner_error: dict[tuple[str, str], int] = {}
    agg_ty_install_error: dict[tuple[str, str], int] = {}
    agg_non_ty_install_error: dict[tuple[str, str], int] = {}
    agg_other_status: dict[tuple[str, str], int] = {}

    for f in m5_files:
        summarize_file(
            f,
            agg_rules,
            agg_total_tasks,
            agg_bcb_present,
            agg_bcb_fail,
            agg_pass,
            agg_bcb_error,
            agg_runner_error,
            agg_ty_install_error,
            agg_non_ty_install_error,
            agg_other_status,
        )

    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model, mode), counter in agg_rules.items():
        total_entries = sum(counter.values())
        present = agg_bcb_present.get((model, mode), 0)
        fail_tasks = agg_bcb_fail.get((model, mode), 0)
        pass_tasks = agg_pass.get((model, mode), 0)
        bcb_error_tasks = agg_bcb_error.get((model, mode), 0)
        runner_error_tasks = agg_runner_error.get((model, mode), 0)
        ty_install_error_tasks = agg_ty_install_error.get((model, mode), 0)
        non_ty_install_error_tasks = agg_non_ty_install_error.get((model, mode), 0)
        other_status_tasks = agg_other_status.get((model, mode), 0)
        coverage_check_total = (
            fail_tasks
            + pass_tasks
            + bcb_error_tasks
            + runner_error_tasks
            + other_status_tasks
        )
        summary = BcbRuleSummary(
            total_tasks=agg_total_tasks.get((model, mode), 0),
            bcb_present_tasks=present,
            bcb_fail_tasks=fail_tasks,
            fail_rate_among_present=(fail_tasks / present) if present else None,
            pass_tasks=pass_tasks,
            bcb_error_tasks=bcb_error_tasks,
            runner_error_tasks=runner_error_tasks,
            ty_install_error_tasks=ty_install_error_tasks,
            non_ty_install_error_tasks=non_ty_install_error_tasks,
            other_status_tasks=other_status_tasks,
            coverage_check_total=coverage_check_total,
            total_rule_entries=total_entries,
            rule_counts=dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        )
        out[model][mode] = asdict(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote summary for {len(agg_rules)} (model, mode) pairs to {args.output_json}")


if __name__ == "__main__":
    main()

