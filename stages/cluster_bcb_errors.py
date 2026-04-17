#!/usr/bin/env python3
"""
Cluster and summarize BCB error messages from compat_results.bcb_test.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m stages.cluster_bcb_errors --help

Input source:
- outputs/d1/**/m5_compat_records.json
- Only records with bcb_test.status == "fail"
- For fail records: bcb_test.details (tracebacks from test_case_*)

Output:
{
  "<cluster_label>": {
    "count": int,
    "example_messages": [ ... ]   # up to --max-examples
  },
  ...
}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_D1 = PROJECT_ROOT / "outputs" / "d1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster BCB error messages from m5_compat_records.json."
    )
    p.add_argument("--outputs-d1", type=Path, default=OUTPUTS_D1, help="Root outputs/d1 directory.")
    p.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUTS_D1 / "bcb_error_clusters.json",
        help="Where to write clustered summary.",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum example messages to keep per cluster.",
    )
    p.add_argument(
        "--python-version",
        type=str,
        default=None,
        help=(
            "Python version to cluster, e.g. 3.12. "
            "When set, only py<ver>/m5_compat_records.json are scanned."
        ),
    )
    return p.parse_args()


def normalize_msg(msg: str) -> str:
    s = re.sub(r"\s+", " ", msg).strip()
    s = re.sub(r"/tmp/[^\s]+", "<TMP_DIR>", s)
    s = re.sub(r"\.cache/uv/[^\s]+", "<UV_CACHE>", s)
    return s


def extract_exception_name(msg: str) -> str | None:
    if not msg:
        return None
    found = re.findall(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:", msg)
    if found:
        return found[-1]
    bare = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Iteration|NotFound))\b",
        msg,
    )
    if bare:
        return bare[-1]
    if "templatenotfound" in msg.lower():
        return "TemplateNotFound"
    return None


def classify_known_bucket(msg: str) -> str | None:
    low = msg.lower()
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
    if "connectionerror" in low or "network is unreachable" in low:
        return "network"
    if "timed out" in low or "timeout" in low:
        return "timeout"
    return None


def classify_bcb_message(msg: str) -> str:
    """Heuristic clustering for one task-level BCB message."""
    low = msg.lower()
    exc = extract_exception_name(msg)
    if exc:
        return f"fail::{exc}"
    bucket = classify_known_bucket(msg)
    if bucket:
        return f"fail::{bucket}"
    if "assert " in low or "assertion failed" in low:
        return "fail::assertion-fail"
    if "importerror" in low:
        return "fail::import-error"
    return "fail::other"


def choose_task_fail_message(bcb: dict) -> str:
    """
    Return task-level representative message for fail-case clustering.
    - fail: first non-empty traceback in details (sorted by key)
    """
    status = str(bcb.get("status") or "").strip().lower() or "unknown"
    if status != "fail":
        return ""
    details = bcb.get("details")
    if isinstance(details, dict):
        for _, v in sorted(details.items()):
            txt = normalize_msg(str(v or ""))
            if txt:
                return txt
    return ""


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

    clusters: Dict[str, Dict[str, object]] = defaultdict(lambda: {"count": 0, "example_messages": []})

    for f in m5_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue

        for rec in data:
            compat = rec.get("compat_results") or {}
            bcb = compat.get("bcb_test") if isinstance(compat, dict) else None
            if not isinstance(bcb, dict):
                continue
            msg = choose_task_fail_message(bcb)
            if not msg:
                continue

            label = classify_bcb_message(msg)
            entry = clusters[label]
            entry["count"] = int(entry["count"]) + 1  # type: ignore[assignment]
            examples: List[str] = entry["example_messages"]  # type: ignore[assignment]
            if len(examples) < args.max_examples and msg and msg not in examples:
                examples.append(msg)

    sorted_items = sorted(clusters.items(), key=lambda kv: -int(kv[1]["count"]))
    ordered: Dict[str, Dict[str, object]] = {}
    for label, payload in sorted_items:
        payload["example_messages"] = list(payload.get("example_messages", []))  # type: ignore[assignment]
        ordered[label] = payload

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Clustered BCB messages into {len(ordered)} groups -> {args.output_json}")


if __name__ == "__main__":
    main()

