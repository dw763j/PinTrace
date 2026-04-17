#!/usr/bin/env python3
"""
Cluster and summarize `ty-install-error` messages from compat_results.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m stages.cluster_ty_install_errors --help

It scans all outputs/d1/**/m5_compat_records.json, collects
compat_results.ty.errors[*] where rule == "ty-install-error",
and groups messages into coarse-grained categories based on
error text (e.g., distutils removed, no matching distribution,
dependency conflict, build failure, network issues, etc.).

Output JSON structure (per cluster label):
{
  "<cluster_label>": {
    "count": int,
    "example_messages": [msg1, msg2, ...]  # up to N examples
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
        description="Cluster ty-install-error messages from m5_compat_records.json."
    )
    p.add_argument(
        "--outputs-d1",
        type=Path,
        default=OUTPUTS_D1,
        help="Root outputs/d1 directory.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUTS_D1 / "ty_install_error_clusters.json",
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
            "Python version whose compat results to cluster, e.g. 3.12. "
            "When set, only py<ver>/m5_compat_records.json files are scanned. "
            "When unset, all m5_compat_records.json files are scanned (legacy behavior)."
        ),
    )
    return p.parse_args()


def normalize_msg(msg: str) -> str:
    """Lightly normalize message text for clustering."""
    # Collapse whitespace, strip tmp paths and uv cache paths roughly.
    s = re.sub(r"\s+", " ", msg).strip()
    # Remove tmpdir paths under /tmp and uv build cache paths to reduce noise.
    s = re.sub(r"/tmp/[^\s]+", "<TMP_DIR>", s)
    s = re.sub(r"\.cache/uv/[^\s]+", "<UV_CACHE>", s)
    return s


def classify_install_message(msg: str) -> str:
    """Heuristic clustering for ty-install-error messages."""
    s = msg.lower()

    if "distutils" in s and "python 3.12" in s:
        return "missing-distutils-py312"

    if "no solution found when resolving dependencies" in s:
        return "dependency-resolve-no-solution"

    if "depends on `pkg_resources`, but doesn't declare it as a build dependency" in s:
        return "python-package-build-mismatch-pkg_resources"

    if "no matching distribution" in s or "no matching version" in s:
        return "no-matching-distribution"

    if "resolutionimpossible" in s or "dependency conflict" in s or "conflicting dependencies" in s:
        return "dependency-conflict"

    if "unsupported python version" in s:
        return "python-version-constraint"

    if "requires-python" in s or "python_version" in s or "not supported for this python" in s:
        return "python-version-constraint"

    # Missing Python.h / dev headers when building C extensions (e.g., matplotlib)
    if "python.h: no such file or directory" in s:
        return "python-dev-headers-missing"

    # Missing system C libraries for lxml
    if "please make sure the libxml2 and libxslt development packages are installed" in s:
        return "c-lib-dev-missing-libxml2-libxslt"

    # Using removed configparser.SafeConfigParser API in package build scripts.
    if "configparser.safeconfigparser" in s:
        return "python-stdlib-compat-configparser"

    if "failed to build" in s or "building wheel" in s or "build_backend" in s or "error: command" in s:
        return "build-backend-failure"

    if "connectionerror" in s or "network is unreachable" in s or "timed out" in s or "timeout" in s:
        return "network-io-error"

    if "permission denied" in s or "permissionerror" in s:
        return "permission-error"

    return "other"


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
    # Collect all normalized messages that still fall into the "other" bucket,
    # so they can be manually inspected later.
    other_messages: list[str] = []

    for f in m5_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue

        for rec in data:
            compat = rec.get("compat_results") or {}
            ty = compat.get("ty") or {}
            errors = ty.get("errors") or []
            for err in errors:
                if not isinstance(err, dict):
                    continue
                if err.get("rule") != "ty-install-error":
                    continue
                raw_msg = str(err.get("message") or "")
                if not raw_msg:
                    continue
                norm = normalize_msg(raw_msg)
                label = classify_install_message(norm)
                entry = clusters[label]
                entry["count"] = int(entry["count"]) + 1  # type: ignore[assignment]
                examples: List[str] = entry["example_messages"]  # type: ignore[assignment]
                if len(examples) < args.max_examples and norm not in examples:
                    examples.append(norm)
                if label == "other":
                    other_messages.append(norm)

    # Sort clusters by count desc
    sorted_items = sorted(clusters.items(), key=lambda kv: -int(kv[1]["count"]))
    ordered: Dict[str, Dict[str, object]] = {}
    for label, payload in sorted_items:
        # Ensure example_messages is a list of strings
        payload["example_messages"] = list(payload.get("example_messages", []))  # type: ignore[assignment]
        ordered[label] = payload

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Clustered ty-install-error messages into {len(ordered)} groups -> {args.output_json}")

if __name__ == "__main__":
    main()

