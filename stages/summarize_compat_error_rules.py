#!/usr/bin/env python3
"""
Summarize compat_results.ty error rules across all models and runs.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m stages.summarize_compat_error_rules --help

Scans outputs/d1/**/m5_compat_records.json and counts how many times each
`compat_results.ty.errors[*].rule` appears, grouped by:
- model_name
- prompt_mode (inline / inline_no_vuln / requirements.txt)

Outputs:
- A JSON file with a nested structure:
  {
    "<model>": {
      "<prompt_mode>": {
        "total_error_entries": int,
        "rule_counts": { "<rule>": int, ... },
        "total_tasks": int,
        "passed_ty_check": int
      },
      ...
    },
    ...
  }
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_D1 = PROJECT_ROOT / "outputs" / "d1"

# Ensure project root is on sys.path so we can import sibling scripts as a module.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RuleSummary:
    total_error_entries: int
    rule_counts: Dict[str, int]
    ty_install_error_clusters: Dict[str, int] | None = None
    total_tasks: int = 0
    passed_ty_check: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize compat_results.ty error rules over all m5_compat_records.json files."
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
        default=OUTPUTS_D1 / "compat_error_rule_summary.json",
        help="Where to write the JSON summary.",
    )
    p.add_argument(
        "--python-version",
        type=str,
        default=None,
        help=(
            "Python version whose compat results to summarize, e.g. 3.12. "
            "When set, only py<ver>/m5_compat_records.json files are scanned. "
            "When unset, all m5_compat_records.json files are scanned (legacy behavior)."
        ),
    )
    return p.parse_args()


def infer_mode_and_model(path: Path) -> tuple[str, str]:
    """
    Infer prompt_mode (top-level subdir under outputs/d1) and model_key from path.

    Example:
      outputs/d1/inline_no_vuln/claude-sonnet-4-6_inline_no_vuln/m5_compat_records.json
      -> ("inline_no_vuln", "claude-sonnet-4-6")
    """
    # path: .../outputs/d1/<mode>/<run_name>/m5_compat_records.json
    parts = path.parts
    try:
        idx = parts.index("d1")
    except ValueError:
        return "unknown", "unknown"
    mode = parts[idx + 1] if len(parts) > idx + 1 else "unknown"
    run_name = parts[idx + 2] if len(parts) > idx + 2 else "unknown"

    # Heuristic: strip common suffixes from run_name to recover api_model.
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


def summarize_file(
    path: Path,
    agg_rules: dict,
    agg_install_clusters: dict,
    agg_total_tasks: dict,
    agg_passed_ty: dict,
) -> None:
    """Update global aggregation dicts with counts from one m5_compat_records.json file."""
    mode, model = infer_mode_and_model(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return

    key = (model, mode)
    if key not in agg_rules:
        agg_rules[key] = Counter()
    if key not in agg_install_clusters:
        agg_install_clusters[key] = Counter()
    agg_total_tasks[key] = agg_total_tasks.get(key, 0) + len(data)

    for rec in data:
        compat = rec.get("compat_results") or {}
        ty = compat.get("ty") or {}
        errors = ty.get("errors") or []
        # Only count as passed when ty result exists (type check was run) and no errors
        if compat.get("ty") and len(errors) == 0:
            agg_passed_ty[key] = agg_passed_ty.get(key, 0) + 1
        for err in errors:
            if not isinstance(err, dict):
                continue
            rule = str(err.get("rule") or "UNSPECIFIED")
            agg_rules[key][rule] += 1
            if rule == "ty-install-error":
                raw_msg = str(err.get("message") or "")
                if raw_msg:
                    # Import clustering helpers lazily to keep module-level imports tidy.
                    from stages.cluster_ty_install_errors import (
                        normalize_msg as cluster_normalize_msg,
                        classify_install_message as cluster_classify_install_message,
                    )

                    norm = cluster_normalize_msg(raw_msg)
                    label = cluster_classify_install_message(norm)
                    agg_install_clusters[key][label] += 1


def main() -> None:
    args = parse_args()
    outputs_d1: Path = args.outputs_d1

    if not outputs_d1.is_dir():
        raise SystemExit(f"outputs/d1 not found: {outputs_d1}")

    # When a specific python_version is requested, only look under py<ver>/.
    if args.python_version:
        py_tag = f"py{args.python_version.replace('.', '')}"
        m5_files = sorted(outputs_d1.glob(f"**/{py_tag}/m5_compat_records.json"))
        # Fallback to legacy layout if nothing found (for backward compatibility).
        if not m5_files:
            m5_files = sorted(outputs_d1.glob("**/m5_compat_records.json"))
    else:
        m5_files = sorted(outputs_d1.glob("**/m5_compat_records.json"))
    if not m5_files:
        raise SystemExit(f"No m5_compat_records.json found under {outputs_d1}")

    # (model, mode) -> Counter(rule -> count)
    agg_rules: dict[tuple[str, str], Counter] = {}
    # (model, mode) -> Counter(install_cluster_label -> count) for ty-install-error only
    agg_install_clusters: dict[tuple[str, str], Counter] = {}
    # (model, mode) -> total task count
    agg_total_tasks: dict[tuple[str, str], int] = {}
    # (model, mode) -> count of tasks that passed ty check (no errors)
    agg_passed_ty: dict[tuple[str, str], int] = {}
    for f in m5_files:
        summarize_file(f, agg_rules, agg_install_clusters, agg_total_tasks, agg_passed_ty)

    # Convert to nested dict: model -> mode -> RuleSummary
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model, mode), counter in agg_rules.items():
        total_entries = sum(counter.values())
        clusters = agg_install_clusters.get((model, mode))
        summary = RuleSummary(
            total_error_entries=total_entries,
            rule_counts=dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))),
            ty_install_error_clusters=(
                dict(sorted(clusters.items(), key=lambda kv: (-kv[1], kv[0]))) if clusters else None
            ),
            total_tasks=agg_total_tasks.get((model, mode), 0),
            passed_ty_check=agg_passed_ty.get((model, mode), 0),
        )
        out[model][mode] = asdict(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote summary for {len(agg_rules)} (model, mode) pairs to {args.output_json}")


if __name__ == "__main__":
    main()

