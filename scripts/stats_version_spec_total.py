#!/usr/bin/env python3
"""Sum version-specification counts (``total_lib_usages``) for a stage/mode.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m scripts.stats_version_spec_total --stage d1 --prompt-mode inline_no_vuln --python-tag py312

Totals every matched ``metrics_summary.json`` under the chosen prompt mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paths import OUTPUTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sum total_lib_usages from metrics_summary.json files.")
    parser.add_argument("--stage", choices=("d1", "d2"), required=True, help="Track: d1 or d2")
    parser.add_argument(
        "--prompt-mode",
        required=True,
        help="Prompt-mode directory name, e.g. inline_no_vuln / inline / requirements.txt",
    )
    parser.add_argument(
        "--python-tag",
        default="py312",
        help="Python tag directory (default py312; use all to aggregate every py*)",
    )
    parser.add_argument("--show-files", action="store_true", help="Print each metrics file and its total")
    return parser.parse_args()


def collect_metrics_files(stage_dir: Path, python_tag: str) -> list[Path]:
    if python_tag == "all":
        return sorted(stage_dir.glob("*/py*/metrics_summary.json"))
    return sorted(stage_dir.glob(f"*/{python_tag}/metrics_summary.json"))


def extract_version_spec_total(payload: dict) -> int | None:
    direct_value = payload.get("total_lib_usages")
    if isinstance(direct_value, (int, float)):
        return int(direct_value)
    version_results = payload.get("version_results")
    if isinstance(version_results, dict):
        specified_value = version_results.get("specified_lib")
        if isinstance(specified_value, (int, float)):
            return int(specified_value)
    return None


def main() -> None:
    args = parse_args()
    target_dir = OUTPUTS / args.stage / args.prompt_mode
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {target_dir}")

    metric_files = collect_metrics_files(target_dir, args.python_tag)
    if not metric_files:
        raise FileNotFoundError(f"No metrics_summary.json under: {target_dir}")

    total = 0
    used = 0
    skipped: list[dict[str, str]] = []

    for metrics_path in metric_files:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = extract_version_spec_total(payload)
        if value is None:
            skipped.append({"file": str(metrics_path), "reason": "missing_or_non_numeric_total_lib_usages_or_version_results.specified_lib"})
            continue
        total += value
        used += 1
        if args.show_files:
            print(f"{metrics_path}: {value}")

    summary = {
        "stage": args.stage,
        "prompt_mode": args.prompt_mode,
        "python_tag": args.python_tag,
        "metrics_files_found": len(metric_files),
        "metrics_files_used": used,
        "metrics_files_skipped": len(skipped),
        "total_version_specifications": total,
    }
    if skipped:
        summary["skipped_examples"] = skipped[:10]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()