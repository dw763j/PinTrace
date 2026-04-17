#!/usr/bin/env python3
"""
D1 (BigCodeBench): list tasks whose third-party TPLs appear in the OSV version matrix.

Rule: parse ``libs``, drop the standard library, and flag a task if any third-party name (lower case)
exists as a key under ``global_cache/osv_version_matrix.json`` → ``packages``.

Data source defaults to the Hugging Face ``bigcode/bigcodebench`` ``v0.1.4`` split; pass ``--bcb-jsonl``
to use a local JSONL file (same schema as ``paths.BIGCODEBENCH``).

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m scripts.stats_d1_bcb_osv_tasks --help
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from paths import BIGCODEBENCH, GLOBAL_CACHE, OUTPUTS


OSV_MATRIX_PATH = GLOBAL_CACHE / "osv_version_matrix.json"
OUTPUT_JSONL_PATH = OUTPUTS / "d1" / "bcb_tasks_tpl_in_osv_matrix.jsonl"
OUTPUT_SUMMARY_PATH = OUTPUTS / "d1" / "bcb_tasks_tpl_in_osv_matrix.summary.json"

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[misc, assignment]

try:
    from stdlib_list import stdlib_list
except ImportError:
    stdlib_list = None  # type: ignore[misc, assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List BCB tasks whose TPLs appear in OSV matrix.")
    parser.add_argument(
        "--bcb-jsonl",
        type=Path,
        default=None,
        help="Local BCB JSONL; defaults to paths.BIGCODEBENCH when present, otherwise Hugging Face",
    )
    parser.add_argument("--python-stdlib-tag", default="3.12", help="stdlib_list tag (default 3.12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if stdlib_list is None:
        raise RuntimeError("Install stdlib-list (see requirements.txt)")
    if not OSV_MATRIX_PATH.exists():
        raise FileNotFoundError(OSV_MATRIX_PATH)

    matrix_json = json.loads(OSV_MATRIX_PATH.read_text(encoding="utf-8"))
    package_object = matrix_json.get("packages")
    if not isinstance(package_object, dict):
        raise ValueError("osv_version_matrix.json: missing packages object")
    osv_names = {str(package_name).lower() for package_name in package_object.keys()}
    stdlib = set(stdlib_list(args.python_stdlib_tag))

    flagged: list[dict[str, Any]] = []
    total = 0
    with_third_party = 0

    bcb_path = args.bcb_jsonl
    if bcb_path is None and BIGCODEBENCH.exists():
        bcb_path = BIGCODEBENCH

    if bcb_path is not None:
        if not bcb_path.exists():
            raise FileNotFoundError(bcb_path)
        with bcb_path.open("r", encoding="utf-8") as file_obj:
            samples = [json.loads(stripped) for raw_line in file_obj if (stripped := raw_line.strip())]
    else:
        if load_dataset is None:
            raise RuntimeError("Provide a local BCB JSONL (paths.BIGCODEBENCH) or install datasets for Hugging Face")
        ds = load_dataset("bigcode/bigcodebench")
        split = ds["v0.1.4"]
        samples = split

    for sample in samples:
        total += 1

        raw_libs: Any = sample.get("libs")
        if raw_libs is None:
            parsed_libs: list[str] = []
        elif isinstance(raw_libs, (list, tuple, set)):
            parsed_libs = [str(item) for item in raw_libs]
        else:
            try:
                literal_value = ast.literal_eval(str(raw_libs))
            except (ValueError, SyntaxError):
                literal_value = None
            if isinstance(literal_value, dict):
                parsed_libs = [str(key) for key in literal_value.keys()]
            elif isinstance(literal_value, (list, tuple, set)):
                parsed_libs = [str(item) for item in literal_value]
            else:
                parsed_libs = []

        libs = [lib.strip() for lib in parsed_libs if lib and str(lib).strip()]
        third_party = sorted({lib for lib in libs if lib not in stdlib})
        if not third_party:
            continue
        with_third_party += 1
        tp_lower = [t.lower() for t in third_party]
        matched = sorted({t for t, tl in zip(third_party, tp_lower, strict=True) if tl in osv_names})
        if not matched:
            continue
        flagged.append(
            {
                "task_id": sample.get("task_id"),
                "third_party_libs": third_party,
                "osv_matrix_matched_libs": matched,
            }
        )

    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as out:
        for row in flagged:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "osv_matrix": str(OSV_MATRIX_PATH.resolve()),
        "bcb_source": str(bcb_path.resolve()) if bcb_path else "huggingface:bigcode/bigcodebench[v0.1.4]",
        "stdlib_tag": args.python_stdlib_tag,
        "total_tasks": total,
        "tasks_with_third_party_libs": with_third_party,
        "tasks_flagged_tpl_in_osv_matrix": len(flagged),
        "output_jsonl": str(OUTPUT_JSONL_PATH.resolve()),
    }
    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
