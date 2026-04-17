#!/usr/bin/env python3
# Usage (repo root): python -m scripts.scan_bcb_test_utilized_tpl --help
"""
Scan BigCodeBench tests for third-party imports outside stdlib and specified TPL.

Output:
- JSONL where each line is a task with extra test-only deps:
  {"task_id": "...", "specified_tpl": [...], "test_utilized_tpl": [...], "test_imports": [...]}
"""

from __future__ import annotations

import argparse
import json
# import sys
from pathlib import Path
from typing import Any

from paths import BIGCODEBENCH, OUTPUTS
from stages.compact_checker import (
    _extract_imported_top_modules,
    _extract_pkg_name_from_spec,
    _infer_test_utilized_tpl,
)

# sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT_PATH = OUTPUTS / "d1" / "bcb_test_utilized_tpl.jsonl"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan BigCodeBench tests for extra deps.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to scan (0 means no limit).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not BIGCODEBENCH.exists():
        raise FileNotFoundError(f"BigCodeBench jsonl not found: {BIGCODEBENCH}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("w", encoding="utf-8")
    matched = 0
    scanned = 0

    with BIGCODEBENCH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scanned += 1
            if args.limit and scanned > args.limit:
                break

            task_id = str(row.get("task_id") or "")
            test_code = str(row.get("test") or "")
            libs_field: Any = row.get("libs")
            if libs_field is None:
                specified_tpl = []
            elif isinstance(libs_field, list):
                specified_tpl = [str(item) for item in libs_field]
            elif isinstance(libs_field, str):
                libs_text = libs_field.strip()
                try:
                    # safe-ish: this dataset stores libs as python literal strings
                    import ast

                    parsed_libs = ast.literal_eval(libs_text)
                    specified_tpl = [str(item) for item in parsed_libs] if isinstance(parsed_libs, list) else []
                except Exception:
                    specified_tpl = []
            else:
                specified_tpl = []
            pkg_specs = [str(x) for x in specified_tpl]

            extra = _infer_test_utilized_tpl(test_code, pkg_specs=pkg_specs)
            if not extra:
                continue

            payload = {
                "task_id": task_id,
                "specified_tpl": sorted({_extract_pkg_name_from_spec(x) for x in pkg_specs if _extract_pkg_name_from_spec(x)}),
                "test_imports": sorted(_extract_imported_top_modules(test_code)),
                "test_utilized_tpl": extra,
            }
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            matched += 1

    out_f.close()
    print(f"scanned={scanned} matched={matched} output={args.output}")


if __name__ == "__main__":
    main()

