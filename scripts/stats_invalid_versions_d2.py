#!/usr/bin/env python3
"""Count invalid D2 versions per model in inline mode.

Invalid means:
- ``version_exists`` is false (version not on PyPI)
- non-empty ``version_status`` (yanked, deprecated, etc.)

Prints:
- per-model invalid counts and rates
- global breakdown by invalid kind (non-existent vs yanked/deprecated, etc.)

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m scripts.stats_invalid_versions_d2 --help
    python -m scripts.stats_invalid_versions_d2 --py-tag py312
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from paths import OUTPUTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count invalid D2 versions in inline mode.")
    parser.add_argument("--d2-inline-root", type=Path, default=OUTPUTS / "d2" / "inline", help="D2 inline root directory")
    parser.add_argument("--py-tag", default="py312", help="Python tag directory (py312, py38, …)")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary output path")
    return parser.parse_args()


def collect_invalid_versions(
    d2_inline_root: Path,
    py_tag: str,
) -> dict[str, dict]:
    """
    Walk all models' ``m3_resolved_records.json`` and aggregate invalid versions.
    """
    pattern = f"{py_tag}/m3_resolved_records.json"
    files = sorted(d2_inline_root.glob(f"*/{pattern}"))

    if not files:
        raise FileNotFoundError(
            f"No m3_resolved_records.json found under {d2_inline_root} "
            f"matching pattern '*/{pattern}'"
        )

    models_stats: dict[str, dict] = {}
    global_invalid_counter: Counter = Counter()
    global_total_libs = 0
    global_total_invalid = 0

    for file_path in files:
        model_dir = file_path.parent.parent.name
        records: list[dict] = json.loads(file_path.read_text(encoding="utf-8"))

        total_libs = 0
        invalid_libs = 0
        invalid_types: Counter = Counter()

        for record in records:
            for lib in record.get("per_lib", []):
                total_libs += 1
                version_exists = lib.get("version_exists", True)
                version_status = lib.get("version_status")

                if not version_exists:
                    invalid_libs += 1
                    invalid_types["non_existent"] += 1
                elif version_status:
                    invalid_libs += 1
                    invalid_types[str(version_status)] += 1

        global_total_libs += total_libs
        global_total_invalid += invalid_libs
        for k, v in invalid_types.items():
            global_invalid_counter[k] += v

        rate = (invalid_libs / total_libs) if total_libs else 0.0
        models_stats[model_dir] = {
            "path": str(file_path),
            "total_libs": total_libs,
            "invalid_libs": invalid_libs,
            "invalid_rate": rate,
            "invalid_by_type": dict(invalid_types),
        }

    global_rate = (global_total_invalid / global_total_libs) if global_total_libs else 0.0

    return {
        "models": dict(sorted(models_stats.items())),
        "summary": {
            "total_models": len(models_stats),
            "total_libs": global_total_libs,
            "total_invalid": global_total_invalid,
            "invalid_rate": global_rate,
            "invalid_by_type": dict(global_invalid_counter.most_common()),
        },
    }


def main() -> None:
    args = parse_args()
    stats = collect_invalid_versions(args.d2_inline_root, args.py_tag)

    print(f"Scanned: {args.d2_inline_root} (pattern */{args.py_tag}/m3_resolved_records.json)")
    print("\n=== Per-model ===")
    for model, m in stats["models"].items():
        print(
            f"{model}: invalid={m['invalid_libs']}/{m['total_libs']} "
            f"({m['invalid_rate']*100:.2f}%)  breakdown={m['invalid_by_type']}"
        )

    s = stats["summary"]
    print("\n=== Global ===")
    print(
        f"models={s['total_models']}  libs={s['total_libs']}  "
        f"invalid={s['total_invalid']} ({s['invalid_rate']*100:.4f}%)"
    )
    print("invalid_by_type:", s["invalid_by_type"])

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
