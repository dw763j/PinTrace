"""D2: invalid-version reason distribution per model (Inline / requirements.txt).

Reads pipeline artifacts ``m3_resolved_records.json`` (same rules as
``scripts/stats_invalid_versions_d2.py``):

- ``version_exists`` is false → reason ``non_existent`` (version not on PyPI)
- non-empty ``version_status`` → reason is that string (yanked / deprecated, etc.)
- else if ``resolution_method == "pypi_not_found"`` → ``pypi_not_found``

Outputs:

- Terminal: per-mode summary, per-model counts and shares by reason
- ``plots/d2_invalid_version_reasons.json``: structured summary (both modes)

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m plots.d2_invalid_version_reasons
    python -m plots.d2_invalid_version_reasons --py-tag py312 --json plots/d2_invalid_version_reasons.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D2_ROOT = PROJECT_ROOT / "outputs" / "d2"
DEFAULT_JSON_OUT = Path(__file__).resolve().parent / "d2_invalid_version_reasons.json"

# Human-readable reason labels (aligned with fields; unknown keys pass through)
REASON_DESCRIPTIONS: dict[str, str] = {
    "non_existent": "specified_version not on PyPI (version_exists=false)",
    "pypi_not_found": "package not resolved on PyPI (resolution_method=pypi_not_found)",
}


def run_dir_to_model_key(run_dir_name: str) -> str:
    """d2_kimi-k2.5_blind_inline -> kimi-k2.5"""
    if run_dir_name.startswith("d2_") and "_blind_" in run_dir_name:
        return run_dir_name[3:].split("_blind_", 1)[0]
    return run_dir_name


def classify_invalid_reason(lib_item: dict[str, Any]) -> str | None:
    version_exists = lib_item.get("version_exists", True)
    version_status = lib_item.get("version_status")
    resolution_method = lib_item.get("resolution_method")
    if not version_exists:
        return "non_existent"
    if version_status:
        return str(version_status)
    if resolution_method == "pypi_not_found":
        return "pypi_not_found"
    return None


def collect_invalid_reasons_for_root(mode_root: Path, py_tag: str) -> dict[str, Any]:
    pattern = f"*/{py_tag}/m3_resolved_records.json"
    files = sorted(mode_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No m3_resolved_records.json under {mode_root} matching pattern {pattern!r}"
        )

    models: dict[str, dict[str, Any]] = {}
    global_reasons: Counter[str] = Counter()
    global_total_libs = 0
    global_invalid = 0

    for file_path in files:
        run_dir_name = file_path.parent.parent.name
        model_key = run_dir_to_model_key(run_dir_name)
        records: list[dict[str, Any]] = json.loads(file_path.read_text(encoding="utf-8"))

        total_libs = 0
        invalid_count = 0
        reasons: Counter[str] = Counter()

        for record in records:
            for lib_item in record.get("per_lib", []):
                total_libs += 1
                reason = classify_invalid_reason(lib_item)
                if reason is not None:
                    invalid_count += 1
                    reasons[reason] += 1
                    global_reasons[reason] += 1
                    global_invalid += 1
                global_total_libs += 1

        rate = invalid_count / total_libs if total_libs else 0.0
        models[run_dir_name] = {
            "model_key": model_key,
            "run_dir": run_dir_name,
            "total_libs": total_libs,
            "invalid_count": invalid_count,
            "invalid_rate": round(rate, 6),
            "invalid_by_reason": dict(sorted(reasons.items(), key=lambda x: (-x[1], x[0]))),
        }

    global_rate = global_invalid / global_total_libs if global_total_libs else 0.0
    return {
        "mode_root": str(mode_root.resolve()),
        "py_tag": py_tag,
        "models": dict(sorted(models.items(), key=lambda x: x[0])),
        "summary": {
            "total_models": len(models),
            "total_libs": global_total_libs,
            "total_invalid": global_invalid,
            "invalid_rate": round(global_rate, 6),
            "invalid_by_reason": dict(sorted(global_reasons.items(), key=lambda x: (-x[1], x[0]))),
        },
    }


def build_report(py_tag: str) -> dict[str, Any]:
    inline_root = D2_ROOT / "inline"
    req_root = D2_ROOT / "requirements.txt"
    return {
        "py_tag": py_tag,
        "reason_descriptions": {
            **REASON_DESCRIPTIONS,
            "_note": "version_status reasons use the raw JSON string as key; see PyPI release metadata.",
        },
        "modes": {
            "inline": collect_invalid_reasons_for_root(inline_root, py_tag),
            "requirements.txt": collect_invalid_reasons_for_root(req_root, py_tag),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    py_tag = report["py_tag"]
    print(f"D2 invalid-version reasons  py_tag={py_tag}\n")

    for mode_name, block in report["modes"].items():
        summ = block["summary"]
        print(f"=== mode: {mode_name} ===")
        print(f"  models: {summ['total_models']}")
        print(f"  per_lib rows: {summ['total_libs']}")
        print(f"  invalid rows: {summ['total_invalid']}  ({summ['invalid_rate'] * 100:.4f}%)")
        print("  by reason (global):")
        for reason, cnt in summ["invalid_by_reason"].items():
            pct = cnt / summ["total_invalid"] * 100 if summ["total_invalid"] else 0.0
            desc = REASON_DESCRIPTIONS.get(reason, "")
            extra = f" — {desc}" if desc else ""
            print(f"    {reason}: {cnt}  ({pct:.2f}%){extra}")
        print()

    # Align by model_key across modes
    inline_models = report["modes"]["inline"]["models"]
    req_models = report["modes"]["requirements.txt"]["models"]
    key_to_inline: dict[str, str] = {m["model_key"]: name for name, m in inline_models.items()}
    key_to_req: dict[str, str] = {m["model_key"]: name for name, m in req_models.items()}
    all_keys = sorted(set(key_to_inline) | set(key_to_req))

    print("=== per model: invalid counts and reason split (Inline | requirements.txt) ===")
    for mk in all_keys:
        i_name = key_to_inline.get(mk)
        r_name = key_to_req.get(mk)
        i_m = inline_models.get(i_name) if i_name else None
        r_m = req_models.get(r_name) if r_name else None
        i_cnt = i_m["invalid_count"] if i_m else 0
        r_cnt = r_m["invalid_count"] if r_m else 0
        i_r = i_m["invalid_by_reason"] if i_m else {}
        r_r = r_m["invalid_by_reason"] if r_m else {}
        all_reasons = sorted(set(i_r) | set(r_r))
        print(f"\n  [{mk}]  invalid: inline={i_cnt}  req.txt={r_cnt}")
        for reason in all_reasons:
            a = i_r.get(reason, 0)
            b = r_r.get(reason, 0)
            if a or b:
                print(f"    {reason}: {a} | {b}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize invalid-version reasons for D2 Inline / requirements.txt.")
    p.add_argument("--py-tag", default="py312", help="Python tag subdir, e.g. py312, py38")
    p.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help=f"JSON output path (default: {DEFAULT_JSON_OUT})",
    )
    p.add_argument("--no-print", action="store_true", help="Write JSON only; skip table printout")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.py_tag)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {args.json.resolve()}\n")
    if not args.no_print:
        print_report(report)


if __name__ == "__main__":
    main()
