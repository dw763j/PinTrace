"""End-to-end StackOverflow dataset build for D2-style experiments.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m dataset_builder.build_stackoverflow_dataset --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from paths import (
    ANSWER_TIME_INDEX_DB,
    GLOBAL_CACHE,
    OSV_INDEX,
    OUTPUTS,
    STACKEXCHANGE_INDEX_DB,
    TOP_PYPI_PACKAGES,
    ensure_dirs,
)

from .alias_mapping_converter import build_alias_mapping_for_targets
from .balanced_sampler import sample_balanced_records
from .package_pool import load_target_packages
from .filter_parsable_code import filter_parsable_records
from .filter_so_records import filter_records
from .stackexchange_builder import build_records_from_stackexchange_xml_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end StackExchange D2 pipeline.")
    parser.add_argument("--run-name", default="so_e2e")
    parser.add_argument("--posts-xml-path", required=True)

    parser.add_argument("--top-pypi-path", default=None, help="Default: paths.TOP_PYPI_PACKAGES")
    parser.add_argument("--osv-dir", default=None, help="Default: paths.OSV_INDEX")
    parser.add_argument("--mapping-path", default=None, help="Default: global_cache/mapping.json")
    parser.add_argument("--top-limit", type=int, default=15000)
    parser.add_argument("--target-packages-path", default=None)
    parser.add_argument("--alias-mapping-path", default=None)
    parser.add_argument(
        "--disable-auto-alias-from-mapping",
        action="store_true",
        help="Disable auto-conversion from --mapping-path when --alias-mapping-path is not provided.",
    )

    parser.add_argument("--match-mode", choices=["strict", "relaxed", "conservative", "balanced", "recall"], default="balanced")
    parser.add_argument("--pass2-workers", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--stackexchange-index-db", default=None, help="Default: paths.STACKEXCHANGE_INDEX_DB")

    parser.add_argument("--cutoff-date", default="2016-01-01T00:00:00")
    parser.add_argument("--date-filter-mode", choices=["exact", "id_threshold"], default="exact")
    parser.add_argument("--answer-time-index-db", default=None, help="Default: paths.ANSWER_TIME_INDEX_DB")
    parser.add_argument("--no-require-import-evidence", action="store_true")

    parser.add_argument("--pair-pass-mode", choices=["any_block", "merged", "any_or_merged"], default="any_or_merged")
    parser.add_argument("--min-code-chars", type=int, default=12)

    parser.add_argument("--balanced-total", type=int, default=5000)
    parser.add_argument("--balanced-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    args.mapping_path = args.mapping_path or str(GLOBAL_CACHE / "mapping.json")
    args.top_pypi_path = args.top_pypi_path or str(TOP_PYPI_PACKAGES)
    args.osv_dir = args.osv_dir or str(OSV_INDEX)
    run_dir = OUTPUTS / "dataset_builder" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    from paths import LOGS
    logger.add(LOGS / f"dataset_e2e_{args.run_name}.log")

    target_packages = load_target_packages(args, run_dir)
    logger.info(f"[E2E] target packages loaded: {len(target_packages)}")

    alias_mapping_path = args.alias_mapping_path
    alias_mapping_stats = None
    if not alias_mapping_path and not args.disable_auto_alias_from_mapping:
        alias_mapping_path = str(run_dir / "alias_mapping.auto.json")
        alias_mapping_stats = build_alias_mapping_for_targets(
            mapping_path=args.mapping_path,
            target_packages=target_packages,
            out_path=alias_mapping_path,
        )
        logger.info(f"[E2E] auto alias mapping generated: {alias_mapping_stats}")

    stage1 = build_records_from_stackexchange_xml_v2(
        posts_xml_path=args.posts_xml_path,
        target_packages=target_packages,
        alias_mapping_path=alias_mapping_path,
        out_jsonl_path=str(run_dir / "records.jsonl"),
        work_dir=str(run_dir / "stackexchange_work"),
        match_mode=args.match_mode,
        pass2_workers=max(1, args.pass2_workers),
        min_question_score=args.min_score,
        max_records=args.max_questions,
        sqlite_index_path=args.stackexchange_index_db or str(STACKEXCHANGE_INDEX_DB),
    )

    stage2 = filter_records(
        run_dir=run_dir,
        posts_xml=Path(args.posts_xml_path),
        cutoff_date=args.cutoff_date,
        require_import_evidence=not bool(args.no_require_import_evidence),
        date_filter_mode=args.date_filter_mode,
        answer_time_index_db=Path(args.answer_time_index_db) if args.answer_time_index_db else ANSWER_TIME_INDEX_DB,
    )

    stage2_output_name = Path(stage2["output_final"]).name
    parsable_output_name = f"{Path(stage2_output_name).stem}_parsable.jsonl"
    stage3 = filter_parsable_records(
        run_dir=run_dir,
        input_jsonl=stage2_output_name,
        output_jsonl=parsable_output_name,
        pair_pass_mode=args.pair_pass_mode,
        min_code_chars=args.min_code_chars,
    )

    balanced_output_name = f"{Path(parsable_output_name).stem}_balanced{args.balanced_total}_fair.jsonl"
    balanced_stats_name = f"{Path(parsable_output_name).stem}_balanced{args.balanced_total}_fair.stats.json"
    stage4 = sample_balanced_records(
        run_dir=run_dir,
        input_jsonl=parsable_output_name,
        output_jsonl=balanced_output_name,
        stats_json=balanced_stats_name,
        target_total=args.balanced_total,
        seed=args.balanced_seed,
    )

    summary = {
        "run_name": args.run_name,
        "posts_xml_path": args.posts_xml_path,
        "target_packages": len(target_packages),
        "alias_mapping_path": alias_mapping_path,
        "alias_mapping_stats": alias_mapping_stats,
        "stages": {
            "stage1_build_records": stage1,
            "stage2_filter_import_date": stage2,
            "stage3_filter_parsable": stage3,
            "stage4_balanced_sample": stage4,
        },
        "artifacts": {
            "records": str(run_dir / "records.jsonl"),
            "post_filtered": str(run_dir / stage2_output_name),
            "parsable": str(run_dir / parsable_output_name),
            "balanced": str(run_dir / balanced_output_name),
            "balanced_stats": str(run_dir / balanced_stats_name),
        },
    }
    (run_dir / "e2e_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"[E2E] completed: {summary['artifacts']}")


if __name__ == "__main__":
    main()
