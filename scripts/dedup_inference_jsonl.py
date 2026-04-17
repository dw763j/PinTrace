#!/usr/bin/env python3
"""
Check inference JSONL duplicates by ID and optionally remove them.

- D1 uses task_id
- D2 uses record_id

python -m scripts.dedup_inference_jsonl

Default behavior: list duplicate IDs only.
Use --delete to rewrite file after creating a backup in the same directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from paths import OUTPUTS


@dataclass
class FileReport:
    path: Path
    id_field: str
    total_lines: int
    valid_records: int
    skipped_lines: int
    duplicate_ids: list[str]
    redundant_records: int


def iter_candidate_jsonl_files(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix == ".jsonl" and root not in seen:
                paths.append(root)
                seen.add(root)
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and optionally deduplicate inference JSONL by task_id/record_id.")
    parser.add_argument("jsonl", nargs="?", type=Path, help="Optional path to a single JSONL file to inspect.")
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[OUTPUTS / "d1", OUTPUTS / "d2"],
        help="Root directories to scan when no single JSONL file is given.",
    )
    parser.add_argument(
        "--id-field",
        choices=["auto", "task_id", "record_id"],
        default="auto",
        help="ID field to deduplicate on. auto will infer from content.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete duplicated records in-place (keep first occurrence). A backup will be created first.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix for backup file. Final backup path is <input_name><suffix>.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, fail on malformed JSON/non-object/missing ID lines. Otherwise skip malformed lines.",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=50,
        help="Maximum number of duplicate IDs to print in detail.",
    )
    return parser.parse_args()


def detect_id_field(lines: list[str]) -> str:
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "task_id" in obj:
            return "task_id"
        if "record_id" in obj:
            return "record_id"
    raise ValueError("Cannot infer id field from file. Please set --id-field task_id or --id-field record_id.")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)


def get_id_value(obj: dict, id_field: str) -> str | None:
    value = obj.get(id_field)
    if value is None:
        return None
    return str(value)


def inspect_file(jsonl_path: Path, id_field_arg: str, strict: bool, max_show: int) -> tuple[FileReport, list[str], list[str]]:
    lines = read_lines(jsonl_path)
    id_field = id_field_arg if id_field_arg != "auto" else detect_id_field(lines)

    id_counter: Counter[str] = Counter()
    valid_count = 0
    skipped_count = 0
    bad_line_indexes: list[int] = []

    for idx, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            if strict:
                raise ValueError(f"Line {idx} in {jsonl_path} is not valid JSON.")
            skipped_count += 1
            bad_line_indexes.append(idx)
            continue
        if not isinstance(obj, dict):
            if strict:
                raise ValueError(f"Line {idx} in {jsonl_path} JSON is not an object.")
            skipped_count += 1
            bad_line_indexes.append(idx)
            continue
        id_value = get_id_value(obj, id_field)
        if id_value is None:
            if strict:
                raise ValueError(f"Line {idx} in {jsonl_path} is missing required field: {id_field}")
            skipped_count += 1
            bad_line_indexes.append(idx)
            continue
        id_counter[id_value] += 1
        valid_count += 1

    duplicate_ids = sorted([id_value for id_value, count in id_counter.items() if count > 1])
    duplicate_total = sum(id_counter[id_value] - 1 for id_value in duplicate_ids)

    report = FileReport(
        path=jsonl_path,
        id_field=id_field,
        total_lines=len(lines),
        valid_records=valid_count,
        skipped_lines=skipped_count,
        duplicate_ids=duplicate_ids,
        redundant_records=duplicate_total,
    )
    return report, [f"  {id_value}\tcount={id_counter[id_value]}" for id_value in duplicate_ids[:max_show]], [str(i) for i in bad_line_indexes]


def deduplicate_file(jsonl_path: Path, id_field: str, strict: bool, backup_suffix: str) -> tuple[Path, int, int]:
    lines = read_lines(jsonl_path)
    backup_path = jsonl_path.with_name(jsonl_path.name + backup_suffix)
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")

    shutil.copy2(jsonl_path, backup_path)

    seen: set[str] = set()
    kept_lines: list[str] = []
    removed = 0

    for raw in lines:
        text = raw.strip()
        if not text:
            kept_lines.append(raw)
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            if strict:
                raise ValueError(f"Encountered malformed JSON in {jsonl_path} during rewrite in strict mode.")
            kept_lines.append(raw)
            continue
        if not isinstance(obj, dict):
            if strict:
                raise ValueError(f"Encountered non-object JSON in {jsonl_path} during rewrite in strict mode.")
            kept_lines.append(raw)
            continue
        id_value = get_id_value(obj, id_field)
        if id_value is None:
            if strict:
                raise ValueError(f"Encountered record missing {id_field} in {jsonl_path} during rewrite in strict mode.")
            kept_lines.append(raw)
            continue

        if id_value in seen:
            removed += 1
            continue

        seen.add(id_value)
        kept_lines.append(raw)

    jsonl_path.write_text("".join(kept_lines), encoding="utf-8")
    return backup_path, removed, len(seen)


def main() -> None:
    args = parse_args()
    if args.jsonl is not None:
        targets = [args.jsonl.resolve()]
    else:
        targets = iter_candidate_jsonl_files([root.resolve() for root in args.roots])

    if not targets:
        print("No JSONL files found.")
        return

    reports: list[FileReport] = []
    detailed_lines_by_file: dict[Path, list[str]] = {}

    for jsonl_path in targets:
        if not jsonl_path.is_file():
            continue
        report, duplicate_lines, _bad_line_indexes = inspect_file(jsonl_path, args.id_field, args.strict, args.max_show)
        reports.append(report)
        detailed_lines_by_file[jsonl_path] = duplicate_lines

    total_files = len(reports)
    total_duplicates = sum(len(report.duplicate_ids) for report in reports)
    total_redundant = sum(report.redundant_records for report in reports)
    total_records = sum(report.valid_records for report in reports)

    print(f"Scanned roots: {', '.join(str(Path(r).resolve()) for r in (args.roots if args.jsonl is None else [])) or '(single file mode)'}")
    print(f"Files scanned: {total_files}")
    print(f"Records scanned: {total_records}")

    duplicate_reports = [report for report in sorted(reports, key=lambda r: str(r.path)) if report.duplicate_ids]
    if duplicate_reports:
        print(f"Files with duplicates: {len(duplicate_reports)}")
        print(f"Duplicate IDs total: {total_duplicates}")
        print(f"Redundant records total: {total_redundant}")
        for report in duplicate_reports:
            print(f"\nFile: {report.path}")
            print(f"ID field: {report.id_field}")
            print(f"Duplicate IDs: {len(report.duplicate_ids)}")
            print(f"Redundant records (can remove): {report.redundant_records}")
            print("Duplicate ID details:")
            for detail_line in detailed_lines_by_file[report.path]:
                print(detail_line)
            if len(report.duplicate_ids) > args.max_show:
                print(f"  ... ({len(report.duplicate_ids) - args.max_show} more)")
        if len(duplicate_reports) < total_files:
            print("\nNo duplicates in the remaining files.")
    else:
        print("No duplicates in any scanned file.")

    if not args.delete:
        return

    deletable_reports = [report for report in reports if report.duplicate_ids]
    if not deletable_reports:
        print("\nNo duplicates found. Nothing to delete.")
        return

    print("\nDelete mode is enabled. Backups will be created before rewrite.")
    for report in deletable_reports:
        backup_path, removed, remaining = deduplicate_file(report.path, report.id_field, args.strict, args.backup_suffix)
        print(f"{report.path}: backup={backup_path} removed={removed} remaining={remaining}")


if __name__ == "__main__":
    main()
