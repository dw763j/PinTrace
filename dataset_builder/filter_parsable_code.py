"""Filter StackExchange records by Python parseability of code blocks.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m dataset_builder.filter_parsable_code --help
"""
from __future__ import annotations

import argparse
import ast
import codeop
import json
import re
import textwrap
import warnings
from pathlib import Path

from loguru import logger
from tqdm import tqdm

PROMPT_RE = re.compile(r"^\s*(>>>|\.\.\.)\s?", re.M)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter SO records by Python parseability of extracted code blocks."
    )
    parser.add_argument("--source-run-name", required=True)
    parser.add_argument(
        "--input-jsonl",
        default="records_import_postdate_ge2020.jsonl",
        help="Input file under dataset_runs/<run-name>/",
    )
    parser.add_argument(
        "--output-jsonl",
        default="records_import_postdate_ge2020_parsable.jsonl",
        help="Output file under dataset_runs/<run-name>/",
    )
    parser.add_argument(
        "--pair-pass-mode",
        choices=["any_block", "merged", "any_or_merged"],
        default="any_or_merged",
        help="How to decide parseability for a whole QA pair.",
    )
    parser.add_argument(
        "--min-code-chars",
        type=int,
        default=12,
        help="Ignore blocks shorter than this after normalization.",
    )
    return parser.parse_args()


def normalize_code(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = PROMPT_RE.sub("", text)
    text = textwrap.dedent(text).strip()
    return text


def classify_python_code(code: str) -> tuple[str, str | None]:
    if not code:
        return "empty", None
    # Some real-world snippets include invalid escape sequences in string
    # literals (e.g., '\s'), which emit SyntaxWarning to stderr. We silence
    # them here to keep filtering logs clean and deterministic.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            ast.parse(code)
            return "ast_ok", None
        except SyntaxError as e:
            # compile_command distinguishes incomplete input from hard syntax errors.
            try:
                c = codeop.compile_command(code, symbol="exec")
                if c is None:
                    return "incomplete", f"{e.msg} @ line {e.lineno}"
            except SyntaxError:
                pass
            return "syntax_error", f"{e.msg} @ line {e.lineno}"


def filter_parsable_records(
    *,
    run_dir: Path,
    input_jsonl: str,
    output_jsonl: str,
    pair_pass_mode: str,
    min_code_chars: int,
) -> dict:
    in_path = run_dir / input_jsonl
    out_path = run_dir / output_jsonl
    stats_path = run_dir / f"{Path(output_jsonl).stem}.stats.json"
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    total_records = 0
    kept_records = 0
    record_reasons: dict[str, int] = {}
    block_class_counts: dict[str, int] = {}

    with in_path.open("r", encoding="utf-8") as reader, out_path.open("w", encoding="utf-8") as writer:
        for line in tqdm(reader, desc="Filter parseable code pairs"):
            total_records += 1
            obj = json.loads(line)
            blocks = obj.get("code_blocks") or []

            normalized_blocks: list[str] = []
            block_classes: list[str] = []
            for block in blocks:
                norm = normalize_code((block or {}).get("content", ""))
                if len(norm) < min_code_chars:
                    continue
                cls, _ = classify_python_code(norm)
                block_class_counts[cls] = block_class_counts.get(cls, 0) + 1
                normalized_blocks.append(norm)
                block_classes.append(cls)

            merged = "\n\n".join(normalized_blocks).strip()
            merged_class, merged_err = classify_python_code(merged)

            any_block_ok = any(c == "ast_ok" for c in block_classes)
            merged_ok = merged_class == "ast_ok"
            if pair_pass_mode == "any_block":
                keep = any_block_ok
            elif pair_pass_mode == "merged":
                keep = merged_ok
            else:
                keep = any_block_ok or merged_ok

            reason = "kept" if keep else (
                "no_valid_blocks" if not normalized_blocks else f"pair_not_parseable:{merged_class}"
            )
            record_reasons[reason] = record_reasons.get(reason, 0) + 1

            if not keep:
                continue

            kept_records += 1
            obj["parsed_code"] = {
                "pair_pass_mode": pair_pass_mode,
                "normalized_block_count": len(normalized_blocks),
                "any_block_ast_ok": any_block_ok,
                "merged_ast_ok": merged_ok,
                "merged_class": merged_class,
                "merged_error": merged_err,
                "selected_code": merged if merged_ok else (normalized_blocks[0] if normalized_blocks else ""),
            }
            writer.write(json.dumps(obj, ensure_ascii=False) + "\n")

    stats = {
        "input_jsonl": str(in_path),
        "output_jsonl": str(out_path),
        "pair_pass_mode": pair_pass_mode,
        "min_code_chars": min_code_chars,
        "total_records": total_records,
        "kept_records": kept_records,
        "keep_rate": (kept_records / total_records) if total_records else 0.0,
        "record_reasons": record_reasons,
        "block_class_counts": block_class_counts,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    args = parse_args()
    run_dir = Path("dataset_runs") / args.source_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "post_filter_code_parse.log")
    stats = filter_parsable_records(
        run_dir=run_dir,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        pair_pass_mode=args.pair_pass_mode,
        min_code_chars=args.min_code_chars,
    )
    logger.success(f"Code parseability filtering completed: {stats}")


if __name__ == "__main__":
    main()

