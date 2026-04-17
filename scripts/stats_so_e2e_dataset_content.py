#!/usr/bin/env python3
# Usage (repository root; optional: source .venv/bin/activate):
#   python -m scripts.stats_so_e2e_dataset_content --latex-temporal-only
#   python -m scripts.stats_so_e2e_dataset_content --print-latex-rows --print-latex-temporal
#   python -m scripts.stats_so_e2e_dataset_content --jsonl path/to/records.jsonl --output-json plots/so_e2e_stats.json
#   python -m scripts.stats_d2_so_e2e_jsonl  # legacy alias; emits the old stats_d2 JSON shape
"""
Stack Overflow E2E balanced-set statistics (``tab:dataset_content`` + ``tab:dataset_temporal``).

- **Question tokens**: ``question_title`` + ``strip_html(question_text)`` (no ``build_prompt`` suffix).
- **Accepted answer code**: ``parsed_code.selected_code`` line/token counts.
- **TPL coverage**: uses ``target_packages`` consistently (same spirit as ``stats_d2_so_e2e_jsonl``'s
  ``distinct_tpl_lower``): per-task multiset stats, ``≥2`` / ``≥3`` package thresholds, global distinct count.
- **Temporal split**: bucket by answer ``CreationDate`` year (via ``answer_time_index.sqlite3``), cumulative
  percentages, distinct ``target_packages`` per year, and fixed knowledge-regime labels for the paper table.

Tokenization uses ``tiktoken`` (default ``o200k_base``; override with ``--tiktoken-model``). See https://github.com/openai/tiktoken
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate.d2.prompt_builder import strip_html  # noqa: E402
from paths import ANSWER_TIME_INDEX_DB, OUTPUTS  # noqa: E402

DEFAULT_JSONL = (
    OUTPUTS
    / "dataset_builder"
    / "so_e2e_balanced"
    / "records_import_postdate_ge20200101_parsable_balanced1000_fair.jsonl"
)

# Matches the paper's ``tab:dataset_temporal`` copy (edit freely if needed)
_KNOWLEDGE_REGIME_BY_YEAR: dict[str, str] = {
    "2020": "Pre-LLM adoption",
    "2021": "Pre-LLM adoption",
    "2022": "Active LLM deployment",
    "2023": "Active LLM deployment",
    "2024": "Near-cutoff",
    "2025": "Post-cutoff (most models)",
}


def load_records_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _target_packages_normalized(record: dict[str, Any]) -> set[str]:
    return {
        str(p).strip().lower()
        for p in (record.get("target_packages") or [])
        if str(p).strip()
    }


def compute_target_package_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    tpl_counter: Counter[str] = Counter()
    n_with = 0
    per_task_counts: list[int] = []
    n_ge2 = 0
    n_ge3 = 0
    for rec in records:
        pkgs = _target_packages_normalized(rec)
        if pkgs:
            n_with += 1
        k = len(pkgs)
        per_task_counts.append(k)
        for p in pkgs:
            tpl_counter[p] += 1
        if k >= 2:
            n_ge2 += 1
        if k >= 3:
            n_ge3 += 1
    n = len(records)
    pct2 = 100.0 * n_ge2 / n if n else 0.0
    pct3 = 100.0 * n_ge3 / n if n else 0.0
    return {
        "records_with_target_packages": n_with,
        "distinct_tpl_lower": len(tpl_counter),
        "tpl_record_counts": dict(sorted(tpl_counter.items(), key=lambda x: (-x[1], x[0]))),
        "tpl_packages_per_task": {
            "mean": statistics.mean(per_task_counts) if per_task_counts else float("nan"),
            "median": statistics.median(per_task_counts) if per_task_counts else float("nan"),
            "stdev": statistics.stdev(per_task_counts) if len(per_task_counts) > 1 else 0.0,
        },
        "tasks_with_ge_2_target_packages": n_ge2,
        "tasks_with_ge_2_target_packages_percent": round(pct2, 2),
        "tasks_with_ge_3_target_packages": n_ge3,
        "tasks_with_ge_3_target_packages_percent": round(pct3, 2),
    }


def _year_from_creation_date(creation_date: str | None) -> str | None:
    if not creation_date:
        return None
    creation_text = creation_date.strip()
    if len(creation_text) >= 7 and creation_text[4] == "-" and creation_text[6] == "-" and creation_text[:4].isdigit():
        return creation_text[:4]
    try:
        dt = datetime.fromisoformat(creation_text.replace("Z", "+00:00"))
        return f"{dt.year:04d}"
    except ValueError:
        return None


def fetch_answer_creation_dates(answer_ids: list[int], db_path: Path) -> dict[int, str]:
    if not answer_ids or not db_path.is_file():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        q = f"SELECT answer_id, creation_date FROM answer_time_index WHERE answer_id IN ({','.join('?' * len(answer_ids))})"
        rows = conn.execute(q, answer_ids).fetchall()
    finally:
        conn.close()
    return {int(r[0]): str(r[1]) for r in rows}


def compute_answer_time_distribution(
    records: list[dict[str, Any]],
    db_path: Path,
) -> dict[str, Any]:
    answer_ids: list[int] = []
    for rec in records:
        aid = rec.get("answer_id")
        if aid is not None and str(aid).isdigit():
            answer_ids.append(int(aid))
    times = fetch_answer_creation_dates(answer_ids, db_path)
    by_year: Counter[str] = Counter()
    by_year_month: Counter[str] = Counter()
    missing_index = 0
    for aid in answer_ids:
        creation_date = times.get(aid)
        if creation_date is None:
            missing_index += 1
            continue
        y = _year_from_creation_date(creation_date)
        ym: str | None = None
        ct = creation_date.strip()
        if len(ct) >= 7 and ct[4] == "-" and ct[6] == "-" and ct[:4].isdigit():
            ym = ct[:7]
        else:
            try:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                ym = f"{dt.year:04d}-{dt.month:02d}"
            except ValueError:
                ym = None
        if y:
            by_year[y] += 1
        if ym:
            by_year_month[ym] += 1
    return {
        "answer_time_index_db": str(db_path.resolve()) if db_path.is_file() else str(db_path),
        "records_with_answer_id": len(answer_ids),
        "matched_in_index": len(answer_ids) - missing_index,
        "missing_in_index": missing_index,
        "by_year": dict(sorted(by_year.items())),
        "by_year_month": dict(sorted(by_year_month.items())),
    }


def compute_temporal_by_year_rows(
    records: list[dict[str, Any]],
    times: dict[int, str],
) -> list[dict[str, Any]]:
    """Per-year task counts, cumulative %, distinct ``target_packages``, knowledge regime."""
    # answer_id -> year
    year_for_record: list[str | None] = []
    for rec in records:
        aid = rec.get("answer_id")
        y: str | None = None
        if aid is not None and str(aid).isdigit():
            cd = times.get(int(aid))
            y = _year_from_creation_date(cd)
        year_for_record.append(y)

    by_year_count: Counter[str] = Counter()
    by_year_libs: dict[str, set[str]] = defaultdict(set)
    for rec, y in zip(records, year_for_record):
        if not y:
            continue
        by_year_count[y] += 1
        by_year_libs[y].update(_target_packages_normalized(rec))

    n_total = len(records)
    years_sorted = sorted(by_year_count.keys())
    cum = 0
    rows: list[dict[str, Any]] = []
    for y in years_sorted:
        cnt = by_year_count[y]
        cum += cnt
        cum_pct = round(100.0 * cum / n_total, 1) if n_total else 0.0
        rows.append(
            {
                "year": y,
                "n_tasks": cnt,
                "cumulative_percent": cum_pct,
                "n_unique_libs_target_packages": len(by_year_libs[y]),
                "knowledge_regime": _KNOWLEDGE_REGIME_BY_YEAR.get(y, "---"),
            }
        )
    return rows


def _mean_median_stdev(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return float("nan"), float("nan"), float("nan")
    m = statistics.mean(xs)
    med = statistics.median(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, med, sd


def _token_len(enc, text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def _selected_code(record: dict[str, Any]) -> str:
    parsed = record.get("parsed_code") or {}
    return str(parsed.get("selected_code") or "").strip()


def _raw_question_text(record: dict[str, Any]) -> str:
    title = (record.get("question_title") or "").strip()
    body = strip_html(record.get("question_text") or "")
    if title and body:
        return f"{title}\n\n{body}"
    if title:
        return title
    if body:
        return body
    return ""


def _round_row(m: float, med: float, sd: float, *, nd: int = 1) -> tuple[str, str, str]:
    def fmt(x: float) -> str:
        if x != x:
            return "—"
        return f"{x:.{nd}f}"

    return fmt(m), fmt(med), fmt(sd)


def compute_content_stats(
    records: list[dict[str, Any]],
    enc: Any,
    *,
    python_version: str,
) -> dict[str, Any]:
    question_tokens: list[float] = []
    code_lines: list[float] = []
    code_tokens: list[float] = []
    n_missing = 0
    for rec in records:
        question_tokens.append(float(_token_len(enc, _raw_question_text(rec))))
        code = _selected_code(rec)
        if not code:
            n_missing += 1
        code_lines.append(float(len(code.splitlines()) if code else 0))
        code_tokens.append(float(_token_len(enc, code)))

    mq, medq, sdq = _mean_median_stdev(question_tokens)
    ml, medl, sdl = _mean_median_stdev(code_lines)
    mt, medt, sdt = _mean_median_stdev(code_tokens)
    return {
        "n_missing_selected_code": n_missing,
        "question_definition": "question_title + strip_html(question_text); no build_prompt suffix",
        "question_raw_tokens": {"mean": mq, "median": medq, "stdev": sdq},
        "accepted_answer_selected_code": {
            "lines": {"mean": ml, "median": medl, "stdev": sdl},
            "tokens": {"mean": mt, "median": medt, "stdev": sdt},
        },
        "stdlib_list_python_version": python_version,
    }


def build_full_report(
    path: Path,
    *,
    enc: Any,
    tiktoken_encoding: str,
    tiktoken_model: str | None,
    python_version: str,
    answer_time_db: Path | None,
) -> dict[str, Any]:
    records = load_records_jsonl(path)
    n_records = len(records)

    tpl_stats = compute_target_package_stats(records)
    content = compute_content_stats(records, enc, python_version=python_version)

    temporal_rows: list[dict[str, Any]] = []
    answer_time_block: dict[str, Any] = {"skipped": True, "reason": "no_db_or_disabled"}
    times_map: dict[int, str] = {}

    if answer_time_db and answer_time_db.is_file():
        answer_time_block = compute_answer_time_distribution(records, answer_time_db)
        answer_ids = [int(rec["answer_id"]) for rec in records if rec.get("answer_id") is not None and str(rec["answer_id"]).isdigit()]
        times_map = fetch_answer_creation_dates(answer_ids, answer_time_db)
        temporal_rows = compute_temporal_by_year_rows(records, times_map)
        answer_time_block["skipped"] = False

    return {
        "input_jsonl": str(path.resolve()),
        "n_records": n_records,
        "tiktoken": {
            "encoding": getattr(enc, "name", tiktoken_encoding),
            "model": tiktoken_model,
            "note": "OpenAI-style token counts via tiktoken; document encoding in paper.",
        },
        **content,
        "library_coverage_target_packages": {
            "distinct_tpl": tpl_stats["distinct_tpl_lower"],
            "tpl_record_counts": tpl_stats["tpl_record_counts"],
            "records_with_target_packages": tpl_stats["records_with_target_packages"],
            "tpl_packages_per_task": tpl_stats["tpl_packages_per_task"],
            "tasks_with_ge_2_target_packages": tpl_stats["tasks_with_ge_2_target_packages"],
            "tasks_with_ge_2_target_packages_percent": tpl_stats["tasks_with_ge_2_target_packages_percent"],
            "tasks_with_ge_3_target_packages": tpl_stats["tasks_with_ge_3_target_packages"],
            "tasks_with_ge_3_target_packages_percent": tpl_stats["tasks_with_ge_3_target_packages_percent"],
        },
        "answer_time_from_index": answer_time_block,
        "temporal_by_year": temporal_rows,
    }


def legacy_d2_json_shape(full: dict[str, Any]) -> dict[str, Any]:
    """Emit the legacy JSON shape expected by ``stats_d2_so_e2e_jsonl`` consumers."""
    lib = full["library_coverage_target_packages"]
    at = full["answer_time_from_index"]
    if at.get("skipped"):
        raise SystemExit(
            "answer_time_from_index was skipped; legacy stats_d2 output needs "
            f"--answer-time-db (default: {ANSWER_TIME_INDEX_DB})"
        )
    return {
        "input_jsonl": full["input_jsonl"],
        "answer_time_index_db": at["answer_time_index_db"],
        "record_lines": full["n_records"],
        "records_with_target_packages": lib["records_with_target_packages"],
        "distinct_tpl_lower": lib["distinct_tpl"],
        "tpl_record_counts": lib["tpl_record_counts"],
        "answer_time_from_index": {
            "records_with_answer_id": at["records_with_answer_id"],
            "matched_in_index": at["matched_in_index"],
            "missing_in_index": at["missing_in_index"],
            "by_year": at["by_year"],
            "by_year_month": at["by_year_month"],
        },
    }


def print_latex_content_rows(full: dict[str, Any]) -> None:
    c = full["question_raw_tokens"]
    code = full["accepted_answer_selected_code"]
    lib = full["library_coverage_target_packages"]
    mq, medq, sdq = c["mean"], c["median"], c["stdev"]
    ml, medl, sdl = code["lines"]["mean"], code["lines"]["median"], code["lines"]["stdev"]
    mt, medt, sdt = code["tokens"]["mean"], code["tokens"]["median"], code["tokens"]["stdev"]
    pt = lib["tpl_packages_per_task"]
    mi, medi, sdi = pt["mean"], pt["median"], pt["stdev"]
    q1, q2, q3 = _round_row(mq, medq, sdq)
    l1, l2, l3 = _round_row(ml, medl, sdl)
    t1, t2, t3 = _round_row(mt, medt, sdt)
    i1, i2, i3 = _round_row(mi, medi, sdi)
    d = lib["distinct_tpl"]
    n2 = lib["tasks_with_ge_2_target_packages"]
    p2 = lib["tasks_with_ge_2_target_packages_percent"]
    n3 = lib["tasks_with_ge_3_target_packages"]
    p3 = lib["tasks_with_ge_3_target_packages_percent"]
    print(
        "\n% --- tab:dataset_content (paste under \\midrule blocks) ---\n"
        f"\\multicolumn{{4}}{{l}}{{\\textit{{Question}}}} \\\\\n"
        f"\\ \\ Token length             & {q1} & {q2} & {q3} \\\\\n"
        f"\\midrule\n"
        f"\\multicolumn{{4}}{{l}}{{\\textit{{Accepted answer code block}}}} \\\\\n"
        f"\\ \\ Lines of code            & {l1} & {l2} & {l3} \\\\\n"
        f"\\ \\ Token length             & {t1} & {t2} & {t3} \\\\\n"
        f"\\ \\ TPL imports per task     & {i1} & {i2} & {i3} \\\\\n"
        f"\\midrule\n"
        f"\\multicolumn{{4}}{{l}}{{\\textit{{Library coverage (target\\_packages)}}}} \\\\\n"
        f"\\ \\ Distinct TPLs            & \\multicolumn{{3}}{{r}}{{{d}}} \\\\\n"
        f"\\ \\ Tasks with $\\geq$2 TPL imports & \\multicolumn{{3}}{{r}}{{{n2} ({p2:.1f}\\%)}} \\\\\n"
        f"\\ \\ Tasks with $\\geq$3 TPL imports & \\multicolumn{{3}}{{r}}{{{n3} ({p3:.1f}\\%)}} \\\\\n",
        flush=True,
    )


def latex_temporal_table_float(full: dict[str, Any]) -> str:
    """Return a full ``table`` float (requires ``booktabs``); empty string if no rows."""
    rows = full.get("temporal_by_year") or []
    if not rows:
        return ""
    lib = full["library_coverage_target_packages"]
    n_tot = full["n_records"]
    d_tot = lib["distinct_tpl"]
    caption = (
        r"\caption{Temporal distribution of the 1,000 Stack Overflow tasks. "
        r"The three knowledge regimes reflect each year's likely relationship to "
        r"LLM training corpora: pre-adoption questions are likely memorized; "
        r"deployment-era questions may be partially covered; post-cutoff "
        r"questions probe out-of-distribution version knowledge. "
        r"\FIXME{GIVE MORE INFO ABOUT OUR DATASET.}}"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        caption,
        r"\label{tab:dataset_temporal}",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"\textbf{Year} & \textbf{\# Tasks} & \textbf{Cumul.\ (\%)}"
        r" & \textbf{\# Unique Libs} & \textbf{Knowledge Regime} \\",
        r"\midrule",
    ]
    for r in rows:
        y = r["year"]
        n = r["n_tasks"]
        cp = r["cumulative_percent"]
        u = r["n_unique_libs_target_packages"]
        reg = r["knowledge_regime"].replace("&", r"\&")
        lines.append(f"{y} & {n} & {cp:.1f} & {u} & {reg} \\\\")
    lines.extend(
        [
            r"\midrule",
            f"Total & {n_tot} & 100.0 & {d_tot} & --- \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def print_latex_temporal_table(full: dict[str, Any]) -> None:
    tex = latex_temporal_table_float(full)
    if not tex:
        print("\n% tab:dataset_temporal: no rows (need answer time DB, omit --skip-answer-time)", flush=True)
        return
    print("\n% --- tab:dataset_temporal (full float) ---", tex, sep="\n", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO E2E JSONL: content + temporal + TPL stats (target_packages).")
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="Balanced D2 StackOverflow JSONL path")
    p.add_argument(
        "--answer-time-db",
        type=Path,
        default=ANSWER_TIME_INDEX_DB,
        help="SQLite index mapping answer_id -> CreationDate (default paths.ANSWER_TIME_INDEX_DB)",
    )
    p.add_argument(
        "--skip-answer-time",
        action="store_true",
        help="Skip SQLite lookups (omit answer_time + temporal_by_year)",
    )
    p.add_argument(
        "--legacy-d2-json-only",
        action="store_true",
        help="Emit only the legacy stats_d2 JSON (requires the answer-time DB)",
    )
    p.add_argument(
        "--tiktoken-encoding",
        type=str,
        default="o200k_base",
        help="tiktoken encoding name",
    )
    p.add_argument(
        "--tiktoken-model",
        type=str,
        default=None,
        metavar="MODEL",
        help="If set, call encoding_for_model(MODEL) and ignore --tiktoken-encoding",
    )
    p.add_argument("--python-version", type=str, default="3.12", help="stdlib_list tag for auxiliary fields")
    p.add_argument("--output-json", type=Path, default=None, help="Write the full structured JSON report")
    p.add_argument("--print-latex-rows", action="store_true", help="Print LaTeX rows for tab:dataset_content")
    p.add_argument(
        "--print-latex-temporal",
        action="store_true",
        help="Print the full tab:dataset_temporal LaTeX float (needs booktabs)",
    )
    p.add_argument(
        "--latex-temporal-only",
        action="store_true",
        help="Temporal LaTeX only (skips JSON stdout; forces answer-time DB usage)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.jsonl
    if not path.is_file():
        raise SystemExit(f"JSONL not found: {path}")

    import tiktoken

    if args.tiktoken_model:
        enc = tiktoken.encoding_for_model(args.tiktoken_model)
    else:
        enc = tiktoken.get_encoding(args.tiktoken_encoding)

    if args.latex_temporal_only:
        args.skip_answer_time = False
        args.print_latex_temporal = True
        args.legacy_d2_json_only = False

    db = None if args.skip_answer_time else args.answer_time_db
    if db and not db.is_file() and not args.skip_answer_time:
        raise SystemExit(f"answer time DB not found: {db} (use --skip-answer-time to continue)")

    full = build_full_report(
        path,
        enc=enc,
        tiktoken_encoding=args.tiktoken_encoding,
        tiktoken_model=args.tiktoken_model,
        python_version=args.python_version,
        answer_time_db=db,
    )

    if args.latex_temporal_only:
        pass  # JSON intentionally suppressed in this mode
    elif args.legacy_d2_json_only:
        legacy = legacy_d2_json_shape(full)
        print(json.dumps(legacy, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(full, ensure_ascii=False, indent=2))

    if args.output_json and not args.legacy_d2_json_only and not args.latex_temporal_only:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output_json}", flush=True)

    if args.print_latex_rows:
        print_latex_content_rows(full)
    if args.print_latex_temporal:
        print_latex_temporal_table(full)


if __name__ == "__main__":
    main()
