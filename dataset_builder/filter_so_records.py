"""Filter StackExchange JSONL by import evidence and answer timestamps.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m dataset_builder.filter_so_records --help
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from loguru import logger
from tqdm import tqdm


def _slugify_cutoff(cutoff: str) -> str:
    core = cutoff.split("T")[0] if "T" in cutoff else cutoff
    return re.sub(r"[^0-9]", "", core) or "cutoff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter built StackExchange records by import evidence and answer post date."
    )
    parser.add_argument(
        "--source-run-name",
        required=True,
        help="Run name under dataset_runs, e.g., so_full_balanced_resume",
    )
    parser.add_argument(
        "--posts-xml-path",
        default=None,
        help="Path to StackExchange Posts.xml. Default: resources/stackoverflow/Posts.xml",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2020-01-01T00:00:00",
        help="Keep answers with CreationDate >= cutoff-date (ISO-like string).",
    )
    parser.add_argument(
        "--no-require-import-evidence",
        action="store_true",
        help="If set, skip import-evidence filtering and only apply date filtering.",
    )
    parser.add_argument(
        "--date-filter-mode",
        choices=["exact", "id_threshold"],
        default="exact",
        help="exact: verify each candidate with Posts.xml CreationDate; id_threshold: use cutoff answer_id lower-bound cache for faster repeated runs.",
    )
    parser.add_argument(
        "--answer-time-index-db",
        default=None,
        help="Optional sqlite path for answer_id->CreationDate index; defaults to paths.ANSWER_TIME_INDEX_DB",
    )
    return parser.parse_args()


def build_answer_time_index(posts_xml: Path, index_db_path: Path, batch_size: int = 200000) -> dict:
    """Extract answer_id and CreationDate from Posts.xml into sqlite for fast reuse."""
    index_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(index_db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS answer_time_index (
            answer_id INTEGER PRIMARY KEY,
            creation_date TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    row = conn.execute("SELECT value FROM meta WHERE key='posts_xml'").fetchone()
    indexed_posts_xml = row[0] if row else ""
    complete_row = conn.execute("SELECT value FROM meta WHERE key='index_complete'").fetchone()
    index_complete = (complete_row[0] == "1") if complete_row else False
    existing = int(conn.execute("SELECT COUNT(*) FROM answer_time_index").fetchone()[0] or 0)

    if index_complete and indexed_posts_xml == str(posts_xml) and existing > 0:
        conn.close()
        return {
            "index_db_path": str(index_db_path),
            "index_reused": True,
            "answer_rows_indexed": existing,
            "posts_xml": str(posts_xml),
        }

    # Rebuild when source XML changes or index incomplete.
    conn.execute("DELETE FROM answer_time_index;")
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('posts_xml', ?)", (str(posts_xml),))
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('index_complete', '0')")
    conn.commit()

    rows: list[tuple[int, str]] = []
    total_rows = 0
    answer_rows = 0
    context = ET.iterparse(str(posts_xml), events=("end",))
    for _, elem in tqdm(context, desc="Build answer_time_index"):
        if elem.tag != "row":
            elem.clear()
            continue
        total_rows += 1
        if elem.attrib.get("PostTypeId") == "2":
            answer_id = elem.attrib.get("Id")
            creation_date = elem.attrib.get("CreationDate", "")
            if answer_id and answer_id.isdigit() and creation_date:
                rows.append((int(answer_id), creation_date))
                answer_rows += 1
        if len(rows) >= batch_size:
            conn.executemany(
                "INSERT OR REPLACE INTO answer_time_index(answer_id, creation_date) VALUES(?, ?)",
                rows,
            )
            conn.commit()
            rows.clear()
        elem.clear()
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO answer_time_index(answer_id, creation_date) VALUES(?, ?)",
            rows,
        )
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('index_complete', '1')")
    conn.commit()
    indexed = int(conn.execute("SELECT COUNT(*) FROM answer_time_index").fetchone()[0] or 0)
    conn.close()
    return {
        "index_db_path": str(index_db_path),
        "index_reused": False,
        "rows_seen_total": total_rows,
        "answer_rows_seen": answer_rows,
        "answer_rows_indexed": indexed,
        "posts_xml": str(posts_xml),
    }


def _keep_ids_by_exact_date_from_index(
    *,
    index_db_path: Path,
    candidate_ids: set[str],
    cutoff_date: str,
) -> tuple[set[str], int]:
    conn = sqlite3.connect(index_db_path)
    conn.execute("CREATE TEMP TABLE tmp_candidate_ids(answer_id INTEGER PRIMARY KEY)")
    candidate_ints = [int(x) for x in candidate_ids if x.isdigit()]
    conn.executemany(
        "INSERT OR REPLACE INTO tmp_candidate_ids(answer_id) VALUES(?)",
        [(x,) for x in candidate_ints],
    )
    seen = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM tmp_candidate_ids t
            JOIN answer_time_index a ON a.answer_id = t.answer_id
            """
        ).fetchone()[0]
        or 0
    )
    rows = conn.execute(
        """
        SELECT t.answer_id
        FROM tmp_candidate_ids t
        JOIN answer_time_index a ON a.answer_id = t.answer_id
        WHERE a.creation_date >= ?
        """,
        (cutoff_date,),
    ).fetchall()
    keep_ids = {str(r[0]) for r in rows}
    conn.close()
    return keep_ids, seen


def _discover_cutoff_answer_id(posts_xml: Path, cutoff_date: str, cache_path: Path) -> tuple[int | None, str | None]:
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("posts_xml") == str(posts_xml) and payload.get("cutoff_creation_date") == cutoff_date:
                cid = payload.get("cutoff_answer_id")
                cdt = payload.get("cutoff_answer_creation_date")
                if isinstance(cid, int) and cid > 0:
                    return cid, cdt if isinstance(cdt, str) else None
        except json.JSONDecodeError:
            pass

    cutoff_answer_id: int | None = None
    cutoff_answer_creation_date: str | None = None
    context = ET.iterparse(str(posts_xml), events=("end",))
    for _, elem in tqdm(context, desc="Discover cutoff answer_id"):
        if elem.tag != "row":
            elem.clear()
            continue
        if elem.attrib.get("PostTypeId") != "2":
            elem.clear()
            continue
        creation_date = elem.attrib.get("CreationDate", "")
        if creation_date >= cutoff_date:
            aid = elem.attrib.get("Id")
            if aid and aid.isdigit():
                cutoff_answer_id = int(aid)
                cutoff_answer_creation_date = creation_date
            elem.clear()
            break
        elem.clear()

    cache_payload = {
        "posts_xml": str(posts_xml),
        "cutoff_creation_date": cutoff_date,
        "cutoff_answer_id": cutoff_answer_id,
        "cutoff_answer_creation_date": cutoff_answer_creation_date,
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return cutoff_answer_id, cutoff_answer_creation_date


def filter_records(
    *,
    run_dir: Path,
    posts_xml: Path,
    cutoff_date: str,
    require_import_evidence: bool,
    date_filter_mode: str,
    answer_time_index_db: Path | None,
) -> dict:
    from paths import ANSWER_TIME_INDEX_DB

    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"records file not found: {records_path}")
    if not posts_xml.exists():
        raise FileNotFoundError(f"posts xml not found: {posts_xml}")

    suffix = _slugify_cutoff(cutoff_date)
    candidates_path = run_dir / "records_import_candidates.jsonl"
    final_path = run_dir / f"records_import_postdate_ge{suffix}.jsonl"
    stats_path = run_dir / f"records_import_postdate_ge{suffix}.stats.json"
    cutoff_cache_path = run_dir / f"cutoff_answer_id_ge{suffix}.json"
    index_db_path = answer_time_index_db or ANSWER_TIME_INDEX_DB

    candidate_ids: set[str] = set()
    total_records = 0
    import_candidates = 0

    logger.info("Step1: selecting import-evidence candidates from records.jsonl")
    with records_path.open("r", encoding="utf-8") as reader, candidates_path.open("w", encoding="utf-8") as writer:
        for line in tqdm(reader, desc="Filter by import evidence"):
            total_records += 1
            obj = json.loads(line)
            if require_import_evidence:
                match_source = ((obj.get("metadata") or {}).get("match_source") or {})
                answer_code_imports = match_source.get("answer_code_imports") or []
                import_gate = match_source.get("import_gate") or []
                if not answer_code_imports and not import_gate:
                    continue
            answer_id = str(obj.get("answer_id") or "")
            if not answer_id:
                continue
            candidate_ids.add(answer_id)
            writer.write(line)
            import_candidates += 1

    logger.info(
        f"Step1 done: total_records={total_records}, import_candidates={import_candidates}, unique_answer_ids={len(candidate_ids)}"
    )

    keep_ids: set[str] = set()
    candidates_seen_in_posts_answers = 0
    cutoff_answer_id: int | None = None
    cutoff_answer_creation_date: str | None = None
    index_stats = build_answer_time_index(posts_xml=posts_xml, index_db_path=index_db_path)
    if date_filter_mode == "exact":
        logger.info("Step2(exact): querying answer_time_index by candidate ids and cutoff date")
        keep_ids, candidates_seen_in_posts_answers = _keep_ids_by_exact_date_from_index(
            index_db_path=index_db_path,
            candidate_ids=candidate_ids,
            cutoff_date=cutoff_date,
        )
    elif date_filter_mode == "id_threshold":
        logger.info("Step2(id_threshold): discovering/reusing cutoff answer_id")
        conn = sqlite3.connect(index_db_path)
        row = conn.execute(
            """
            SELECT answer_id, creation_date
            FROM answer_time_index
            WHERE creation_date >= ?
            ORDER BY answer_id ASC
            LIMIT 1
            """,
            (cutoff_date,),
        ).fetchone()
        conn.close()
        if row is None:
            cutoff_answer_id, cutoff_answer_creation_date = _discover_cutoff_answer_id(
                posts_xml=posts_xml,
                cutoff_date=cutoff_date,
                cache_path=cutoff_cache_path,
            )
        else:
            cutoff_answer_id, cutoff_answer_creation_date = int(row[0]), str(row[1])
            cutoff_cache_path.write_text(
                json.dumps(
                    {
                        "posts_xml": str(posts_xml),
                        "cutoff_creation_date": cutoff_date,
                        "cutoff_answer_id": cutoff_answer_id,
                        "cutoff_answer_creation_date": cutoff_answer_creation_date,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if cutoff_answer_id is None:
            logger.warning("No cutoff answer_id found; fallback to empty keep set.")
        else:
            for aid in candidate_ids:
                if aid.isdigit() and int(aid) >= cutoff_answer_id:
                    keep_ids.add(aid)
            candidates_seen_in_posts_answers = len(candidate_ids)
    else:
        raise ValueError(f"Unsupported date_filter_mode: {date_filter_mode}")

    logger.info(f"Step2 done: mode={date_filter_mode}, candidates_seen={candidates_seen_in_posts_answers}, keep_ids={len(keep_ids)}")

    logger.info("Step3: materializing final filtered records")
    final_records = 0
    with candidates_path.open("r", encoding="utf-8") as reader, final_path.open("w", encoding="utf-8") as writer:
        for line in tqdm(reader, desc="Write final filtered records"):
            obj = json.loads(line)
            answer_id = str(obj.get("answer_id") or "")
            if answer_id in keep_ids:
                writer.write(line)
                final_records += 1

    stats = {
        "source_records": str(records_path),
        "posts_xml": str(posts_xml),
        "cutoff_creation_date": cutoff_date,
        "require_import_evidence": require_import_evidence,
        "date_filter_mode": date_filter_mode,
        "cutoff_answer_id": cutoff_answer_id,
        "cutoff_answer_creation_date": cutoff_answer_creation_date,
        "answer_time_index": index_stats,
        "total_records": total_records,
        "import_candidates": import_candidates,
        "candidate_unique_answer_ids": len(candidate_ids),
        "candidates_seen_in_posts_answers": candidates_seen_in_posts_answers,
        "postdate_keep_unique_answer_ids": len(keep_ids),
        "final_records": final_records,
        "output_candidates": str(candidates_path),
        "output_final": str(final_path),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    from paths import OUTPUTS, STACKOVERFLOW, ensure_dirs

    args = parse_args()
    ensure_dirs()
    posts_xml_path = args.posts_xml_path or str(STACKOVERFLOW / "Posts.xml")
    run_dir = OUTPUTS / "dataset_builder" / args.source_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "post_filter.log")

    stats = filter_records(
        run_dir=run_dir,
        posts_xml=Path(posts_xml_path),
        cutoff_date=args.cutoff_date,
        require_import_evidence=not bool(args.no_require_import_evidence),
        date_filter_mode=args.date_filter_mode,
        answer_time_index_db=Path(args.answer_time_index_db) if args.answer_time_index_db else None,
    )
    logger.success(f"Post filtering completed: {stats}")


if __name__ == "__main__":
    main()

