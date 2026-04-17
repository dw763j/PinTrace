"""StackExchange ``Posts.xml`` → JSONL record extraction (Pass1/Pass2).

Usage: imported by ``dataset_builder.build_stackoverflow_dataset`` (no CLI).

    from dataset_builder.stackexchange_builder import build_records_from_stackexchange_xml_v2
"""
from __future__ import annotations

import ast
import html
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .schema import CodeBlock, SORecord

INLINE_CODE_BLOCK_RE = re.compile(r"<pre><code>(.*?)</code></pre>", re.S)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-\.]*")
PY_CONTEXT_CUE_RE = re.compile(
    r"(?:import|from|pip\s+install|conda\s+install)\s+([A-Za-z_][A-Za-z0-9_\-\.]*)",
    re.I,
)


def load_alias_mapping(
    target_packages: set[str],
    alias_mapping_path: str | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build alias->canonical map.

    alias_mapping_path format:
    {
      "opencv-python": ["cv2", "opencv_python"],
      "scikit-learn": ["sklearn"]
    }
    """
    canonical_to_aliases: dict[str, list[str]] = {pkg: [pkg] for pkg in target_packages}
    if alias_mapping_path:
        with open(alias_mapping_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for canonical, aliases in payload.items():
            c = str(canonical).lower()
            if c not in canonical_to_aliases:
                continue
            valid_aliases = [str(a).lower() for a in aliases if isinstance(a, str) and a.strip()]
            canonical_to_aliases[c] = sorted(set([c] + valid_aliases))

    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in canonical_to_aliases.items():
        for alias in aliases:
            alias_to_canonical[alias] = canonical
    return alias_to_canonical, canonical_to_aliases


def _extract_code_blocks_from_html(text: str | None) -> list[CodeBlock]:
    if not text:
        return []
    out: list[CodeBlock] = []
    for match in INLINE_CODE_BLOCK_RE.finditer(text):
        code = html.unescape(match.group(1)).strip()
        if code:
            out.append(CodeBlock(language="python", content=code))
    return out


def _tags_str_to_list(tags: str | None) -> list[str]:
    if not tags:
        return []
    raw = tags.replace("<", " ").replace(">", " ")
    return [x.strip().lower() for x in raw.split() if x.strip()]


def _match_aliases_in_text(texts: list[str], alias_to_canonical: dict[str, str]) -> set[str]:
    joined = "\n".join([t.lower() for t in texts if t])
    tokens = set(TOKEN_RE.findall(joined))
    hits = set()
    for token in tokens:
        canonical = alias_to_canonical.get(token)
        if canonical:
            hits.add(canonical)
    return hits


def _match_aliases_in_python_context_cues(texts: list[str], alias_to_canonical: dict[str, str]) -> set[str]:
    """Match package aliases only when preceded by Python/package-management cues."""
    hits = set()
    for text in texts:
        if not text:
            continue
        for m in PY_CONTEXT_CUE_RE.finditer(text):
            token = m.group(1).strip().lower()
            token = re.split(r"[<>=!\[\],;\s]", token)[0]
            top = token.split(".")[0]
            canonical = alias_to_canonical.get(top)
            if canonical:
                hits.add(canonical)
    return hits


def _match_aliases_in_tags(tags: list[str], alias_to_canonical: dict[str, str]) -> set[str]:
    """Match package aliases in tags using alias_to_canonical map."""
    hits = set()
    for tag in tags:
        canonical = alias_to_canonical.get(tag.lower())
        if canonical:
            hits.add(canonical)
    return hits


def _extract_import_roots_from_code(code_blocks: list[CodeBlock]) -> set[str]:
    roots = set()
    for block in code_blocks:
        code = block.content
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0].lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    roots.add(node.module.split(".")[0].lower())
    return roots  # parsed import root module names


def _extract_import_hits_from_code(code_blocks: list[CodeBlock], alias_to_canonical: dict[str, str]) -> set[str]:
    """Extract import hits from code blocks using alias_to_canonical map."""
    hits = set()
    for block in code_blocks:
        code = block.content
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0].lower()
                    canonical = alias_to_canonical.get(top)
                    if canonical:
                        hits.add(canonical)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0].lower()
                    canonical = alias_to_canonical.get(top)
                    if canonical:
                        hits.add(canonical)
    return hits


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (
            question_id TEXT PRIMARY KEY,
            accepted_answer_id TEXT NOT NULL,
            title TEXT,
            body TEXT,
            tags_json TEXT,
            is_python_tag INTEGER,
            question_score INTEGER,
            question_tag_hits_json TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_accepted ON questions(accepted_answer_id);")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS written_answers (
            answer_id TEXT PRIMARY KEY
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def _meta_get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return str(row[0])


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
        (key, value),
    )


def _reset_index_state(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM questions")
    conn.execute("DELETE FROM written_answers")
    conn.execute("DELETE FROM metadata")
    conn.commit()


def _run_pass1_questions(
    *,
    posts_xml_path: str,
    conn: sqlite3.Connection,
    min_question_score: int,
    skip_if_exists: bool = True,
) -> dict[str, int]:
    indexed_posts_xml = _meta_get(conn, "posts_xml_path", "")
    indexed_min_score = _meta_get(conn, "pass1_min_question_score", "")
    source_changed = bool(indexed_posts_xml and indexed_posts_xml != posts_xml_path)
    try:
        min_score_changed = bool(indexed_min_score and int(indexed_min_score) != int(min_question_score))
    except ValueError:
        min_score_changed = bool(indexed_min_score)
    if source_changed or min_score_changed:
        logger.warning(
            "Pass1 index settings changed; rebuilding index."
        )
        _reset_index_state(conn)
    _meta_set(conn, "posts_xml_path", posts_xml_path)
    _meta_set(conn, "pass1_min_question_score", str(int(min_question_score)))
    conn.commit()

    pass1_complete = _meta_get(conn, "pass1_complete", "0") == "1"
    existing = int(conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0] or 0)
    if skip_if_exists and pass1_complete:
        logger.info(f"Pass1 skipped: already completed, indexed_questions={existing}.")
        saved_stats_raw = _meta_get(conn, "pass1_stats", "")
        if saved_stats_raw:
            try:
                saved = json.loads(saved_stats_raw)
                saved["skipped_existing"] = 1
                return saved
            except json.JSONDecodeError:
                pass
        return {"indexed_questions": existing, "skipped_existing": 1}

    saved_stats_raw = _meta_get(conn, "pass1_stats", "")
    saved_last_row = int(_meta_get(conn, "pass1_last_row", "0") or 0)
    if saved_stats_raw:
        try:
            stats = json.loads(saved_stats_raw)
        except json.JSONDecodeError:
            stats = {}
    else:
        stats = {}
    stats = {
        "seen_questions": int(stats.get("seen_questions", 0)),
        "accepted_questions": int(stats.get("accepted_questions", 0)),
        "indexed_questions": int(stats.get("indexed_questions", existing)),
        "skip_low_score": int(stats.get("skip_low_score", 0)),
        "python_tag_questions": int(stats.get("python_tag_questions", 0)),
    }
    if existing > stats["indexed_questions"]:
        stats["indexed_questions"] = existing

    if saved_last_row > 0:
        logger.info(f"Pass1 resume from row={saved_last_row}, indexed_questions={stats['indexed_questions']}")

    row_idx = 0

    context = ET.iterparse(posts_xml_path, events=("end",))
    for _, elem in tqdm(context, desc="Pass1 index questions"):
        if elem.tag != "row":
            elem.clear()
            continue
        row_idx += 1
        if row_idx <= saved_last_row:
            elem.clear()
            continue
        if elem.attrib.get("PostTypeId") != "1":
            if row_idx % 10000 == 0:
                conn.commit()
                _meta_set(conn, "pass1_last_row", str(row_idx))
                _meta_set(conn, "pass1_stats", json.dumps(stats, ensure_ascii=False))
                conn.commit()
            elem.clear()
            continue

        stats["seen_questions"] += 1
        qid = elem.attrib.get("Id")
        if not qid:
            elem.clear()
            continue
        # Data dump note: artificial post ids starting from 1,000,000,001 can be ignored.
        try:
            if int(qid) >= 1_000_000_000:
                elem.clear()
                continue
        except ValueError:
            pass

        accepted_answer_id = elem.attrib.get("AcceptedAnswerId")
        if not accepted_answer_id:
            elem.clear()
            continue
        stats["accepted_questions"] += 1

        score = int(elem.attrib.get("Score", "0") or 0)
        if score < min_question_score:
            stats["skip_low_score"] += 1
            elem.clear()
            continue

        title = html.unescape(elem.attrib.get("Title", "") or "")
        body = html.unescape(elem.attrib.get("Body", "") or "")
        tags = _tags_str_to_list(elem.attrib.get("Tags"))
        is_python_tag = int(("python" in tags) or any(tag.startswith("python-") for tag in tags))
        if is_python_tag:
            stats["python_tag_questions"] += 1
        conn.execute(
            """
            INSERT OR REPLACE INTO questions(
              question_id, accepted_answer_id, title, body, tags_json, is_python_tag, question_score, question_tag_hits_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                qid,
                accepted_answer_id,
                title,
                body,
                json.dumps(tags, ensure_ascii=False),
                is_python_tag,
                score,
                "[]",
            ),
        )
        stats["indexed_questions"] += 1
        if row_idx % 10000 == 0:
            conn.commit()
            _meta_set(conn, "pass1_last_row", str(row_idx))
            _meta_set(conn, "pass1_stats", json.dumps(stats, ensure_ascii=False))
            conn.commit()
        elem.clear()
    conn.commit()
    _meta_set(conn, "pass1_last_row", str(row_idx))
    _meta_set(conn, "pass1_stats", json.dumps(stats, ensure_ascii=False))
    _meta_set(conn, "pass1_complete", "1")
    conn.commit()
    return stats


def _run_pass2_answers(
    *,
    posts_xml_path: str,
    conn: sqlite3.Connection,
    alias_to_canonical: dict[str, str],
    out_jsonl_path: str,
    match_mode: str = "strict",
    workers: int = 1,
    max_records: int | None = None,
) -> dict[str, int]:
    out_file = Path(out_jsonl_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "seen_answers": 0,
        "matched_accepted_answers": 0,
        "skip_already_written": 0,
        "skip_no_code": 0,
        "skip_non_python_context": 0,
        "skip_no_target_hit": 0,
        "written": 0,
    }

    accepted_rows = conn.execute(
        """
        SELECT accepted_answer_id, question_id, title, body, tags_json, is_python_tag, question_score, question_tag_hits_json
        FROM questions
        """
    ).fetchall()
    accepted_map = {
        str(row[0]): (str(row[1]), row[2], row[3], row[4], int(row[5]), int(row[6] or 0), row[7])
        for row in accepted_rows
        if row[0] is not None
    }
    written_answer_rows = conn.execute("SELECT answer_id FROM written_answers").fetchall()
    written_answers = {str(r[0]) for r in written_answer_rows if r[0] is not None}
    logger.info(
        f"Pass2 preload: accepted_map={len(accepted_map)} written_answers={len(written_answers)} workers={workers}"
    )

    effective_mode = {"strict": "balanced", "relaxed": "recall"}.get(match_mode, match_mode)  # "conservative", "balanced", "recall", "strict", "relaxed"
    if effective_mode not in {"conservative", "balanced", "recall"}:
        raise ValueError(f"Unsupported match_mode: {match_mode}")

    def process_one(payload: dict) -> tuple[str, str, dict | None]:
        answer_id = payload["answer_id"]
        qrow = payload["question_row"]
        answer_body = payload["answer_body"]
        answer_score = payload["answer_score"]

        blocks = _extract_code_blocks_from_html(answer_body)
        if not blocks:
            return ("skip_no_code", answer_id, None)

        qid, title, q_body, tags_json, is_python_tag, q_score, _q_tag_hits_json = qrow
        tags = json.loads(tags_json or "[]")
        q_tag_hits = _match_aliases_in_tags(tags, alias_to_canonical)  # question tags hit target TPL or alias
        import_roots = _extract_import_roots_from_code(blocks)
        is_python_context = bool(is_python_tag) or bool(import_roots)
        if not is_python_context:
            return ("skip_non_python_context", answer_id, None)

        answer_code_hits = _extract_import_hits_from_code(blocks, alias_to_canonical)
        if effective_mode == "conservative":
            hits = set(answer_code_hits)
        elif effective_mode == "balanced":
            hits = set(q_tag_hits).union(answer_code_hits)
        else:
            cue_hits = _match_aliases_in_python_context_cues(
                [title or "", q_body or "", answer_body] + [b.content for b in blocks],
                alias_to_canonical,
            )
            hits = set(q_tag_hits).union(answer_code_hits).union(cue_hits)

        raw_text_hits = _match_aliases_in_text([title or "", q_body or "", answer_body], alias_to_canonical)
        hits = sorted(hits)
        if not hits:
            return ("skip_no_target_hit", answer_id, None)

        record = SORecord(
            record_id=f"stackexchange-{qid}",
            source="stackexchange_dump",
            question_id=str(qid),
            answer_id=str(answer_id),
            question_title=title,
            question_text=q_body,
            answer_text=answer_body,
            code_blocks=blocks,
            tags=tags,
            accepted=True,
            score=answer_score,
            target_packages=hits,
            metadata={
                "question_score": int(q_score or 0),
                "answer_score": int(answer_score or 0),
                "match_source": {
                    "python_tag_gate": bool(is_python_tag),
                    "import_gate": sorted(import_roots),  # import roots parsed from answer code
                    "match_mode": effective_mode,
                    "question_tags": sorted(q_tag_hits),
                    "answer_code_imports": sorted(answer_code_hits),
                    "raw_text_hits_observed": sorted(raw_text_hits),
                },
            },
        )
        return ("written", answer_id, record.to_dict())

    def flush_done(
        done: set[Future],
        future_map: dict[Future, str],
        writer,
    ) -> None:
        for fut in done:
            _answer_id = future_map.pop(fut, "")
            status, answer_id, record_dict = fut.result()
            if status == "written" and record_dict is not None:
                writer.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
                conn.execute("INSERT OR REPLACE INTO written_answers(answer_id) VALUES(?)", (answer_id,))
                written_answers.add(answer_id)
                stats["written"] += 1
                if stats["written"] % 2000 == 0:
                    conn.commit()
            else:
                stats[status] += 1

    max_in_flight = max(32, workers * 8)
    context = ET.iterparse(posts_xml_path, events=("end",))
    with out_file.open("a", encoding="utf-8") as writer:
        if workers <= 1:
            for _, elem in tqdm(context, desc="Pass2 extract accepted answers"):
                if elem.tag != "row":
                    elem.clear()
                    continue
                if elem.attrib.get("PostTypeId") != "2":
                    elem.clear()
                    continue
                stats["seen_answers"] += 1
                answer_id = elem.attrib.get("Id")
                if not answer_id:
                    elem.clear()
                    continue
                question_row = accepted_map.get(str(answer_id))
                if question_row is None:
                    elem.clear()
                    continue
                stats["matched_accepted_answers"] += 1
                if str(answer_id) in written_answers:
                    stats["skip_already_written"] += 1
                    elem.clear()
                    continue
                payload = {
                    "answer_id": str(answer_id),
                    "question_row": question_row,
                    "answer_body": html.unescape(elem.attrib.get("Body", "") or ""),
                    "answer_score": int(elem.attrib.get("Score", "0") or 0),
                }
                status, resolved_answer_id, record_dict = process_one(payload)
                if status == "written" and record_dict is not None:
                    writer.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
                    conn.execute("INSERT OR REPLACE INTO written_answers(answer_id) VALUES(?)", (resolved_answer_id,))
                    written_answers.add(resolved_answer_id)
                    stats["written"] += 1
                    if stats["written"] % 2000 == 0:
                        conn.commit()
                else:
                    stats[status] += 1
                if max_records and stats["written"] >= max_records:
                    conn.commit()
                    elem.clear()
                    break
                elem.clear()
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map: dict[Future, str] = {}
                for _, elem in tqdm(context, desc="Pass2 extract accepted answers"):
                    if elem.tag != "row":
                        elem.clear()
                        continue
                    if elem.attrib.get("PostTypeId") != "2":
                        elem.clear()
                        continue
                    stats["seen_answers"] += 1
                    answer_id = elem.attrib.get("Id")
                    if not answer_id:
                        elem.clear()
                        continue
                    question_row = accepted_map.get(str(answer_id))
                    if question_row is None:
                        elem.clear()
                        continue
                    stats["matched_accepted_answers"] += 1
                    if str(answer_id) in written_answers:
                        stats["skip_already_written"] += 1
                        elem.clear()
                        continue
                    payload = {
                        "answer_id": str(answer_id),
                        "question_row": question_row,
                        "answer_body": html.unescape(elem.attrib.get("Body", "") or ""),
                        "answer_score": int(elem.attrib.get("Score", "0") or 0),
                    }
                    fut = executor.submit(process_one, payload)
                    future_map[fut] = str(answer_id)
                    if len(future_map) >= max_in_flight:
                        done, _ = wait(set(future_map.keys()), return_when=FIRST_COMPLETED)
                        flush_done(done, future_map, writer)
                        if max_records and stats["written"] >= max_records:
                            break
                    elem.clear()
                if future_map and (not max_records or stats["written"] < max_records):
                    done, _ = wait(set(future_map.keys()))
                    flush_done(done, future_map, writer)
    conn.commit()
    return stats


def build_records_from_stackexchange_xml_v2(
    *,
    posts_xml_path: str,
    target_packages: set[str],
    alias_mapping_path: str | None,
    out_jsonl_path: str,
    work_dir: str,
    match_mode: str = "strict",
    pass2_workers: int = 1,
    min_question_score: int = 0,
    max_records: int | None = None,
    sqlite_index_path: str | None = None,
) -> dict:
    """High-scale Stack Exchange builder using 2-pass streaming + sqlite index."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if sqlite_index_path:
        db_path = Path(sqlite_index_path)
    else:
        from paths import STACKEXCHANGE_INDEX_DB

        db_path = STACKEXCHANGE_INDEX_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    alias_to_canonical, canonical_to_aliases = load_alias_mapping(target_packages, alias_mapping_path)
    logger.info(
        f"StackExchange builder start: targets={len(target_packages)} aliases={len(alias_to_canonical)} match_mode={match_mode} pass2_workers={pass2_workers} db={db_path}"
    )
    conn = _init_db(db_path)
    try:
        pass1 = _run_pass1_questions(
            posts_xml_path=posts_xml_path,
            conn=conn,
            min_question_score=min_question_score,
            skip_if_exists=True,
        )
        current_pass2_out = str(Path(out_jsonl_path).resolve())
        previous_pass2_out = _meta_get(conn, "pass2_out_jsonl_path", "")
        if previous_pass2_out != current_pass2_out:
            conn.execute("DELETE FROM written_answers")
            _meta_set(conn, "pass2_out_jsonl_path", current_pass2_out)
            conn.commit()
        pass2 = _run_pass2_answers(
            posts_xml_path=posts_xml_path,
            conn=conn,
            alias_to_canonical=alias_to_canonical,
            out_jsonl_path=out_jsonl_path,
            match_mode=match_mode,
            workers=pass2_workers,
            max_records=max_records,
        )
        summary = {
            "posts_xml_path": posts_xml_path,
            "out_jsonl_path": out_jsonl_path,
            "sqlite_index_path": str(db_path),
            "target_packages": len(target_packages),
            "alias_count": len(alias_to_canonical),
            "match_mode": match_mode,
            "pass2_workers": pass2_workers,
            "canonical_to_aliases_sample": dict(list(canonical_to_aliases.items())[:20]),
            "pass1": pass1,
            "pass2": pass2,
        }
        with (work / "stackexchange_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary
    finally:
        conn.close()

