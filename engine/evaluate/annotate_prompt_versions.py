"""Detect and finalize version mentions in PinTrace/Stack Overflow prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from stages.openai_api import call_model, llm_output_to_json

PYTHON_VERSION_RE = re.compile(r"\bpython\s*(?:version\s*)?v?(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)
SPECIFIER_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_.-]*)\s*(===|==|~=|>=|<=|!=|>|<)\s*v?(\d+(?:\.\d+){1,3}(?:[A-Za-z0-9.-]*)?)",
    re.IGNORECASE,
)
VERSION_WORD_RE = re.compile(r"\b(?:version|ver\.?|v)\s*(\d+(?:\.\d+){1,3}(?:[A-Za-z0-9.-]*)?)\b", re.IGNORECASE)
LLM_PROMPT_SCHEMA_VERSION = "pintrace-version-mention-v1"
PYTHON_MINOR_RE = re.compile(
    r"python\s*[23]\s*\.\s*(?:\d+|x)|python[23]\s*\.\s*(?:\d+|x)|python[23]\d{1,2}\b|"
    r"[\\/]python[23]\d{1,2}|python_(?:full_)?version|python_requires",
    re.IGNORECASE,
)
PYTHON_MAJOR_ONLY_RE = re.compile(r"^(?:python\s*[23]|python[23]|py\s*[23]|py[23]|[23])$", re.IGNORECASE)
DEPENDENCY_VERSION_RE = re.compile(
    r"\d+\.\d+|\d+\.x\b|(?:===|==|~=|>=|<=|!=|>|<)\s*v?\d|\bv(?:ersion)?\s*\d|\b[vV]\d+(?:\.\d+)*|v\d+",
    re.IGNORECASE,
)
NON_VERSION_TOKENS = {
    "",
    "unknown",
    "none",
    "none specified",
    "n/a",
    "na",
    "bare",
    "bare mention",
    "unspecified",
    "not specified",
    "no version",
    "implicit",
    "implicit version comparison",
    "implicit via clientsecretcredential",
    "latest",
    "stable",
    "en/stable",
    "master",
    "main",
    "new",
    "fernet",
    "__version__",
    "latest version",
    "clean (no version)",
}


def _load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def detect_mentions(row: dict) -> list[dict]:
    text = "\n".join(
        str(row.get(key) or "")
        for key in ("question_title", "question_text", "question_body", "prompt", "instruct_prompt")
    )
    targets = {str(name).lower().replace("_", "-") for name in row.get("target_packages", [])}
    evidence: list[dict] = []
    for kind, pattern in (
        ("python_version", PYTHON_VERSION_RE),
        ("pep508_specifier", SPECIFIER_RE),
        ("version_expression", VERSION_WORD_RE),
    ):
        for match in pattern.finditer(text):
            if kind == "pep508_specifier" and targets:
                package = match.group(1).lower().replace("_", "-")
                if package not in targets:
                    continue
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            evidence.append({"kind": kind, "match": match.group(0), "context": text[start:end].replace("\n", " ")})
    for target in sorted(targets):
        target_pattern = re.compile(
            rf"\b{re.escape(target)}\b\s*(?:package\s*)?(?:version\s*)?[v=<>~!]*\s*(\d+(?:\.\d+){{1,3}}(?:[A-Za-z0-9.-]*)?)",
            re.IGNORECASE,
        )
        for match in target_pattern.finditer(text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            candidate = {"kind": "target_package_version", "match": match.group(0), "context": text[start:end].replace("\n", " ")}
            if not any(item["match"] == candidate["match"] for item in evidence):
                evidence.append(candidate)
    return evidence


def _task_id(row: dict) -> str:
    value = row.get("record_id") or row.get("task_id") or row.get("question_id")
    if value is None:
        raise ValueError("Dataset row lacks record_id/task_id/question_id")
    return str(value)


def build_llm_prompt(row: dict) -> str:
    """Build the frozen binary-classification prompt used for all PinTrace rows."""
    payload = {
        "task_id": _task_id(row),
        "target_packages": row.get("target_packages", []),
        "question_title": row.get("question_title", ""),
        "question_text": row.get("question_text") or row.get("question_body") or "",
    }
    return f"""You are labeling possible version leakage in a Python dependency benchmark.

Decide whether the input explicitly states either:
1. a Python runtime/interpreter version, or
2. a version, release, pin, range, or version comparison for a target Python dependency.

Count forms such as Python 3.12, python3.10, Python310, package==1.2.3, package v2,
"since package 1.4", or "before version 5". Do not count dates, HTTP/HTML/API/opset
protocol versions, question IDs, ordinary decimal numbers, git branches, documentation
channels such as latest/stable, or versions of unrelated software unless that software
is one of target_packages. A bare package name, import, or API symbol without a version
is clean. Python 3 / python3 / Python3 without a minor version is clean.

Return one JSON object only:
{{"label":"clean|contaminated","mentions":[{{"subject":"python|dependency","name":"...","version_text":"...","quote":"exact short quote"}}],"reason":"one short sentence"}}
For clean inputs, mentions must be an empty list. For contaminated inputs, include at
least one exact quote from the supplied title/text.

Schema: {LLM_PROMPT_SCHEMA_VERSION}
Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"""


def parse_llm_judgment(content: str) -> dict:
    """Validate one structured LLM judgment before it enters the frozen split."""
    value = llm_output_to_json(content)
    if not isinstance(value, dict):
        raise TypeError("LLM response is not a JSON object")
    label = str(value.get("label") or "").strip().lower()
    mentions = value.get("mentions")
    reason = str(value.get("reason") or "").strip()
    if label not in {"clean", "contaminated"}:
        raise ValueError(f"Invalid LLM label: {label!r}")
    if not isinstance(mentions, list) or any(not isinstance(item, dict) for item in mentions):
        raise ValueError("LLM mentions must be a list of objects")
    if label == "clean" and mentions:
        raise ValueError("Clean judgment must not contain mentions")
    if label == "contaminated" and not mentions:
        raise ValueError("Contaminated judgment must contain evidence")
    for mention in mentions:
        if str(mention.get("subject") or "").strip().lower() not in {"python", "dependency"}:
            raise ValueError("Every LLM mention must identify python or dependency")
        if not str(mention.get("version_text") or "").strip():
            raise ValueError("Every LLM mention must contain version_text")
        if not str(mention.get("quote") or "").strip():
            raise ValueError("Every LLM mention must contain an evidence quote")
    return {"label": label, "mentions": mentions, "reason": reason}


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def mention_is_contaminating(mention: dict) -> bool:
    """True only when the mention names a Python 3.x runtime or a dependency version."""
    subject = str(mention.get("subject") or "").strip().lower()
    version_text = _norm_text(mention.get("version_text"))
    quote = _norm_text(mention.get("quote"))
    name = _norm_text(mention.get("name"))
    blob = f"{name} {version_text} {quote}"
    if PYTHON_MINOR_RE.search(blob):
        return True
    if subject == "python":
        if PYTHON_MINOR_RE.search(blob):
            return True
        if PYTHON_MAJOR_ONLY_RE.match(version_text) or PYTHON_MAJOR_ONLY_RE.match(quote):
            return False
        if re.search(r"python\s*[23]\b", blob, re.IGNORECASE) and not PYTHON_MINOR_RE.search(blob):
            return False
        return bool(re.search(r"[23]\.\d", blob))
    if "opset" in blob.lower():
        return False
    if version_text.lower() in NON_VERSION_TOKENS and not DEPENDENCY_VERSION_RE.search(quote):
        return False
    if DEPENDENCY_VERSION_RE.search(blob):
        return True
    if re.fullmatch(r"\d+", version_text) and name and re.search(rf"{re.escape(name)}|{re.escape(version_text)}", quote, re.IGNORECASE):
        return True
    return False


def audit_judgment(judgment: dict) -> dict:
    """Drop bare-name and python3-only mentions before freezing the split."""
    original_mentions = list(judgment.get("mentions") or [])
    kept = [mention for mention in original_mentions if mention_is_contaminating(mention)]
    dropped = [mention for mention in original_mentions if mention not in kept]
    if judgment.get("label") == "contaminated" and not kept:
        return {
            "label": "clean",
            "mentions": [],
            "reason": _norm_text(judgment.get("reason")),
            "audit": "dropped_insufficient_version_evidence",
            "dropped_mentions": dropped,
        }
    result = {**judgment, "mentions": kept}
    if dropped:
        result["audit"] = "filtered_insufficient_version_evidence"
        result["dropped_mentions"] = dropped
    return result


def ground_evidence(mention: dict, source_text: str) -> str | None:
    """Return the strongest source-grounding level for an LLM evidence item."""
    quote = str(mention.get("quote") or "").strip()
    if quote and quote in source_text:
        return "exact_quote"
    normalize = lambda value: " ".join(
        re.sub(r"<[^>]+>", " ", html.unescape(value)).casefold().split()
    )
    normalized_source = normalize(source_text)
    normalized_quote = normalize(quote)
    if normalized_quote and normalized_quote in normalized_source:
        return "normalized_quote"
    version_text = normalize(str(mention.get("version_text") or ""))
    name = normalize(str(mention.get("name") or ""))
    subject = str(mention.get("subject") or "").strip().lower()
    subject_grounded = subject == "python" and "python" in normalized_source
    subject_grounded = subject_grounded or (subject == "dependency" and name and name in normalized_source)
    if version_text and version_text in normalized_source and subject_grounded:
        return "grounded_fields"
    return None


def run_llm(
    dataset: Path,
    output_jsonl: Path,
    *,
    api_url: str,
    api_model: str,
    api_key: str,
    workers: int,
    max_records: int | None,
) -> None:
    """Classify every input with one frozen LLM prompt, with resumable JSONL output."""
    rows = _load_dataset(dataset)
    if max_records is not None:
        rows = rows[:max_records]
    expected_ids = [_task_id(row) for row in rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Dataset contains duplicate task ids")

    completed: set[str] = set()
    if output_jsonl.exists():
        for row in _load_dataset(output_jsonl):
            completed.add(_task_id(row))
    pending = [row for row in rows if _task_id(row) not in completed]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    failures: list[str] = []

    def classify(row: dict) -> dict:
        task_id = _task_id(row)
        prompt = build_llm_prompt(row)
        source_text = "\n".join(
            str(row.get(key) or "")
            for key in ("question_title", "question_text", "question_body")
        )
        message = prompt
        last_error: Exception | None = None
        for _attempt in range(4):
            response = call_model(
                token=api_key,
                api_address=api_url,
                model=api_model,
                message=message,
            )
            content = str(response.get("content") or "")
            try:
                judgment = parse_llm_judgment(content)
                for mention in judgment["mentions"]:
                    grounding = ground_evidence(mention, source_text)
                    if grounding is None:
                        raise ValueError(f"Evidence quote is not present in task {task_id}")
                    mention["evidence_grounding"] = grounding
            except (TypeError, ValueError) as exc:
                last_error = exc
                message = (
                    prompt
                    + "\n\nYour previous JSON was invalid: "
                    + str(exc)
                    + ". A bare package name without a version is clean. "
                    + "Python 3 / python3 / Python3 without a minor version is clean. "
                    + "Contaminated requires Python 3.x or a version/range "
                    + "for a target dependency. Every mention must include non-empty "
                    + "version_text and an exact quote from the supplied title/text. "
                    + "Return one valid JSON object only.\nPrevious output:\n"
                    + content
                )
                continue
            judgment = audit_judgment(judgment)
            return {
                "task_id": task_id,
                **judgment,
                "regex_candidate": bool(detect_mentions(row)),
                "model": api_model,
                "prompt_schema": LLM_PROMPT_SCHEMA_VERSION,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_response": content,
                "usage": response.get("usage") or {},
            }
        raise last_error or RuntimeError(f"LLM labeling failed for {task_id}")

    with (
        output_jsonl.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        futures = {executor.submit(classify, row): _task_id(row) for row in pending}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve partial batch for resume
                failures.append(f"{task_id}: {exc}")
                continue
            with write_lock:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()

    print(json.dumps({"total": len(rows), "already_done": len(completed), "written": len(pending) - len(failures), "failed": len(failures)}))
    if failures:
        raise RuntimeError(f"LLM labeling incomplete; first failure: {failures[0]}")


def finalize_llm(dataset: Path, labels_jsonl: Path, clean_path: Path, contaminated_path: Path) -> None:
    """Freeze a complete one-label-per-task LLM partition."""
    dataset_ids = {_task_id(row) for row in _load_dataset(dataset)}
    labels: dict[str, str] = {}
    for row in _load_dataset(labels_jsonl):
        task_id = _task_id(row)
        label = str(row.get("label") or "").strip().lower()
        if task_id in labels:
            raise ValueError(f"Duplicate LLM label for {task_id}")
        if label not in {"clean", "contaminated"}:
            raise ValueError(f"Invalid LLM label for {task_id}: {label!r}")
        labels[task_id] = label
    missing = sorted(dataset_ids - labels.keys())
    extra = sorted(labels.keys() - dataset_ids)
    if missing or extra:
        raise ValueError(f"LLM labels do not match dataset: missing={len(missing)}, extra={len(extra)}")
    _write_ids(clean_path, [task_id for task_id, label in labels.items() if label == "clean"])
    _write_ids(contaminated_path, [task_id for task_id, label in labels.items() if label == "contaminated"])
    print(json.dumps({"clean": sum(label == "clean" for label in labels.values()), "contaminated": sum(label == "contaminated" for label in labels.values())}))


def apply_evidence_audit(input_jsonl: Path, output_jsonl: Path) -> None:
    """Rewrite LLM labels so bare names and python3-only mentions are clean."""
    audited: list[dict] = []
    flipped = 0
    filtered = 0
    for row in _load_dataset(input_jsonl):
        original_label = str(row.get("llm_label") or row.get("label") or "").strip().lower()
        updated = audit_judgment(
            {
                "label": str(row.get("label") or "").strip().lower(),
                "mentions": list(row.get("mentions") or []),
                "reason": row.get("reason") or "",
            }
        )
        new_row = {
            **row,
            "label": updated["label"],
            "mentions": updated["mentions"],
            "llm_label": original_label,
        }
        if updated.get("audit"):
            new_row["audit"] = updated["audit"]
        if updated.get("dropped_mentions"):
            new_row["dropped_mentions"] = updated["dropped_mentions"]
        if original_label == "contaminated" and updated["label"] == "clean":
            flipped += 1
        elif updated.get("dropped_mentions"):
            filtered += 1
        audited.append(new_row)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"total": len(audited), "flipped_to_clean": flipped, "filtered_mentions": filtered}))


def run_auto(dataset: Path, review_csv: Path, output_jsonl: Path) -> None:
    annotations: list[dict] = []
    for row in _load_dataset(dataset):
        evidence = detect_mentions(row)
        annotations.append(
            {
                "task_id": _task_id(row),
                "auto_label": "contaminated" if evidence else "clean",
                "evidence": evidence,
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            handle.write(json.dumps(annotation, ensure_ascii=False, sort_keys=True) + "\n")
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "auto_label", "evidence", "reviewer_1", "reviewer_2", "final_label", "notes"])
        writer.writeheader()
        for annotation in annotations:
            writer.writerow(
                {
                    "task_id": annotation["task_id"],
                    "auto_label": annotation["auto_label"],
                    "evidence": json.dumps(annotation["evidence"], ensure_ascii=False),
                    "reviewer_1": "",
                    "reviewer_2": "",
                    "final_label": "",
                    "notes": "",
                }
            )
    print(json.dumps({"total": len(annotations), "auto_contaminated": sum(a["auto_label"] == "contaminated" for a in annotations)}))


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task_id in sorted(ids):
            handle.write(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n")


def finalize(review_csv: Path, clean_path: Path, contaminated_path: Path) -> None:
    clean: list[str] = []
    contaminated: list[str] = []
    incomplete: list[str] = []
    disagreements: list[str] = []
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task_id = str(row.get("task_id") or "")
            label = str(row.get("final_label") or "").strip().lower()
            reviewer_1 = str(row.get("reviewer_1") or "").strip().lower()
            reviewer_2 = str(row.get("reviewer_2") or "").strip().lower()
            if reviewer_1 not in {"clean", "contaminated"} or reviewer_2 not in {"clean", "contaminated"}:
                incomplete.append(task_id)
                continue
            if reviewer_1 != reviewer_2 and label not in {"clean", "contaminated"}:
                disagreements.append(task_id)
                continue
            if label not in {"clean", "contaminated"}:
                label = reviewer_1
            (clean if label == "clean" else contaminated).append(task_id)
    if incomplete or disagreements:
        raise ValueError(
            f"Review incomplete: missing double review={len(incomplete)}, unresolved disagreements={len(disagreements)}"
        )
    _write_ids(clean_path, clean)
    _write_ids(contaminated_path, contaminated)
    print(json.dumps({"clean": len(clean), "contaminated": len(contaminated)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--clean-task-ids", type=Path)
    parser.add_argument("--contaminated-task-ids", type=Path)
    parser.add_argument("--llm", action="store_true", help="Classify the full dataset with an LLM.")
    parser.add_argument("--finalize-llm", action="store_true", help="Freeze a complete LLM-labeled split.")
    parser.add_argument("--llm-labels", type=Path)
    parser.add_argument("--audit-llm", action="store_true", help="Drop bare-name and python3-only mentions.")
    parser.add_argument("--audited-labels", type=Path)
    parser.add_argument("--api-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize_llm:
        if not args.dataset or not args.llm_labels or not args.clean_task_ids or not args.contaminated_task_ids:
            raise ValueError("--finalize-llm requires --dataset, --llm-labels and both task-id outputs")
        finalize_llm(args.dataset, args.llm_labels, args.clean_task_ids, args.contaminated_task_ids)
    elif args.audit_llm:
        if not args.llm_labels:
            raise ValueError("--audit-llm requires --llm-labels")
        apply_evidence_audit(args.llm_labels, args.audited_labels or args.llm_labels)
    elif args.llm:
        if not args.dataset or not args.llm_labels or not args.api_url or not args.api_model:
            raise ValueError("--llm requires --dataset, --llm-labels, --api-url and --api-model")
        api_key = args.api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        run_llm(
            args.dataset,
            args.llm_labels,
            api_url=args.api_url,
            api_model=args.api_model,
            api_key=api_key,
            workers=args.workers,
            max_records=args.max_records,
        )
    elif args.finalize:
        if not args.review_csv or not args.clean_task_ids or not args.contaminated_task_ids:
            raise ValueError("--finalize requires --review-csv and both task-id outputs")
        finalize(args.review_csv, args.clean_task_ids, args.contaminated_task_ids)
    else:
        if not args.dataset or not args.output_jsonl or not args.review_csv:
            raise ValueError("auto mode requires --dataset, --output-jsonl and --review-csv")
        run_auto(args.dataset, args.review_csv, args.output_jsonl)


if __name__ == "__main__":
    main()
