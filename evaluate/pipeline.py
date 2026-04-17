"""Unified end-to-end pipeline runner for D1 and D2.

Usage:
    python -m evaluate.pipeline --track d1 ...
    python -m evaluate.pipeline --track d2 ...

Optional email on successful completion (see scripts/notifier_utils.py): set
EXPERIMENT_NOTIFY_ENABLED=1 and SMTP/NOTIFY_* environment variables.
Also pass --email to enable sending for this run.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from loguru import logger

from paths import BIGCODEBENCH, OSV_INDEX, OUTPUTS, XDG_CACHE_DIR, ensure_dirs
from stages.compact_checker import check_records_compatibility, load_bigcodebench_problems
from stages.metrics import aggregate_metrics, save_metrics
from stages.openrouter_api import OPENROUTER_BASE
from stages.utils import load_mapping
from stages.version_resolver import (
    extract_d2_code_and_tpl_versions,
    extract_tpl_versions_from_llm_output,
    resolve_records_versions,
)
from stages.vuln_checker import analyze_records_vulnerabilities, load_osv_index

from evaluate.inference import (
    LLMConfig,
    _call_llm,
    filter_third_party,
    load_d2_records_jsonl,
    load_dataset_split,
    run_d1_inference,
    run_d2_inference,
)
from scripts.notifier_utils import send_experiment_notification


def read_dotenv(path: str = ".env") -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    parsed = dotenv_values(path)
    return {k: v for k, v in parsed.items() if isinstance(v, str)}


def save_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_extraction(records: list[dict], *, top_n: int = 30) -> dict:
    total_imports = 0
    specified = 0
    unspecified = 0
    per_lib_counter = Counter()
    per_lib_specified = Counter()
    for rec in records:
        for item in rec.get("extracted_versions", []):
            lib = item.get("library")
            if not lib:
                continue
            total_imports += 1
            per_lib_counter[lib] += 1
            if item.get("version") is None:
                unspecified += 1
            else:
                specified += 1
                per_lib_specified[lib] += 1
    return {
        "total_tasks": len(records),
        "total_imports": total_imports,
        "specified_versions": specified,
        "unspecified_versions": unspecified,
        "version_spec_rate": specified / total_imports if total_imports else 0.0,
        "top_libraries": per_lib_counter.most_common(top_n),
        "top_libraries_with_specified_versions": per_lib_specified.most_common(top_n),
    }


def summarize_resolution(records: list[dict]) -> dict:
    total = 0
    method_counter = Counter()
    unresolved = 0
    missing_specified = 0
    for rec in records:
        for lib in rec.get("per_lib", []):
            total += 1
            method_counter[lib.get("resolution_method", "unknown")] += 1
            if not lib.get("resolved_version"):
                unresolved += 1
            if lib.get("specified_version") and not lib.get("version_exists", True):
                missing_specified += 1
    return {
        "total_resolved_items": total,
        "resolution_method_counts": dict(method_counter),
        "unresolved_items": unresolved,
        "specified_but_nonexistent_versions": missing_specified,
    }


def summarize_vulnerability(records: list[dict], *, top_n: int = 30) -> dict:
    total_checked = 0
    vulnerable = 0
    severity_counter = Counter()
    vuln_pkg_counter = Counter()
    for rec in records:
        for item in rec.get("vuln_findings", []):
            total_checked += 1
            if item.get("is_vulnerable"):
                vulnerable += 1
                if item.get("max_severity"):
                    severity_counter[item["max_severity"]] += 1
                vuln_pkg_counter[item.get("pypi_name", "unknown")] += 1
    return {
        "total_checked_versions": total_checked,
        "vulnerable_versions": vulnerable,
        "vuln_rate": vulnerable / total_checked if total_checked else 0.0,
        "severity_distribution": dict(severity_counter),
        "top_vulnerable_packages": vuln_pkg_counter.most_common(top_n),
    }


def summarize_compatibility(records: list[dict], *, with_bcb: bool) -> dict:
    total = 0
    griffe_runtime_errors = 0
    ty_stats = {"compatible": 0, "errors": 0, "rule_counter": Counter()}
    bcb_stats = {"total": 0, "pass": 0, "fail": 0, "error_like": 0, "status_counter": Counter()}
    for rec in records:
        compat = rec.get("compat_results")
        if not isinstance(compat, dict):
            continue
        total += 1
        ty_result = compat.get("ty", {})
        if ty_result.get("is_compatible"):
            ty_stats["compatible"] += 1
        errors = ty_result.get("errors", [])
        ty_stats["errors"] += len(errors)
        for err in errors:
            rule = err.get("rule") or "unknown"
            ty_stats["rule_counter"][rule] += 1
            msg = str(err.get("message", ""))
            if "griffe" in msg.lower():
                griffe_runtime_errors += 1
        if with_bcb:
            bcb = compat.get("bcb_test")
            if isinstance(bcb, dict):
                bcb_stats["total"] += 1
                status = str(bcb.get("status", "unknown"))
                bcb_stats["status_counter"][status] += 1
                if status == "pass":
                    bcb_stats["pass"] += 1
                elif status == "fail":
                    bcb_stats["fail"] += 1
                else:
                    bcb_stats["error_like"] += 1
    out: dict = {
        "total_compat_checks": total,
        "ty": {
            "compatible_checks": ty_stats["compatible"],
            "compat_rate": ty_stats["compatible"] / total if total else 0.0,
            "error_count": ty_stats["errors"],
            "top_error_rules": ty_stats["rule_counter"].most_common(30),
        },
        "griffe_runtime_error_count": griffe_runtime_errors,
    }
    if with_bcb and bcb_stats["total"]:
        out["bcb_test"] = {
            "total": bcb_stats["total"],
            "pass": bcb_stats["pass"],
            "fail": bcb_stats["fail"],
            "error_like": bcb_stats["error_like"],
            "pass_rate": bcb_stats["pass"] / bcb_stats["total"] if bcb_stats["total"] else 0.0,
            "status_distribution": dict(bcb_stats["status_counter"]),
        }
    return out


def add_compatibility_results(
    records: list[dict],
    *,
    checkpoint_path: str,
    output_path: str,
    python_version: str,
    with_bcb: bool,
    bigcodebench_path: str | None = None,
    bcb_timeout_seconds: int = 180,
    compat_checkpoint_every: int = 20,
) -> list[dict]:
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    processed: dict[str, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        processed = {r["task_id"]: r for r in saved.get("records", [])}
        logger.info(f"Resumed compat checkpoint: {len(processed)} done.")
    pending = [r for r in records if r.get("task_id") not in processed]
    logger.info(f"Compatibility pending records: {len(pending)}")

    bcb_problems = {}
    project_root = str(Path(__file__).resolve().parents[1])
    if with_bcb and bigcodebench_path:
        bcb_problems = load_bigcodebench_problems(bigcodebench_path)

    if pending:
        if compat_checkpoint_every <= 0:
            compat_checkpoint_every = 20
        for start in range(0, len(pending), compat_checkpoint_every):
            end = min(start + compat_checkpoint_every, len(pending))
            chunk = pending[start:end]
            enriched = check_records_compatibility(
                chunk,
                python_version=python_version,
                bcb_problems=bcb_problems if with_bcb else None,
                bcb_timeout_seconds=bcb_timeout_seconds,
                project_root=project_root,
            )
            for rec in enriched:
                processed[rec["task_id"]] = rec
            save_json(checkpoint_path, {"records": list(processed.values())})

    final_records = [processed[r["task_id"]] for r in records if r.get("task_id") in processed]
    save_json(output_path, final_records)
    return final_records


def extract_d2_records(llm_output_path: str, *, max_records: int | None, python_version: str) -> list[dict]:
    mapping, _ = load_mapping()
    records: list[dict] = []
    with open(llm_output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record_id = obj.get("record_id", "")
            llm_output = obj.get("llm_output", "")
            code, extracted_versions = extract_d2_code_and_tpl_versions(
                llm_output,
                python_version=python_version,
                mapping=mapping,
            )
            records.append(
                {
                    "task_id": record_id,
                    "record_id": record_id,
                    "question_id": obj.get("question_id"),
                    "answer_id": obj.get("answer_id"),
                    "question_title": obj.get("question_title"),
                    "expected_libs": obj.get("target_packages", []),
                    "prompt_mode": obj.get("prompt_mode", ""),
                    "pinning_mode": obj.get("pinning_mode", "inline"),
                    "match_mode": obj.get("match_mode", ""),
                    "reference_code": obj.get("reference_code", ""),
                    "code": code,
                    "extracted_versions": extracted_versions,
                }
            )
            if max_records is not None and len(records) >= max_records:
                break
    return records


def _infer_model_name(llm_output_path: str) -> str:
    try:
        with open(llm_output_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first:
            obj = json.loads(first)
            model = (obj.get("llm_config") or {}).get("api_model")
            if model:
                return str(model)
    except Exception:  # noqa: BLE001
        pass
    return Path(llm_output_path).stem


def _is_d1_output_valid(record: dict) -> bool:
    llm_output = record.get("llm_output")
    if not isinstance(llm_output, dict):
        return False
    content = llm_output.get("content")
    return isinstance(content, str) and bool(content.strip())


def _is_d2_output_valid(record: dict) -> bool:
    content = record.get("llm_output")
    return isinstance(content, str) and bool(content.strip())


def _load_ablation_done_ids(output_path: Path, *, track: str) -> set[str]:
    if not output_path.exists():
        return set()
    id_field = "task_id" if track == "d1" else "record_id"
    out: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = obj.get(id_field)
            if not isinstance(record_id, str) or not record_id:
                continue
            if track == "d1" and _is_d1_output_valid(obj):
                out.add(record_id)
            if track == "d2" and _is_d2_output_valid(obj):
                out.add(record_id)
    return out


def _load_ablation_prompt_items(path: str, *, track: str) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if track == "d1":
                task_id = obj.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    task_id = f"ablation-d1-{idx}"
                obj["task_id"] = task_id
            else:
                record_id = obj.get("record_id")
                if not isinstance(record_id, str) or not record_id:
                    record_id = f"ablation-d2-{idx}"
                obj["record_id"] = record_id
            items.append(obj)
    return items


def run_ablation_inference(
    *,
    track: str,
    prompts_jsonl: str,
    output_path: Path,
    config: LLMConfig,
    workers: int,
    max_examples: int | None,
    python_version: str,
    prompt_mode: str,
    pinning_mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = _load_ablation_prompt_items(prompts_jsonl, track=track)
    id_field = "task_id" if track == "d1" else "record_id"
    done_ids = _load_ablation_done_ids(output_path, track=track)
    todo = [item for item in items if item[id_field] not in done_ids]
    if max_examples is not None:
        todo = todo[:max_examples]
    logger.info(
        f"Ablation inference: track={track} total={len(items)} done={len(done_ids)} todo={len(todo)} "
        f"model={config.api_model} workers={workers}"
    )
    if not todo:
        return

    write_lock = threading.Lock()
    written = 0
    failed = 0

    def _call_item(item: dict) -> dict:
        prompt = item["prompt"]
        result = _call_llm(prompt=prompt, config=config)
        if track == "d1":
            return {
                "llm_config": {
                    "api_url": config.api_url,
                    "api_model": config.api_model,
                    "python_version": python_version,
                },
                "task_id": item["task_id"],
                "instruct_prompt": prompt,
                "augmented_prompt": prompt,
                "third_party_libs": item.get("tpls", []),
                "llm_output": result,
            }

        out = {
            "record_id": item["record_id"],
            "question_id": item.get("question_id"),
            "answer_id": item.get("answer_id"),
            "question_title": item.get("question_title"),
            "target_packages": item.get("tpls", []),
            "prompt_mode": item.get("context_mode") or item.get("prompt_mode") or prompt_mode,
            "pinning_mode": pinning_mode,
            "match_mode": item.get("match_mode", ""),
            "reference_code": item.get("reference_code", ""),
            "prompt": prompt,
            "llm_output": result.get("content", ""),
            "usage": result.get("usage", {}),
            "response_time": result.get("response_time", 0.0),
            "llm_config": {
                "api_url": config.api_url,
                "api_model": config.api_model,
            },
        }
        return out

    with output_path.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_call_item, item): item[id_field] for item in todo}
            for future in as_completed(futures):
                record_id = futures[future]
                try:
                    record = future.result()
                    with write_lock:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()
                        written += 1
                        logger.info(f"[{written}/{len(todo)}] wrote {id_field}={record_id}")
                except Exception as exc:  # noqa: BLE001
                    with write_lock:
                        failed += 1
                    logger.error(f"Failed {id_field}={record_id}: {exc}")
    logger.success(f"Ablation inference finished: written={written}, failed={failed}, output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified D1/D2 end-to-end TPL security pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--track", choices=["d1", "d2"], required=True)
    parser.add_argument("--inference-mode", choices=["standard", "ablation"], default="standard")
    parser.add_argument("--ablation-prompts-jsonl", default=None, help="Ablation prompt jsonl path (uses each line's 'prompt').")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="v0.1.4")
    parser.add_argument("--filter-third-party", action="store_true", help="Whether to filter out records with third-party imports (D1 only).")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--prompt-mode", default=None, help="d1: pinning mode; d2: context mode.")
    parser.add_argument("--pinning-mode", default="inline", help="D2 pinning mode.")
    parser.add_argument("--api-url", "--api-base-url", dest="api_url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-mode", default="openai", choices=["openai", "openrouter", "openwebui"])
    parser.add_argument("--provider-order", default=None, help="Comma-separated provider order for OpenRouter API mode (e.g. 'openai,together').")
    parser.add_argument("--provider-allow-fallbacks", action="store_true", help="Whether to allow provider fallbacks (only for openrouter with provider order).")
    parser.add_argument("--http-referer", default=None, help="Optional HTTP-Referer header for OpenRouter API calls.")
    parser.add_argument("--site-title", default=None, help="Optional title to include in X-OpenRouter-Title header for OpenRouter API calls.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-hint-chars", type=int, default=1200)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for concurrent processing.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--reference-date", default="2026-02-01T00:00:00+00:00")
    parser.add_argument("--llm-output-path", default=None)
    parser.add_argument("--outputs-d1", default=None)
    parser.add_argument("--outputs-d2", default=None)
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--osv-index-path", default=None, help="Path to OSV index JSON file (for vulnerability analysis). Defaults to loading from OSV_INDEX path.")
    parser.add_argument("--xdg-cache-path", default=None, help="Path to XDG cache directory. Defaults to XDG_CACHE_DIR environment variable or ~/.cache.")
    parser.add_argument("--bigcodebench-path", default=None, help="Path to BigCodeBench dataset. Defaults to BIGCODEBENCH path.")
    parser.add_argument("--bcb-timeout", type=int, default=180, help="Timeout for BigCodeBench evaluations.")
    parser.add_argument("--compat-checkpoint-every", type=int, default=20, help="Interval for compatibility checkpoints.")
    parser.add_argument("--skip-vuln", action="store_true", help="Skip vulnerability analysis.")
    parser.add_argument("--skip-compat", action="store_true", help="Skip compatibility checks.")
    parser.add_argument("--email", action="store_true", help="Send completion email notification.")
    return parser.parse_args()


def _notify_evaluate_pipeline_done(
    *,
    args: argparse.Namespace,
    run_name_base: str,
    run_dir: Path,
    llm_output_path: str | None,
    started_at_utc: datetime,
    stages: set[int],
    summary: dict,
) -> None:
    if not args.email:
        logger.info("[notify] skipped (pass --email to enable)")
        return

    duration_sec = round((datetime.now(timezone.utc) - started_at_utc).total_seconds(), 2)
    notify_summary: dict[str, Any] = {
        "pipeline": "evaluate.pipeline",
        "track": args.track,
        "run_name": run_name_base,
        "run_dir": str(run_dir),
        "stages": sorted(stages),
        "duration_seconds": duration_sec,
        "llm_output_path": str(Path(llm_output_path).resolve()) if llm_output_path else None,
    }
    if summary:
        notify_summary["metrics_summary"] = summary
    sent, msg = send_experiment_notification(
        experiment_name=f"evaluate.pipeline {args.track} {run_name_base}",
        status="SUCCESS",
        summary=notify_summary,
    )
    if sent:
        logger.success(f"[notify] {msg}")
    else:
        logger.info(f"[notify] {msg}")


def main() -> None:
    args = parse_args()
    started_at_utc = datetime.now(timezone.utc)
    dotenv = read_dotenv(".env")
    ensure_dirs()
    runtime_cache_dir = Path(args.xdg_cache_path) if args.xdg_cache_path else XDG_CACHE_DIR
    runtime_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_cache_dir))
    os.environ.setdefault("HF_HOME", str(runtime_cache_dir / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(runtime_cache_dir / "hf" / "hub"))

    api_model = args.api_model or dotenv.get("OPENAI_OVERALL_MODEL", "gpt-4o-mini")
    api_url = args.api_url or dotenv.get("OPENAI_BASE_URL", "")
    api_key = (
        args.api_key
        or dotenv.get("API_KEY", "")
        or dotenv.get("OPENAI_API_KEY", "")
        or dotenv.get("CHATBOX_TOKEN", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    stages = set(args.stages)
    if args.inference_mode == "ablation" and not args.ablation_prompts_jsonl:
        raise ValueError("Ablation mode requires --ablation-prompts-jsonl")
    if args.track == "d1":
        prompt_mode = args.prompt_mode or "inline"
        run_name_base = args.run_name or f"{api_model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        outputs_root = Path(args.outputs_d1).resolve() if args.outputs_d1 else (OUTPUTS / "d1")
        run_dir_base = outputs_root / prompt_mode / run_name_base
        py_tag = f"py{str(args.python_version).replace('.', '')}"
        run_dir = run_dir_base / py_tag
        run_dir_base.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        llm_output_path = args.llm_output_path or str(run_dir_base / f"{api_model.replace('/', '_')}_{prompt_mode}.jsonl")
        with_bcb = True
        log_path = run_dir / "pipeline_d1.log"
    else:
        prompt_mode = args.prompt_mode or "blind"
        pinning_mode = args.pinning_mode
        model_slug = str(api_model).replace("/", "_").replace(":", "_")
        llm_stem = Path(args.llm_output_path).stem if args.llm_output_path else f"{model_slug}_{prompt_mode}"
        pin_suffix = "requirements_txt" if pinning_mode == "requirements.txt" else pinning_mode
        run_name_base = args.run_name or f"d2_{llm_stem}_{pin_suffix}"
        outputs_root = Path(args.outputs_d2).resolve() if args.outputs_d2 else (OUTPUTS / "d2")
        fresh_run_dir_base = outputs_root / pinning_mode / run_name_base
        default_llm_output = str(fresh_run_dir_base / f"{model_slug}_{prompt_mode}.jsonl")
        llm_output_path = args.llm_output_path or default_llm_output
        if args.llm_output_path:
            run_dir_base = Path(llm_output_path).expanduser().resolve().parent
            if not args.run_name:
                run_name_base = run_dir_base.name
        else:
            run_dir_base = fresh_run_dir_base
        py_tag = f"py{str(args.python_version).replace('.', '')}"
        run_dir = run_dir_base / py_tag
        run_dir_base.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        with_bcb = False
        log_path = run_dir / "pipeline_d2.log"

    logger.add(log_path)
    paper_dir = run_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    stage_timeline: list[dict] = []
    run_manifest = {
        "track": args.track,
        "run_name": run_name_base,
        "run_dir_base": str(run_dir_base),
        "run_dir": str(run_dir),
        "started_at": now_iso(),
        "llm_output_path": str(Path(llm_output_path).resolve()) if llm_output_path else None,
        "all_the_args": {k: v for k, v in args.__dict__.items() if k not in ["api_key"]},
    }
    save_json(str(paper_dir / "run_manifest.json"), run_manifest)
    extraction_summary: dict = {}
    resolution_summary: dict = {}
    vuln_summary: dict = {}
    compat_summary: dict = {}
    summary: dict = {}

    def _needs_stage_output(stage_num: int) -> bool:
        return any(s in stages for s in range(stage_num + 1, 7))

    if 1 in stages:
        stage_start = time.perf_counter()
        if args.inference_mode == "ablation":
            run_ablation_inference(
                track=args.track,
                prompts_jsonl=args.ablation_prompts_jsonl,
                output_path=Path(llm_output_path),
                config=LLMConfig(
                    api_url=api_url,
                    api_key=api_key,
                    api_model=api_model,
                    timeout=args.timeout,
                    api_mode=args.api_mode,
                    provider={
                        "order": [p.strip() for p in args.provider_order.split(",") if p.strip()],
                        "allow_fallbacks": args.provider_allow_fallbacks,
                    }
                    if args.api_mode == "openrouter" and args.provider_order
                    else None,
                    http_referer=args.http_referer,
                ),
                workers=args.workers,
                max_examples=args.max_examples,
                python_version=args.python_version,
                prompt_mode=prompt_mode,
                pinning_mode=args.pinning_mode,
            )
        elif args.track == "d1":
            dataset = load_dataset_split(args.dataset or "bigcode/bigcodebench", args.split)
            if args.filter_third_party:
                dataset = filter_third_party(dataset, args.python_version)
            run_d1_inference(
                dataset=dataset,
                config=LLMConfig(
                    api_url=api_url,
                    api_key=api_key,
                    api_model=api_model,
                    timeout=args.timeout,
                    api_mode=args.api_mode,
                ),
                output_path=Path(llm_output_path),
                max_examples=args.max_examples,
                prompt_mode=prompt_mode,
                python_version=args.python_version,
                max_workers=args.workers,
            )
        else:
            if not args.dataset:
                raise ValueError("D2 stage 1 requires --dataset")
            if not api_url and args.api_mode == "openrouter":
                api_url = OPENROUTER_BASE
            if not api_url:
                raise ValueError("D2 stage 1 requires --api-url (or openrouter default)")
            provider = None
            if args.api_mode == "openrouter" and args.provider_order:
                provider = {
                    "order": [p.strip() for p in args.provider_order.split(",") if p.strip()],
                    "allow_fallbacks": args.provider_allow_fallbacks,
                }
            records = load_d2_records_jsonl(args.dataset)
            run_d2_inference(
                records=records,
                output_path=Path(llm_output_path),
                config=LLMConfig(
                    api_url=api_url,
                    api_key=api_key,
                    api_model=api_model,
                    timeout=args.timeout,
                    api_mode=args.api_mode,
                    provider=provider,
                    http_referer=args.http_referer
                ),
                mode=prompt_mode,
                pinning_mode=args.pinning_mode,
                max_hint_chars=args.max_hint_chars,
                workers=args.workers,
                max_examples=args.max_examples,
            )
        stage_timeline.append(
            {
                "stage": "M1_inference",
                "status": "done",
                "duration_ms": round((time.perf_counter() - stage_start) * 1000, 2),
            }
        )
    else:
        if _needs_stage_output(1) and not os.path.exists(llm_output_path):
            raise ValueError(f"Stage 1 skipped but llm output not found: {llm_output_path}")
        stage_timeline.append({"stage": "M1_inference", "status": "skipped", "duration_ms": 0.0})

    if 2 in stages:
        stage_start = time.perf_counter()
        if args.track == "d1":
            extracted_records, _ = extract_tpl_versions_from_llm_output(llm_output_path)
            if args.max_records is not None:
                extracted_records = extracted_records[: args.max_records]
        else:
            extracted_records = extract_d2_records(
                llm_output_path,
                max_records=args.max_records,
                python_version=args.python_version,
            )
        extracted_path = str(run_dir / "m2_extracted_records.json")
        save_json(extracted_path, extracted_records)
        extraction_summary = summarize_extraction(extracted_records, top_n=20 if args.track == "d1" else 30)
        save_json(str(paper_dir / "m2_extraction_summary.json"), extraction_summary)
        stage_timeline.append(
            {
                "stage": "M2_extraction",
                "status": "done",
                "duration_ms": round((time.perf_counter() - stage_start) * 1000, 2),
                "summary": extraction_summary,
            }
        )
    else:
        if _needs_stage_output(2):
            extracted_path = str(run_dir / "m2_extracted_records.json")
            if not os.path.exists(extracted_path):
                raise ValueError(f"Stage 2 skipped but file missing: {extracted_path}")
            extracted_records = load_records(extracted_path)
            if args.max_records is not None:
                extracted_records = extracted_records[: args.max_records]
            extraction_summary = summarize_extraction(extracted_records, top_n=20 if args.track == "d1" else 30)
            save_json(str(paper_dir / "m2_extraction_summary.json"), extraction_summary)
        stage_timeline.append({"stage": "M2_extraction", "status": "skipped", "duration_ms": 0.0, "summary": extraction_summary})

    if 3 in stages:
        stage_start = time.perf_counter()
        resolved_records = resolve_records_versions(
            extracted_records,
            reference_date=args.reference_date,
            checkpoint_path=str(run_dir / "checkpoints" / "version_resolution.json"),
            output_path=str(run_dir / "m3_resolved_records.json"),
            max_workers=args.workers,
        )
        resolution_summary = summarize_resolution(resolved_records)
        save_json(str(paper_dir / "m3_resolution_summary.json"), resolution_summary)
        stage_timeline.append(
            {
                "stage": "M3_resolution",
                "status": "done",
                "duration_ms": round((time.perf_counter() - stage_start) * 1000, 2),
                "summary": resolution_summary,
            }
        )
    else:
        if _needs_stage_output(3):
            resolved_path = str(run_dir / "m3_resolved_records.json")
            if not os.path.exists(resolved_path):
                raise ValueError(f"Stage 3 skipped but file missing: {resolved_path}")
            resolved_records = load_records(resolved_path)
            resolution_summary = summarize_resolution(resolved_records)
            save_json(str(paper_dir / "m3_resolution_summary.json"), resolution_summary)
        stage_timeline.append({"stage": "M3_resolution", "status": "skipped", "duration_ms": 0.0, "summary": resolution_summary})

    if 4 in stages:
        if args.skip_vuln:
            vuln_records = resolved_records
            vuln_summary = summarize_vulnerability(vuln_records, top_n=20 if args.track == "d1" else 30)
            save_json(str(paper_dir / "m4_vulnerability_summary.json"), vuln_summary)
            stage_timeline.append({"stage": "M4_vulnerability", "status": "skipped", "duration_ms": 0.0, "summary": vuln_summary})
        else:
            stage_start = time.perf_counter()
            osv_index = load_osv_index(args.osv_index_path or str(OSV_INDEX))
            vuln_records = analyze_records_vulnerabilities(
                resolved_records,
                osv_index=osv_index,
                cve_cache_path=str(run_dir / "cache" / "cve_cache.json"),
                checkpoint_path=str(run_dir / "checkpoints" / "vuln.json"),
                output_path=str(run_dir / "m4_vuln_records.json"),
                max_workers=args.workers,
            )
            vuln_summary = summarize_vulnerability(vuln_records, top_n=20 if args.track == "d1" else 30)
            save_json(str(paper_dir / "m4_vulnerability_summary.json"), vuln_summary)
            stage_timeline.append(
                {
                    "stage": "M4_vulnerability",
                    "status": "done",
                    "duration_ms": round((time.perf_counter() - stage_start) * 1000, 2),
                    "summary": vuln_summary,
                }
            )
    else:
        if _needs_stage_output(4):
            vuln_path = str(run_dir / "m4_vuln_records.json")
            if not os.path.exists(vuln_path):
                raise ValueError(f"Stage 4 skipped but file missing: {vuln_path}")
            vuln_records = load_records(vuln_path)
            vuln_summary = summarize_vulnerability(vuln_records, top_n=20 if args.track == "d1" else 30)
            save_json(str(paper_dir / "m4_vulnerability_summary.json"), vuln_summary)
        stage_timeline.append({"stage": "M4_vulnerability", "status": "skipped", "duration_ms": 0.0, "summary": vuln_summary})

    if 5 in stages:
        if args.skip_compat:
            final_records = vuln_records
            compat_summary = summarize_compatibility(final_records, with_bcb=with_bcb)
            save_json(str(paper_dir / "m5_compatibility_summary.json"), compat_summary)
            stage_timeline.append({"stage": "M5_compatibility", "status": "skipped", "duration_ms": 0.0, "summary": compat_summary})
        else:
            stage_start = time.perf_counter()
            bcb_path = args.bigcodebench_path
            if with_bcb and bcb_path is None:
                bcb_path = str(BIGCODEBENCH)
            final_records = add_compatibility_results(
                vuln_records,
                checkpoint_path=str(run_dir / "checkpoints" / "compat.json"),
                output_path=str(run_dir / "m5_compat_records.json"),
                python_version=args.python_version,
                with_bcb=with_bcb,
                bigcodebench_path=bcb_path,
                bcb_timeout_seconds=args.bcb_timeout,
                compat_checkpoint_every=args.compat_checkpoint_every,
            )
            compat_summary = summarize_compatibility(final_records, with_bcb=with_bcb)
            save_json(str(paper_dir / "m5_compatibility_summary.json"), compat_summary)
            stage_timeline.append(
                {
                    "stage": "M5_compatibility",
                    "status": "done",
                    "duration_ms": round((time.perf_counter() - stage_start) * 1000, 2),
                    "summary": compat_summary,
                }
            )
    else:
        if _needs_stage_output(5):
            compat_path = str(run_dir / "m5_compat_records.json")
            if not os.path.exists(compat_path):
                raise ValueError(f"Stage 5 skipped but file missing: {compat_path}")
            final_records = load_records(compat_path)
            compat_summary = summarize_compatibility(final_records, with_bcb=with_bcb)
            save_json(str(paper_dir / "m5_compatibility_summary.json"), compat_summary)
        stage_timeline.append({"stage": "M5_compatibility", "status": "skipped", "duration_ms": 0.0, "summary": compat_summary})

    if 6 in stages:
        stage_start = time.perf_counter()
        model_name = api_model if 1 in stages else _infer_model_name(llm_output_path)
        summary = aggregate_metrics(final_records, model_name=model_name)
        save_metrics(summary, json_path=str(run_dir / "metrics_summary.json"))
        stage_timeline.append(
            {
                "stage": "M6_metrics",
                "status": "done",
                "duration_ms": round((time.perf_counter() - stage_start) * 1000, 2),
                "summary": summary,
            }
        )
    else:
        if _needs_stage_output(6):
            metrics_path = str(run_dir / "metrics_summary.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
        stage_timeline.append({"stage": "M6_metrics", "status": "skipped", "duration_ms": 0.0, "summary": summary})

    save_json(str(paper_dir / "stage_timeline.json"), stage_timeline)
    save_json(
        str(paper_dir / "analysis_ready_bundle.json"),
        {
            "run_manifest": run_manifest,
            "stage_timeline": stage_timeline,
            "extraction_summary": extraction_summary,
            "resolution_summary": resolution_summary,
            "vulnerability_summary": vuln_summary,
            "compatibility_summary": compat_summary,
            "metrics_summary": summary,
        },
    )
    logger.success(f"Pipeline completed: track={args.track}, stages={sorted(stages)}")
    _notify_evaluate_pipeline_done(
        args=args,
        run_name_base=run_name_base,
        run_dir=run_dir,
        llm_output_path=llm_output_path,
        started_at_utc=started_at_utc,
        stages=stages,
        summary=summary,
    )


if __name__ == "__main__":
    main()

