#!/usr/bin/env python
"""
Batch runner for D2 pipeline from JSONL job configs.

Each JSONL line is one job. The job object can include pipeline_d2 args, e.g.:

    {"run_name":"qwen_blind","stages":[1,2,3,4,5,6],"dataset":"outputs/dataset_builder/so.jsonl","api_mode":"openrouter","api_model":"Qwen/Qwen3-235B-A22B-Instruct-2507","api_url":"https://openrouter.ai/api/v1","api_key":"sk-xxx","prompt_mode":"blind","pinning_mode":"inline","workers":8,"outputs_d2":"outputs/d2","python_version":"3.12","timeout":120}
    {"run_name":"d2_Qwen_blind_inline","stages":[2,3,5,6],"llm_output_path":"outputs/d2/d2_Qwen3-235B-A22B-Instruct-2507_blind_inline/Qwen3-235B-A22B-Instruct-2507_blind.jsonl","pinning_mode":"inline","skip_vuln":true,"workers":8,"outputs_d2":"outputs/d2","python_version":"3.12"}

Run:
    python -m scripts.batch_pipeline_d2 --jobs-jsonl path/to/jobs.jsonl --parallel 4

Per-job logs are named like the experiment run plus Python tag, e.g.
``d2_Qwen3-235B-A22B-Instruct-2507_blind_inline_no_vuln_py312.log`` (from ``python_version``,
default 3.12). Stem follows ``evaluate.d2.pipeline_d2`` default ``run_name`` when ``run_name``
is omitted; otherwise the configured ``run_name`` is used. Duplicate stems in one batch get
``__L<line>`` suffix.

Console and ``batch_pipeline_d2.log`` lines are prefixed with ``[pin=… ctx=…]``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.notifier_utils import send_experiment_notification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "batch_pipeline_d2"


# Keys supported by evaluate.d2.pipeline_d2 CLI.
SUPPORTED_KEYS = {
    "dataset",
    "llm_output_path",
    "run_name",
    "outputs_d2",
    "stages",
    "api_url",
    "api_key",
    "api_model",
    "api_mode",
    "provider_order",
    "provider_allow_fallbacks",
    "http_referer",
    "site_title",
    "timeout",
    "prompt_mode",
    "pinning_mode",
    "max_hint_chars",
    "max_examples",
    "reference_date",
    "workers",
    "max_records",
    "xdg_cache_path",
    "skip_vuln",
    "skip_compat",
    "python_version",
}


_RUN_NAME_PLACEHOLDER = re.compile(r"^job_\d+$", re.IGNORECASE)


@dataclass
class Job:
    idx: int
    config: dict[str, Any]
    name: str
    log_stem: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run evaluate.d2.pipeline_d2 from JSONL configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--jobs-jsonl",
        type=Path,
        required=True,
        help="JSONL file; each line is one D2 pipeline job config.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of jobs to run concurrently.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to start evaluate.d2.pipeline_d2.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root as subprocess cwd.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Per-job stdout/stderr log directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling new jobs after first failure (parallel mode only affects pending queue).",
    )
    return parser.parse_args()


def _sanitize_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in text)[:120]


def _pinning_fs(config: dict[str, Any]) -> str:
    """Path segment for pinning_mode (default inline)."""
    raw = config.get("pinning_mode") or "inline"
    return _sanitize_name(str(raw).replace(".", "_"))


def _context_mode(config: dict[str, Any]) -> str:
    """prompt_mode in pipeline = blind | hint (context)."""
    return str(config.get("prompt_mode") or "blind")


def _job_mode_tag(config: dict[str, Any]) -> str:
    return f"pin={_pinning_fs(config)} ctx={_context_mode(config)}"


def _model_slug(config: dict[str, Any]) -> str:
    m = config.get("api_model") or "unknown_model"
    return str(m).replace("/", "_").replace(":", "_")


def _pin_run_suffix(config: dict[str, Any]) -> str:
    """Match evaluate.d2.pipeline_d2 default run_name suffix."""
    pm = config.get("pinning_mode") or "inline"
    return "requirements_txt" if pm == "requirements.txt" else str(pm)


def _py_tag(config: dict[str, Any]) -> str:
    """Same as pipeline ``py_tag`` (e.g. 3.12 -> py312)."""
    ver = config.get("python_version") or "3.12"
    return f"py{str(ver).replace('.', '')}"


def _experiment_stem_for_job(job: Job) -> str:
    rn = job.config.get("run_name")
    if rn is not None and str(rn).strip() and not _RUN_NAME_PLACEHOLDER.match(str(rn).strip()):
        return _sanitize_name(str(rn).strip())
    slug = _model_slug(job.config)
    ctx = _context_mode(job.config)
    pin = _pin_run_suffix(job.config)
    return _sanitize_name(f"d2_{slug}_{ctx}_{pin}")


def _job_log_stem_with_py(job: Job) -> str:
    return f"{_experiment_stem_for_job(job)}_{_py_tag(job.config)}"


def assign_unique_log_stems(jobs: list[Job]) -> None:
    groups: dict[str, list[Job]] = defaultdict(list)
    for j in jobs:
        groups[_job_log_stem_with_py(j)].append(j)
    for stem, group in groups.items():
        if len(group) == 1:
            group[0].log_stem = stem
        else:
            for j in group:
                j.log_stem = f"{stem}__L{j.idx:04d}"


def job_subprocess_log_path(log_dir: Path, job: Job) -> Path:
    """``log_dir / <experiment_stem>_<py_tag>.log`` (stem set by :func:`assign_unique_log_stems`)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = job.log_stem or _job_log_stem_with_py(job)
    return log_dir / f"{stem}.log"


def load_jobs(path: Path) -> list[Job]:
    if not path.is_file():
        raise FileNotFoundError(f"jobs jsonl not found: {path}")
    jobs: list[Job] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {i}: job must be a JSON object")
            run_name = str(obj.get("run_name") or f"job_{i}")
            name = _sanitize_name(run_name)
            jobs.append(Job(idx=i, config=obj, name=name))
    return jobs


def _append_flag(cmd: list[str], key: str, value: Any) -> None:
    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    if key == "stages":
        if not isinstance(value, list) or not value:
            raise ValueError("stages must be a non-empty list")
        cmd.append(flag)
        cmd.extend(str(v) for v in value)
        return
    cmd.extend([flag, str(value)])


def build_cmd(job: Job, python_bin: str) -> list[str]:
    unknown = sorted(set(job.config.keys()) - SUPPORTED_KEYS)
    if unknown:
        raise ValueError(f"job line {job.idx} has unsupported keys: {unknown}")

    cmd = [python_bin, "-m", "evaluate.d2.pipeline_d2"]
    # Keep a stable order so logs are easy to compare.
    ordered_keys = [
        "run_name",
        "stages",
        "dataset",
        "llm_output_path",
        "api_mode",
        "api_model",
        "api_url",
        "api_key",
        "provider_order",
        "provider_allow_fallbacks",
        "http_referer",
        "site_title",
        "prompt_mode",
        "pinning_mode",
        "timeout",
        "workers",
        "max_examples",
        "max_records",
        "outputs_d2",
        "python_version",
        "reference_date",
        "xdg_cache_path",
        "max_hint_chars",
        "skip_vuln",
        "skip_compat",
    ]
    for key in ordered_keys:
        _append_flag(cmd, key, job.config.get(key))
    return cmd


def run_one_job(
    job: Job,
    *,
    python_bin: str,
    project_root: Path,
    log_dir: Path,
    dry_run: bool,
) -> tuple[Job, int]:
    cmd = build_cmd(job, python_bin=python_bin)
    cmd_str = " ".join(cmd)
    tag = _job_mode_tag(job.config)
    logger.info(f"[{tag}] [RUN] line={job.idx} run_name={job.config.get('run_name', '')}")
    logger.info(f"[{tag}]       {cmd_str}")

    if dry_run:
        return job, 0

    log_path = job_subprocess_log_path(log_dir, job)
    env = os.environ.copy()
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"### pinning_mode={job.config.get('pinning_mode', 'inline')} ")
        log_f.write(f"prompt_mode={job.config.get('prompt_mode', 'blind')}\n")
        log_f.write(f"### CMD: {cmd_str}\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=env,
        )
        ret = proc.wait()

    if ret == 0:
        logger.info(f"[{tag}] [DONE] line={job.idx} run_name={job.config.get('run_name', '')}")
    else:
        logger.error(f"[{tag}] [FAIL] line={job.idx} exit={ret} log={log_path}")
    return job, ret


def notify_batch_result(
    *,
    args: argparse.Namespace,
    total_jobs: int,
    completed: int,
    failures: int,
    started_at: datetime,
    status: str,
) -> None:
    duration_sec = round((datetime.now(timezone.utc) - started_at).total_seconds(), 2)
    summary = {
        "status": status,
        "pipeline": "d2",
        "jobs_jsonl": str(args.jobs_jsonl),
        "total_jobs": total_jobs,
        "completed_jobs": completed,
        "failures": failures,
        "parallel": args.parallel,
        "dry_run": args.dry_run,
        "fail_fast": args.fail_fast,
        "python_bin": args.python_bin,
        "project_root": str(args.project_root),
        "log_dir": str(args.log_dir),
        "duration_seconds": duration_sec,
    }
    sent, msg = send_experiment_notification(
        experiment_name="batch_pipeline_d2",
        status=status,
        summary=summary,
    )
    if sent:
        logger.success(f"[notify] {msg}")
    else:
        logger.info(f"[notify] {msg}")


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = Path(args.jobs_jsonl)
    file_name = jobs_path.stem or "jobs"

    logger.add(args.log_dir / f"batch_{file_name}.log")

    jobs = load_jobs(args.jobs_jsonl)
    if not jobs:
        logger.info("No jobs found in JSONL, nothing to do.")
        notify_batch_result(
            args=args,
            total_jobs=0,
            completed=0,
            failures=0,
            started_at=started_at,
            status="NO_JOBS",
        )
        return

    assign_unique_log_stems(jobs)

    logger.info(f"jobs_jsonl={args.jobs_jsonl}")
    logger.info(f"jobs={len(jobs)}, parallel={args.parallel}, dry_run={args.dry_run}")

    failures = 0
    completed = 0

    if args.parallel <= 1:
        for job in jobs:
            _, code = run_one_job(
                job,
                python_bin=args.python_bin,
                project_root=args.project_root,
                log_dir=args.log_dir,
                dry_run=args.dry_run,
            )
            completed += 1
            if code != 0:
                failures += 1
                if args.fail_fast:
                    logger.error("fail_fast enabled, stop after first failure")
                    break
            logger.info(f"progress={completed}/{len(jobs)} failures={failures}")
        notify_batch_result(
            args=args,
            total_jobs=len(jobs),
            completed=completed,
            failures=failures,
            started_at=started_at,
            status="SUCCESS" if failures == 0 else "FAILED",
        )
        sys.exit(1 if failures else 0)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(
                run_one_job,
                job,
                python_bin=args.python_bin,
                project_root=args.project_root,
                log_dir=args.log_dir,
                dry_run=args.dry_run,
            ): job
            for job in jobs
        }
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                _, code = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                tag = _job_mode_tag(job.config)
                logger.error(f"[{tag}] [FAIL] line={job.idx} raised exception: {exc}")
                if args.fail_fast:
                    logger.error("fail_fast enabled, cancelling pending jobs")
                    for pending in futures:
                        pending.cancel()
                    break
            else:
                if code != 0:
                    failures += 1
                    if args.fail_fast:
                        logger.error("fail_fast enabled, cancelling pending jobs")
                        for pending in futures:
                            pending.cancel()
                        break
            completed += 1
            logger.info(f"progress={completed}/{len(jobs)} failures={failures}")

    logger.success(f"batch finished: total={len(jobs)} failures={failures}")
    notify_batch_result(
        args=args,
        total_jobs=len(jobs),
        completed=completed,
        failures=failures,
        started_at=started_at,
        status="SUCCESS" if failures == 0 else "FAILED",
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
