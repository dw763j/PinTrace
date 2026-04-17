#!/usr/bin/env python
"""Unified batch runner for evaluate.pipeline.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m scripts.batch_pipeline --help

This script merges the capabilities of:
- scripts/batch_pipeline_d1.py (auto-discover D1 runs from outputs tree)
- scripts/batch_pipeline_d2.py (JSONL-configured batch jobs)

It also supports the new ablation mode exposed by evaluate.pipeline:
- --inference-mode ablation
- --ablation-prompts-jsonl <path>

Run modes:
1) D1 auto-discovery mode (legacy D1 batch style)
2) JSONL jobs mode (legacy D2 batch style, now supports both d1/d2)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import LOGS, OUTPUTS
from scripts.notifier_utils import send_experiment_notification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "batch_pipeline"
D1_MODES_DEFAULT = ("inline", "inline_no_vuln", "requirements.txt")

SUPPORTED_KEYS = {
    "track",
    "inference_mode",
    "ablation_prompts_jsonl",
    "dataset",
    "split",
    "filter_third_party",
    "python_version",
    "run_name",
    "prompt_mode",
    "pinning_mode",
    "api_url",
    "api_key",
    "api_model",
    "api_mode",
    "provider_order",
    "provider_allow_fallbacks",
    "http_referer",
    "site_title",
    "timeout",
    "max_hint_chars",
    "max_examples",
    "workers",
    "max_records",
    "reference_date",
    "llm_output_path",
    "outputs_d1",
    "outputs_d2",
    "stages",
    "osv_index_path",
    "xdg_cache_path",
    "bigcodebench_path",
    "bcb_timeout",
    "compat_checkpoint_every",
    "skip_vuln",
    "skip_compat",
    "email",
}


@dataclass
class Job:
    idx: int
    config: dict[str, Any]
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified batch runner for evaluate.pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--jobs-jsonl",
        type=Path,
        default=None,
        help="JSONL file; each line is one job config for evaluate.pipeline.",
    )
    mode_group.add_argument(
        "--d1-auto",
        action="store_true",
        help="Auto-discover D1 jobs from outputs/d1/<mode>/<run_name>/*.jsonl.",
    )

    parser.add_argument("--parallel", type=int, default=1, help="Concurrent jobs.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for subprocesses.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Subprocess cwd.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Per-job logs directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first failure.")
    parser.add_argument("--email", action="store_true", help="Send batch completion email notification.")

    # D1 auto-discovery options
    parser.add_argument("--outputs-d1", type=Path, default=OUTPUTS / "d1", help="D1 outputs root.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(D1_MODES_DEFAULT),
        choices=list(D1_MODES_DEFAULT),
        help="Prompt modes used in D1 auto-discovery.",
    )
    parser.add_argument("--skip-done", action="store_true", help="Skip run if metrics_summary.json exists.")
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        metavar="N",
        help="Stages for D1 auto jobs.",
    )
    parser.add_argument("--python-version", default="3.12", help="python_version for D1 auto jobs.")
    parser.add_argument("--pipeline-run-name", default=None, help="Optional logger suffix for this batch run.")
    parser.add_argument("--api-model", default=None, help="Optional api_model override for D1 auto jobs.")
    return parser.parse_args()


def _sanitize_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in text)[:140]


def _infer_api_model_from_run_name(run_name: str) -> str:
    api_model = run_name
    for suffix in (
        "_inline",
        "_inline_no_vuln",
        "_requirements.txt",
        "_req",
        "-req",
    ):
        if api_model.endswith(suffix):
            api_model = api_model[: -len(suffix)]
    return api_model


def _load_jsonl_jobs(path: Path) -> list[Job]:
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
            if obj.get("track") not in {"d1", "d2"}:
                raise ValueError(f"line {i}: 'track' must be 'd1' or 'd2'")
            name = _sanitize_name(str(obj.get("run_name") or f"job_{i}"))
            jobs.append(Job(idx=i, config=obj, name=name))
    return jobs


def _discover_d1_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    idx = 0
    for mode in args.modes:
        mode_dir = args.outputs_d1 / mode
        if not mode_dir.is_dir():
            continue
        for sub in sorted(mode_dir.iterdir()):
            if not sub.is_dir():
                continue
            run_name = sub.name
            jsonls = sorted(sub.glob("*.jsonl"))
            if not jsonls:
                continue
            llm_output = jsonls[0]
            config: dict[str, Any] = {
                "track": "d1",
                "stages": args.stages,
                "outputs_d1": str(args.outputs_d1),
                "llm_output_path": str(llm_output),
                "prompt_mode": mode,
                "run_name": run_name,
                "python_version": args.python_version,
                "api_model": args.api_model or _infer_api_model_from_run_name(run_name),
                "workers": max(2, 16 // max(1, args.parallel)),
            }
            idx += 1
            jobs.append(Job(idx=idx, config=config, name=_sanitize_name(run_name)))
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


def _build_cmd(job: Job, python_bin: str) -> list[str]:
    unknown = sorted(set(job.config.keys()) - SUPPORTED_KEYS)
    if unknown:
        raise ValueError(f"job line {job.idx} has unsupported keys: {unknown}")
    cmd = [python_bin, "-m", "evaluate.pipeline"]
    ordered_keys = [
        "track",
        "inference_mode",
        "ablation_prompts_jsonl",
        "run_name",
        "stages",
        "dataset",
        "llm_output_path",
        "split",
        "filter_third_party",
        "prompt_mode",
        "pinning_mode",
        "api_mode",
        "api_model",
        "api_url",
        "api_key",
        "provider_order",
        "provider_allow_fallbacks",
        "http_referer",
        "site_title",
        "timeout",
        "workers",
        "max_examples",
        "max_records",
        "outputs_d1",
        "outputs_d2",
        "python_version",
        "reference_date",
        "xdg_cache_path",
        "max_hint_chars",
        "osv_index_path",
        "bigcodebench_path",
        "bcb_timeout",
        "compat_checkpoint_every",
        "skip_vuln",
        "skip_compat",
        "email",
    ]
    for key in ordered_keys:
        _append_flag(cmd, key, job.config.get(key))
    return cmd


def _metrics_path_for_job(job: Job) -> Path | None:
    track = job.config.get("track")
    run_name = job.config.get("run_name")
    py_ver = str(job.config.get("python_version") or "3.12")
    py_tag = f"py{py_ver.replace('.', '')}"
    if not isinstance(run_name, str) or not run_name:
        return None
    if track == "d1":
        prompt_mode = str(job.config.get("prompt_mode") or "inline")
        root = Path(str(job.config.get("outputs_d1") or (OUTPUTS / "d1")))
        return root / prompt_mode / run_name / py_tag / "metrics_summary.json"
    if track == "d2":
        pinning_mode = str(job.config.get("pinning_mode") or "inline")
        root = Path(str(job.config.get("outputs_d2") or (OUTPUTS / "d2")))
        return root / pinning_mode / run_name / py_tag / "metrics_summary.json"
    return None


def _job_tag(job: Job) -> str:
    track = job.config.get("track", "?")
    prompt_mode = job.config.get("prompt_mode")
    pinning_mode = job.config.get("pinning_mode")
    inference_mode = job.config.get("inference_mode", "standard")
    parts = [f"track={track}", f"infer={inference_mode}"]
    if prompt_mode:
        parts.append(f"ctx={prompt_mode}")
    if pinning_mode:
        parts.append(f"pin={pinning_mode}")
    return " ".join(parts)


def _run_one_job(
    job: Job,
    *,
    python_bin: str,
    project_root: Path,
    log_dir: Path,
    dry_run: bool,
    skip_done: bool,
) -> tuple[Job, int]:
    if skip_done:
        mpath = _metrics_path_for_job(job)
        if mpath is not None and mpath.is_file():
            logger.info(f"[{_job_tag(job)}] [SKIP] line={job.idx} run_name={job.config.get('run_name')} (metrics exists)")
            return job, 0

    cmd = _build_cmd(job, python_bin=python_bin)
    cmd_str = " ".join(cmd)
    logger.info(f"[{_job_tag(job)}] [RUN]  line={job.idx} run_name={job.config.get('run_name', '')}")
    logger.info(f"[{_job_tag(job)}]       {cmd_str}")

    if dry_run:
        return job, 0

    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"{job.config.get('track', 'x')}_{job.name}__L{job.idx:04d}.log"
    log_path = log_dir / log_name
    env = os.environ.copy()

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"### TAG: {_job_tag(job)}\n")
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
        logger.info(f"[{_job_tag(job)}] [DONE] line={job.idx} run_name={job.config.get('run_name', '')}")
    else:
        logger.error(f"[{_job_tag(job)}] [FAIL] line={job.idx} exit={ret} log={log_path}")
    return job, ret


def _notify_batch_result(
    *,
    args: argparse.Namespace,
    total_jobs: int,
    completed: int,
    failures: int,
    started_at: datetime,
    status: str,
) -> None:
    if not args.email:
        logger.info("[notify] skipped (pass --email to enable)")
        return

    duration_sec = round((datetime.now(timezone.utc) - started_at).total_seconds(), 2)
    mode = "d1_auto" if args.d1_auto else "jobs_jsonl"
    summary = {
        "status": status,
        "pipeline": "evaluate.pipeline",
        "mode": mode,
        "jobs_jsonl": str(args.jobs_jsonl) if args.jobs_jsonl else None,
        "total_jobs": total_jobs,
        "completed_jobs": completed,
        "failures": failures,
        "parallel": args.parallel,
        "dry_run": args.dry_run,
        "fail_fast": args.fail_fast,
        "skip_done": args.skip_done,
        "python_bin": args.python_bin,
        "project_root": str(args.project_root),
        "log_dir": str(args.log_dir),
        "duration_seconds": duration_sec,
    }
    sent, msg = send_experiment_notification(
        experiment_name="batch_pipeline",
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

    if args.pipeline_run_name:
        logger.add(LOGS / f"batch_pipeline_{args.pipeline_run_name}.log")
    else:
        logger.add(LOGS / "batch_pipeline.log")

    if args.d1_auto:
        jobs = _discover_d1_jobs(args)
        logger.info(f"mode=d1_auto outputs_d1={args.outputs_d1} modes={args.modes}")
    else:
        jobs = _load_jsonl_jobs(args.jobs_jsonl)
        logger.info(f"mode=jobs_jsonl jobs_jsonl={args.jobs_jsonl}")

    if not jobs:
        logger.info("No jobs found. Nothing to do.")
        _notify_batch_result(
            args=args,
            total_jobs=0,
            completed=0,
            failures=0,
            started_at=started_at,
            status="NO_JOBS",
        )
        return

    logger.info(f"jobs={len(jobs)} parallel={args.parallel} dry_run={args.dry_run} fail_fast={args.fail_fast}")

    failures = 0
    completed = 0

    if args.parallel <= 1:
        for job in jobs:
            _, code = _run_one_job(
                job,
                python_bin=args.python_bin,
                project_root=args.project_root,
                log_dir=args.log_dir,
                dry_run=args.dry_run,
                skip_done=args.skip_done,
            )
            completed += 1
            if code != 0:
                failures += 1
                if args.fail_fast:
                    logger.error("fail_fast enabled, stop after first failure")
                    break
            logger.info(f"progress={completed}/{len(jobs)} failures={failures}")
        _notify_batch_result(
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
                _run_one_job,
                job,
                python_bin=args.python_bin,
                project_root=args.project_root,
                log_dir=args.log_dir,
                dry_run=args.dry_run,
                skip_done=args.skip_done,
            ): job
            for job in jobs
        }
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                _, code = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.error(f"[{_job_tag(job)}] [FAIL] line={job.idx} raised exception: {exc}")
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
    _notify_batch_result(
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
