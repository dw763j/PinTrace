#!/usr/bin/env python
"""
Batch runner for D1 pipeline stages 2–6.

Goals:
- Run ``pipeline_d1`` stages M2–M6 for every LLM output (``*.jsonl``) already stored under ``outputs/d1``.
- Support parallelism, resume-friendly checkpoints, and long-running ``nohup``/tmux sessions.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    # Sequential pass across inline / inline_no_vuln / requirements.txt
    python -m scripts.batch_pipeline_d1

    # Only selected modes with 8-way parallelism
    python -m scripts.batch_pipeline_d1 --modes inline inline_no_vuln --parallel 8

    # Skip runs that already produced metrics_summary.json
    python -m scripts.batch_pipeline_d1 --skip-done

    # Background-friendly invocation
    nohup python -m scripts.batch_pipeline_d1 --parallel 8 \\
        > logs/batch_pipeline/python_batch_nohup.out 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from loguru import logger
from paths import OUTPUTS, LOGS

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.notifier_utils import send_experiment_notification


MODES_DEFAULT = ("inline", "inline_no_vuln", "requirements.txt")


@dataclass
class Job:
    mode: str
    run_name: str
    llm_output_path: Path
    run_dir: Path
    api_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run D1 pipeline (stages 2–6) for existing LLM outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outputs-d1",
        type=Path,
        default=OUTPUTS / "d1",
        help="Root directory containing D1 LLM outputs (inline / inline_no_vuln / requirements.txt).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(MODES_DEFAULT),
        choices=list(MODES_DEFAULT),
        help="Prompt modes to include.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of jobs to run in parallel. With 16 cores, 4–8 is reasonable.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip runs where metrics_summary.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands instead of executing.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        metavar="N",
        help="Pipeline stages to run (e.g. --stages 5 6 to only redo compat + metrics).",
    )
    parser.add_argument(
        "--uv-cmd",
        default="uv",
        help="uv executable to use (if not on PATH, set full path here).",
    )
    parser.add_argument(
        "--python-version",
        default="3.12",
        help="Python version to use inside D1 pipeline (ty + BCB environments).",
    )
    parser.add_argument(
        "--pipeline-run-name",
        default=None,
        help="Run name to use for the D1 pipeline.",
    )
    return parser.parse_args()


def infer_api_model(run_name: str) -> str:
    """Heuristic to recover api_model from run_name."""
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


def discover_jobs(outputs_d1: Path, modes: Iterable[str]) -> List[Job]:
    jobs: List[Job] = []
    for mode in modes:
        mode_dir = outputs_d1 / mode
        if not mode_dir.is_dir():
            continue
        for sub in sorted(mode_dir.iterdir()):
            if not sub.is_dir():
                continue
            run_name = sub.name
            jsonls = sorted(sub.glob("*.jsonl"))
            if not jsonls:
                continue
            # One jsonl per run_dir in the current outputs layout
            llm_output = jsonls[0]
            api_model = infer_api_model(run_name)
            jobs.append(
                Job(
                    mode=mode,
                    run_name=run_name,
                    llm_output_path=llm_output,
                    run_dir=sub,
                    api_model=api_model,
                )
            )
    return jobs


def build_cmd(
    job: Job,
    uv_cmd: str,
    workers: int,
    stages: list[int],
    python_version: str,
    outputs_d1: Path,
) -> list[str]:
    """Construct the uv run command for one job."""
    cmd = [
        uv_cmd,
        "run",
        "python",
        "-m",
        "evaluate.d1.pipeline_d1",
        "--outputs-d1",
        str(outputs_d1),
        "--llm-output-path",
        str(job.llm_output_path),
        "--prompt-mode",
        job.mode,
        "--run-name",
        job.run_name,
        "--api-model",
        job.api_model,
        "--stages",
        *[str(s) for s in stages],
        "--workers",
        str(workers),
        "--python-version",
        python_version,
    ]
    return cmd


def run_job(
    job: Job,
    uv_cmd: str,
    skip_done: bool,
    dry_run: bool,
    parallel: int,
    stages: list[int],
    python_version: str,
    outputs_d1: Path,
) -> tuple[Job, int]:
    """Run a single pipeline job. Returns (job, exit_code).

    Results for different python_version values are separated into subdirectories
    under the same run_name, to avoid overwriting prior experiments:
      outputs/d1/<mode>/<run_name>/py<ver>/metrics_summary.json
    """
    py_tag = f"py{str(python_version).replace('.', '')}"
    metrics_path_new = outputs_d1 / job.mode / job.run_name / py_tag / "metrics_summary.json"
    metrics_path_legacy = outputs_d1 / job.mode / job.run_name / "metrics_summary.json"
    if skip_done and (metrics_path_new.is_file() or metrics_path_legacy.is_file()):
        logger.info(
            f"[SKIP] {job.mode}/{job.run_name} "
            f"(metrics_summary.json exists for {py_tag if metrics_path_new.is_file() else 'legacy'})"
        )
        return job, 0

    # Roughly allocate workers per job based on 16 logical cores.
    workers = max(2, 16 // max(1, parallel))

    cmd = build_cmd(
        job,
        uv_cmd=uv_cmd,
        workers=workers,
        stages=stages,
        python_version=python_version,
        outputs_d1=outputs_d1,
    )
    cmd_str = " ".join(cmd)
    logger.info(f"[RUN]  {job.mode}/{job.run_name}")
    logger.info(f"       {cmd_str}")

    if dry_run:
        return job, 0

    log_dir = OUTPUTS.parent / "logs" / "batch_pipeline_python"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.mode}_{job.run_name}.log"

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"### CMD: {cmd_str}\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(OUTPUTS.parent),  # project root
            env=os.environ.copy(),
        )
        ret = proc.wait()

    if ret == 0:
        logger.info(f"[DONE] {job.mode}/{job.run_name}")
    else:
        logger.error(f"[FAIL] {job.mode}/{job.run_name} (exit={ret}) log={log_path}")
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
        "pipeline": "d1",
        "total_jobs": total_jobs,
        "completed_jobs": completed,
        "failures": failures,
        "parallel": args.parallel,
        "dry_run": args.dry_run,
        "skip_done": args.skip_done,
        "stages": args.stages,
        "python_version": args.python_version,
        "modes": args.modes,
        "outputs_d1": str(args.outputs_d1),
        "pipeline_run_name": args.pipeline_run_name or "default",
        "duration_seconds": duration_sec,
    }
    sent, msg = send_experiment_notification(
        experiment_name="batch_pipeline_d1",
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
    outputs_d1: Path = args.outputs_d1
    modes: list[str] = args.modes
    
    if args.pipeline_run_name:
        logger.add(LOGS / f"batch_pipeline_d1_{args.pipeline_run_name}.log")
    else:
        logger.add(LOGS / "batch_pipeline_d1.log")

    logger.info(f"[batch] pipeline_run_name     = {args.pipeline_run_name if args.pipeline_run_name else 'default'}")
    logger.info(f"[batch] outputs_d1   = {outputs_d1}")
    logger.info(f"[batch] modes        = {modes}")
    logger.info(f"[batch] stages       = {args.stages}")
    logger.info(f"[batch] parallel     = {args.parallel}")
    logger.info(f"[batch] skip_done    = {args.skip_done}")
    logger.info(f"[batch] dry_run      = {args.dry_run}")

    jobs = discover_jobs(outputs_d1, modes)
    if not jobs:
        logger.info("[batch] No jobs discovered. Nothing to do.")
        notify_batch_result(
            args=args,
            total_jobs=0,
            completed=0,
            failures=0,
            started_at=started_at,
            status="NO_JOBS",
        )
        return

    total_jobs = len(jobs)
    logger.info(f"[batch] Discovered {total_jobs} jobs.")

    # Sequential
    if args.parallel <= 1:
        failures = 0
        completed = 0
        for job in jobs:
            _, code = run_job(
                job,
                uv_cmd=args.uv_cmd,
                skip_done=args.skip_done,
                dry_run=args.dry_run,
                parallel=1,
                stages=args.stages,
                python_version=args.python_version,
                outputs_d1=outputs_d1,
            )
            completed += 1
            if code != 0:
                failures += 1
            logger.info(
                f"[batch] progress: {completed}/{total_jobs} "
                f"(failures={failures})",
            )
        notify_batch_result(
            args=args,
            total_jobs=total_jobs,
            completed=completed,
            failures=failures,
            started_at=started_at,
            status="SUCCESS" if failures == 0 else "FAILED",
        )
        logger.info(f"[batch] Done. Failures: {failures}")
        sys.exit(1 if failures else 0)

    # Parallel using ThreadPoolExecutor (each task is a subprocess)
    failures = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        fut_to_job = {
            ex.submit(
                run_job,
                job,
                args.uv_cmd,
                args.skip_done,
                args.dry_run,
                args.parallel,
                args.stages,
                args.python_version,
                outputs_d1,
            ): job
            for job in jobs
        }
        for fut in as_completed(fut_to_job):
            job = fut_to_job[fut]
            try:
                _, code = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[FAIL] {job.mode}/{job.run_name} raised exception: {exc}")
                failures += 1
            else:
                if code != 0:
                    failures += 1
            completed += 1
            logger.info(
                f"[batch] progress: {completed}/{total_jobs} "
                f"(failures={failures}, parallel={args.parallel})",
            )

    logger.success(f"[batch] Done. Failures: {failures}")
    notify_batch_result(
        args=args,
        total_jobs=total_jobs,
        completed=completed,
        failures=failures,
        started_at=started_at,
        status="SUCCESS" if failures == 0 else "FAILED",
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

