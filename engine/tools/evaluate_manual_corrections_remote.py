"""Evaluate corrected saved artifacts under the primary frozen configuration."""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path


def measurement_state(row, result):
    eligible = bool(row["code"].strip() and row["requirements"])
    static_status = "not_run"
    if "static" in result:
        command = result["static"]
        if command.get("timeout") or command.get("returncode") == 124:
            static_status = "timeout"
        elif result["static_pass"]:
            static_status = "pass"
        else:
            try:
                diagnostics = json.loads(command.get("stdout") or "[]")
            except ValueError:
                diagnostics = None
            static_status = "diagnostic" if command.get("returncode") in {0, 1} and isinstance(diagnostics, list) and diagnostics else "tool_error"
    dynamic_status = "not_run"
    if "dynamic" in result:
        command = result["dynamic"]
        oracle_status = result.get("bcb_result", {}).get("status")
        if command.get("timeout") or command.get("returncode") == 124 or oracle_status == "timeout":
            dynamic_status = "timeout"
        elif result["dynamic_pass"]:
            dynamic_status = "pass"
        elif command.get("returncode") == 0 and oracle_status == "fail":
            dynamic_status = "test_failure"
        elif command.get("returncode") == 0 and oracle_status == "error":
            dynamic_status = "execution_error"
        else:
            dynamic_status = "runner_error"
    if not row["code"].strip():
        stage = "missing_code"
    elif not row["requirements"]:
        stage = "empty_dependency_input"
    elif not result["resolved"]:
        stage = "resolution_failed"
    elif result.get("venv", {}).get("returncode") != 0:
        stage = "venv_failed"
    elif not result["installed"]:
        stage = "installation_failed"
    elif result.get("tool_installation", {}).get("returncode") != 0:
        stage = "tool_setup_failed"
    elif not result["static_pass"]:
        stage = "static_" + static_status
    elif dynamic_status not in {"pass", "not_run"}:
        stage = "dynamic_" + dynamic_status
    else:
        stage = "passed_available_checks"
    return {
        "environment_eligible": eligible,
        "resolver_status": "empty" if not eligible else "resolved" if result["resolved"] else "failed",
        "complete_lock": eligible and result["resolved"],
        "installation_pass": eligible and result["installed"],
        "static_pass": eligible and result["static_pass"],
        "static_executed": "static" in result,
        "static_status": static_status,
        "dynamic_executed": "dynamic" in result,
        "dynamic_status": dynamic_status,
        "dynamic_pass": result["dynamic_pass"],
        "failure_stage": stage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PINTRACE_ENGINE_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root))
    os.environ["EXPERIMENT_NOTIFY_ENABLED"] = "0"
    from evaluation_support_remote import evaluate
    problems = {
        str(record["task_id"]): record
        for line in (args.root / "resources/BigCodeBench-v0.1.4-local.jsonl").open()
        if line.strip() for record in [json.loads(line)]
    }
    jobs = []
    for line in args.prepared.open():
        row = json.loads(line)
        if not row["evaluation_changed"] or row["review_required"] or not row.get("evaluation_eligible", True):
            continue
        if args.manual_only and not row["manual_case_id"]:
            continue
        if args.cases and row["manual_case_id"] not in args.cases and row["request_id"] not in args.cases:
            continue
        jobs.append(row)

    def process(row):
        path = args.out / "cases" / row["request_id"]
        problem = problems.get(row["task_id"]) if row["track"] == "d1" else None
        result = evaluate(
            row["code"], row["requirements"], path, root=args.root,
            cutoff="2026-02-01T00:00:00Z", bcb_problem=problem,
            files=row["files"], primary_filename=row["primary_filename"],
        )
        record = {
            **{name: row[name] for name in ["request_id", "track", "mode", "model", "task_id", "manual_case_id", "response_sha256", "code_changed", "requirements_changed"]},
            "old_outcome": row["old_outcome"], "evaluation": result,
            "measurement_state": measurement_state(row, result),
        }
        temporary = path / "record.tmp"
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(path / "record.json")
        return record

    print(f"Evaluating {len(jobs)} saved responses", flush=True)
    completed = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, row): row for row in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                record = future.result()
            except Exception as error:
                row = futures[future]
                errors.append({"request_id": row["request_id"], "error": repr(error)})
                print(json.dumps(errors[-1]), flush=True)
                continue
            completed.append(record)
            result = record["evaluation"]
            print(json.dumps({
                "completed": len(completed), "planned": len(jobs),
                "case": record["manual_case_id"] or record["request_id"],
                "resolved": result["resolved"], "installed": result["installed"],
                "static": result["static_pass"], "dynamic": result["dynamic_pass"],
            }), flush=True)
    if errors:
        (args.out / "evaluation_errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(f"{len(errors)} evaluations need a retry; completed records are preserved")
    completed.sort(key=lambda row: row["request_id"])
    temporary = args.out / "evaluations.tmp"
    with temporary.open("w") as output:
        for row in completed:
            output.write(json.dumps(row) + "\n")
    temporary.replace(args.out / "evaluations.jsonl")
    print(f"Evaluation complete {len(completed)}", flush=True)


if __name__ == "__main__":
    main()
