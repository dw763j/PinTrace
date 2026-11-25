import os
"""Reproduce response-level construction and execution evidence for RQ3."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1]))
MODEL_ORDER = [
    "gpt-5.4", "claude-sonnet-4-6", "gemini-3.1-pro-preview", "deepseek-v3.2",
    "kimi-k2.5", "qwen3.5-397b-a17b", "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-instruct-2507", "minimax-m2.5", "meta-llama/llama-4-scout",
]
CONDITIONS = {"requirements.txt": "MR", "inline": "VA"}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_evidence(root=ROOT):
    source = root / "inputs/main/task_outcomes.jsonl"
    paired = root / "analysis/paired_cluster_bootstrap.csv"
    outcomes = [json.loads(line) for line in source.open(encoding="utf-8")]
    with paired.open(encoding="utf-8-sig") as stream:
        intervals = {(r["model"], r["population"], r["metric"]): r for r in csv.DictReader(stream)}
    stage_rows, execution_rows = [], []
    stopping_details, test_statuses = {}, {}
    for model in MODEL_ORDER + ["All observations"]:
        for mode, condition in CONDITIONS.items():
            selected = [r for r in outcomes if r["mode"] == mode
                        and (model == "All observations" or r["model"] == model)]
            pin = [r for r in selected if r["track"] == "d2"]
            expected = 10000 if model == "All observations" else 1000
            assert len(pin) == expected
            n = len(pin)
            entered = sum(r["environment_eligible"] is True for r in pin)
            locked = sum(r["complete_lock"] is True for r in pin)
            installed = sum(r["installation_pass"] is True for r in pin)
            static = sum(r["static_pass"] is True for r in pin)
            assert all(r["static_executed"] is True for r in pin if r["installation_pass"] is True)
            counts = [n - entered, entered - locked, locked - installed, installed - static, static]
            denominators = [n, entered, locked, installed, installed]
            assert sum(counts) == n and all(c >= 0 for c in counts)
            keys = ["not_entered", "resolution_failure", "installation_failure", "static_diagnostics", "static_pass"]
            for key, count, denominator in zip(keys, counts, denominators):
                stage_rows.append({
                    "model": model, "condition": condition, "stage": key, "N": n,
                    "count": count, "reached_stage_N": denominator,
                    "population_pct": 100 * count / n,
                    "conditional_pct": 100 * count / denominator if denominator else "",
                })
            if model == "All observations":
                stopped = [r for r in pin if r["environment_eligible"] is not True]
                stopping_details[condition] = dict(Counter(r["failure_stage"] for r in stopped))

            bcb = [r for r in selected if r["track"] == "d1"]
            n = len(bcb)
            assert n == (8130 if model == "All observations" else 813)
            accepted = [r for r in bcb if r["static_pass"] is True]
            assert all(r["dynamic_executed"] is True for r in accepted)
            static_n = len(accepted)
            joint = sum(r["dynamic_pass"] is True for r in accepted)
            nonpass = static_n - joint
            tests = sum(r["dynamic_pass"] is True for r in bcb)
            installed_n = sum(r["installation_pass"] is True for r in bcb)
            assert all(r["installation_pass"] is True and r["dynamic_executed"] is True
                       for r in bcb if r["dynamic_pass"] is True)
            assert all(r["dynamic_executed"] is not True
                       for r in bcb if r["installation_pass"] is not True)
            all_nonpass = n - tests
            no_environment = n - installed_n
            assert no_environment + (installed_n - tests) + tests == n
            interval = intervals[model, "BigCodeBench", "dynamic"]
            assert abs(100 * tests / n - float(interval[condition + "_pct"])) < 1e-8
            execution_rows.append({
                "model": model, "condition": condition, "N": n,
                "installed": installed_n,
                "static_pass": static_n, "static_test_pass": joint,
                "static_test_nonpass": nonpass, "static_test_nonpass_pct": 100 * nonpass / static_n,
                "test_pass": tests, "test_pass_pct": 100 * tests / n,
                "test_only_pass": tests - joint,
                "test_pass_ci_low": float(interval[condition + "_ci_low"]),
                "test_pass_ci_high": float(interval[condition + "_ci_high"]),
                "all_test_nonpass": all_nonpass,
                "no_installed_environment": no_environment,
                "installed_test_nonpass": installed_n - tests,
                "no_environment_among_nonpasses_pct": 100 * no_environment / all_nonpass if all_nonpass else "",
                "installed_test_pass_pct": 100 * tests / installed_n if installed_n else "",
            })
            if model == "All observations":
                test_statuses[condition] = dict(Counter(r["dynamic_status"] for r in accepted if r["dynamic_pass"] is not True))

    analysis = root / "analysis"
    write_csv(analysis / "rq3_stage_outcomes.csv", stage_rows)
    write_csv(analysis / "rq3_execution_outcomes.csv", execution_rows)
    groups = [r for r in execution_rows if r["model"] != "All observations"]
    summary = {
        "stage_population": "Complete PinTrace, response-level recorded stopping stage or static pass",
        "execution_population": "813-task third-party-library BigCodeBench subset",
        "unit": "Model response; counts summed before aggregation percentages",
        "not_entered_details": stopping_details,
        "static_passing_test_nonpass_statuses": test_statuses,
        "minimum_observed_static_passing_nonpass_group": min(groups, key=lambda r: r["static_test_nonpass_pct"]),
        "models_above_half_among_static_passes": {
            condition: [r["model"] for r in groups if r["condition"] == condition and r["static_test_nonpass_pct"] > 50]
            for condition in CONDITIONS.values()
        },
        "models_with_majority_preinstallation_nonpasses": {
            condition: [r["model"] for r in groups if r["condition"] == condition
                        and r["no_environment_among_nonpasses_pct"] > 50]
            for condition in CONDITIONS.values()
        },
        "environment_and_execution_counts": {
            r["condition"]: {key: r[key] for key in [
                "N", "installed", "test_pass", "all_test_nonpass", "no_installed_environment",
                "installed_test_nonpass", "no_environment_among_nonpasses_pct", "installed_test_pass_pct"]}
            for r in execution_rows if r["model"] == "All observations"
        },
        "sources_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, paired)},
    }
    (analysis / "rq3_results_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, stage_rows, execution_rows


if __name__ == "__main__":
    summary, stages, execution = build_evidence()
    aggregate = [r for r in stages if r["model"] == "All observations" and r["stage"] == "installation_failure"]
    print(json.dumps({"stage_rows": len(stages), "execution_rows": len(execution),
                      "installation_failures_among_complete_locks": aggregate,
                      "minimum_observed_test_nonpass_pct": summary["minimum_observed_static_passing_nonpass_group"]["static_test_nonpass_pct"]}))
