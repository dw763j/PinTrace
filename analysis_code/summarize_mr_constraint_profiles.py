import os
"""Summarize complete MR manifests and compare their forms within matched tasks."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

DISPLAY_NAMES = {
    "gpt-5.4": "GPT-5.4", "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "deepseek-v3.2": "DeepSeek V3.2",
    "kimi-k2.5": "Kimi K2.5", "qwen3.5-397b-a17b": "Qwen3.5 397B",
    "qwen3-235b-a22b-instruct-2507": "Qwen3 235B",
    "qwen3-30b-a3b-instruct-2507": "Qwen3 30B", "minimax-m2.5": "MiniMax M2.5",
    "meta-llama/llama-4-scout": "Llama 4 Scout"}
CATEGORIES = ["bare_only", "ranged_only", "exact_only", "mixed", "contains_unparseable", "no_manifest_entries"]


def records(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def refresh_mr_profiles(root, declarations=None, responses=None):
    root = Path(root)
    source = root / "inputs/main"
    if declarations is None:
        declarations = records(source / "source_carrier_declarations.jsonl")
    if responses is None:
        responses = records(source / "task_outcomes.jsonl")
    grouped = defaultdict(set)
    for record in declarations:
        if record["track"] == "d2" and record["mode"] == "requirements.txt" and record["rq1_eligible"]:
            grouped[(record["model"], record["task_id"])].add(record["constraint_kind"])
    selected = [r for r in responses if r["track"] == "d2" and r["mode"] == "requirements.txt"]
    assert len(selected) == len({(r["model"], r["task_id"]) for r in selected})
    models = list(DISPLAY_NAMES)
    assert {r["model"] for r in selected} == set(models)
    counts = defaultdict(Counter)
    task_profiles = defaultdict(dict)
    for response in selected:
        kinds = grouped[(response["model"], response["task_id"])]
        assert kinds <= {"bare", "range", "exact", "invalid"}
        if not kinds:
            category = "no_manifest_entries"
        elif "invalid" in kinds:
            category = "contains_unparseable"
        elif len(kinds) > 1:
            category = "mixed"
        else:
            category = {"bare": "bare_only", "range": "ranged_only", "exact": "exact_only"}[next(iter(kinds))]
        counts[response["model"]][category] += 1
        task_profiles[response["task_id"]][response["model"]] = category
    profiles = [dict(model=model, N=sum(counts[model].values()), **{c: counts[model][c] for c in CATEGORIES})
                for model in models]
    assert all(r["N"] == 1000 for r in profiles)
    aggregate = dict(model="All observations", N=sum(r["N"] for r in profiles),
                     **{c: sum(r[c] for r in profiles) for c in CATEGORIES})
    profiles.append(aggregate)
    assert all(sum(r[c] for c in CATEGORIES) == r["N"] for r in profiles)
    tasks = []
    for task, model_profiles in sorted(task_profiles.items()):
        assert set(model_profiles) == set(models)
        task_counts = Counter(model_profiles.values())
        tasks.append(dict(task_id=task, model_count=len(models),
                          **{c + "_models": task_counts[c] for c in CATEGORIES},
                          all_three_single_forms=all(task_counts[c] > 0 for c in CATEGORIES[:3])))
    assert len(tasks) == 1000 and aggregate["N"] == 10000
    summary = dict(population="Full PinTrace, MR", tasks=len(tasks), models=len(models),
                   responses=aggregate["N"], tasks_all_three_single_forms=sum(r["all_three_single_forms"] for r in tasks),
                   models_majority_bare_only=sum(r["bare_only"] > r["N"] / 2 for r in profiles[:-1]),
                   source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in ["task_outcomes.jsonl", "source_carrier_declarations.jsonl"]})
    summary["tasks_all_three_single_forms_pct"] = 100 * summary["tasks_all_three_single_forms"] / len(tasks)
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    for filename, data in [("rq1_mr_manifest_profiles.csv", profiles), ("rq1_mr_task_form_diversity.csv", tasks)]:
        with (analysis / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (analysis / "rq1_mr_profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    headers = ["Model", "Bare-only (%)", "Ranged-only (%)", "Exact-only (%)", "Mixed (%)", "Unparseable (%)", "No entries (%)"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in profiles:
        name = DISPLAY_NAMES.get(row["model"], row["model"])
        lines.append("| " + " | ".join([name] + [f"{100*row[c]/row['N']:.2f}" for c in CATEGORIES]) + " |")
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "rq1_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profiles, tasks, summary


if __name__ == "__main__":
    _, _, result = refresh_mr_profiles(Path(os.environ.get("PINTRACE_WORKSPACE", Path(__file__).resolve().parents[1])))
    print(json.dumps(result, indent=2))
