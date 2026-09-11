"""Discover frozen M1 outputs and build validated M2-M6 revision jobs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawRun:
    track: str
    mode: str
    model: str
    run_name: str
    path: Path
    task_count: int
    task_ids: frozenset[str]

    @property
    def model_key(self) -> str:
        return self.model.strip().lower()


def _parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected TRACK=PATH, got {value!r}")
        key, raw_path = value.split("=", 1)
        result[key] = Path(raw_path).resolve()
    return result


def _read_raw_metadata(path: Path, track: str) -> tuple[str, frozenset[str]]:
    model: str | None = None
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: expected JSON object")
            config = row.get("llm_config") or {}
            row_model = config.get("api_model") or row.get("api_model")
            if row_model:
                if model is not None and model != str(row_model):
                    raise ValueError(f"{path}: mixed api_model values")
                model = str(row_model)
            task_id = row.get("task_id") if track == "d1" else row.get("record_id")
            if task_id is None:
                raise ValueError(f"{path}:{line_no}: missing task identifier")
            ids.add(str(task_id))
    if not model:
        raise ValueError(f"{path}: cannot determine model from llm_config.api_model")
    return model, frozenset(ids)


def _run_name_matches_mode(track: str, mode: str, run_name: str) -> bool:
    """Recognize both main-study and mitigation run directory conventions."""
    if track == "d1":
        return run_name.endswith(f"_{mode}")
    if mode == "inline":
        return run_name.endswith("_blind_inline")
    if mode == "requirements.txt":
        return run_name.endswith("_blind_requirements_txt")
    return run_name.endswith(f"_{mode}")


def discover_runs(source_roots: dict[str, Path], tracks: list[str], modes: list[str]) -> list[RawRun]:
    runs: list[RawRun] = []
    for track in tracks:
        root = source_roots[track]
        for mode in modes:
            mode_root = root / mode
            for path in sorted(mode_root.glob("*/*.jsonl")):
                run_name = path.parent.name
                if not _run_name_matches_mode(track, mode, run_name):
                    continue
                model, task_ids = _read_raw_metadata(path, track)
                runs.append(RawRun(track, mode, model, run_name, path.resolve(), len(task_ids), task_ids))
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=["d1", "d2"], required=True)
    parser.add_argument("--source-roots", nargs="+", required=True, metavar="TRACK=PATH")
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--python-versions", nargs="+", required=True)
    parser.add_argument("--stages", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--analysis-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-models", type=int, default=10)
    parser.add_argument("--expected-d1-tasks", type=int, default=813)
    parser.add_argument("--expected-d2-tasks", type=int, default=1000)
    parser.add_argument("--skip-vuln", action="store_true")
    parser.add_argument("--uv-bin", default="uv")
    parser.add_argument("--bcb-task-ids-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = _parse_mapping(args.source_roots)
    missing_roots = sorted(set(args.tracks) - set(roots))
    if missing_roots:
        raise ValueError(f"Missing source roots for: {missing_roots}")
    runs = discover_runs(roots, args.tracks, args.modes)

    groups: dict[tuple[str, str], dict[str, RawRun]] = {}
    for run in runs:
        key = (run.track, run.mode)
        if run.model_key in groups.setdefault(key, {}):
            other = groups[key][run.model_key]
            raise ValueError(f"Duplicate raw run for {key} model={run.model}: {other.path} and {run.path}")
        groups[key][run.model_key] = run

    required_groups = [(track, mode) for track in args.tracks for mode in args.modes]
    for key in required_groups:
        if key not in groups:
            raise ValueError(f"No raw runs discovered for {key}")
    selected_models = set.intersection(*(set(groups[key]) for key in required_groups))
    if len(selected_models) != args.expected_models:
        raise ValueError(
            f"Expected {args.expected_models} models common to all groups, found "
            f"{len(selected_models)}: {sorted(selected_models)}"
        )
    ignored_models_by_group = {
        key: sorted(set(groups[key]) - selected_models)
        for key in required_groups
        if set(groups[key]) != selected_models
    }

    canonical_ids: dict[str, frozenset[str]] = {}
    for track in args.tracks:
        track_runs = [
            groups[(track, mode)][model]
            for mode in args.modes
            for model in selected_models
        ]
        canonical_ids[track] = track_runs[0].task_ids
        mismatched = [run.path for run in track_runs if run.task_ids != canonical_ids[track]]
        if mismatched:
            raise ValueError(
                f"{track} task-id populations differ across selected models/modes; examples: {mismatched[:3]}"
            )

    jobs: list[dict] = []
    for track, mode in required_groups:
        expected_tasks = args.expected_d1_tasks if track == "d1" else args.expected_d2_tasks
        for model in sorted(selected_models):
            run = groups[(track, mode)][model]
            if run.task_count != expected_tasks:
                raise ValueError(
                    f"{run.path}: expected {expected_tasks} unique tasks, found {run.task_count}"
                )
            for python_version in args.python_versions:
                job = {
                    "track": track,
                    "run_name": run.run_name,
                    "stages": args.stages,
                    "llm_output_path": str(run.path),
                    "analysis_output_root": str((args.analysis_output_root / track).resolve()),
                    "python_version": python_version,
                    "resolver": "uv",
                    "uv_bin": args.uv_bin,
                    "python_platform": "x86_64-unknown-linux-gnu",
                    "exclude_newer": "2026-02-01T00:00:00Z",
                    "prerelease": "if-necessary-or-explicit",
                    "ty_version": "0.0.23",
                    "workers": 4,
                }
                if track == "d1":
                    job["prompt_mode"] = mode
                else:
                    job["prompt_mode"] = "blind"
                    job["pinning_mode"] = mode
                if args.skip_vuln:
                    job["skip_vuln"] = True
                jobs.append(job)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
    if args.bcb_task_ids_output and "d1" in canonical_ids:
        args.bcb_task_ids_output.parent.mkdir(parents=True, exist_ok=True)
        with args.bcb_task_ids_output.open("w", encoding="utf-8") as handle:
            for task_id in sorted(canonical_ids["d1"]):
                handle.write(json.dumps({"task_id": task_id}) + "\n")
    print(json.dumps({
        "jobs": len(jobs),
        "models": sorted(selected_models),
        "ignored_models_by_group": {f"{key[0]}:{key[1]}": value for key, value in ignored_models_by_group.items()},
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
