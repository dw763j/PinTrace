#!/usr/bin/env python3
"""
Cleanup selected stage artifacts under outputs/d1 or outputs/d2.

Examples:
  python -m scripts.cleanup_stage_results --track d1 --stages 2 3 4 5 --dry-run
  python -m scripts.cleanup_stage_results --track d2 --stages 2 3 4 5 --dry-run
  python -m scripts.cleanup_stage_results --track d2 --all-except-llm --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_VARIANT_MODES = ["inline", "inline_no_vuln", "requirements.txt"]
D2_STAGE_CHECKPOINT_FILES = {
    3: "version_resolution.json",
    4: "vuln.json",
    5: "compat.json",
}

D1_STAGE_PAPER_SUMMARY_FILES = {
    2: "m2_extraction_summary.json",
    3: "m3_resolution_summary.json",
    4: "m4_vulnerability_summary.json",
    5: "m5_compatibility_summary.json",
}


def parse_args(default_track: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete selected D1/D2 pipeline outputs.")
    if default_track is None:
        parser.add_argument("--track", choices=["d1", "d2"], required=True, help="Track to clean.")
    parser.add_argument("--outputs-d1", type=Path, default=Path("outputs") / "d1", help="Root outputs/d1 directory.")
    parser.add_argument("--outputs-d2", type=Path, default=Path("outputs") / "d2", help="Root outputs/d2 directory.")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_VARIANT_MODES, help="D1 prompt modes to include (subdirs under outputs/d1).")
    parser.add_argument("--pinning-modes", nargs="+", default=DEFAULT_VARIANT_MODES, help="D2 pinning modes to include (subdirs under outputs/d2).")
    parser.add_argument("--stages", type=int, nargs="+", help="Stage numbers to clean. For D2, valid range is 1-6. Ignored when --all-except-llm is set.")
    parser.add_argument("--all-except-llm", dest="all_except_llm", action="store_true", help="Delete all artifacts except top-level LLM outputs (*.jsonl) in each run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be deleted without actually deleting.")
    args = parser.parse_args()
    if default_track is not None:
        args.track = default_track
    return args


def main(default_track: str | None = None) -> None:
    args = parse_args(default_track=default_track)
    track: str = args.track

    outputs_root: Path = args.outputs_d1 if track == "d1" else args.outputs_d2
    if not outputs_root.is_dir():
        print(f"[ERROR] outputs root not found for {track}: {outputs_root}", file=sys.stderr)
        sys.exit(1)

    run_dirs: list[Path] = []
    if track == "d1":
        for mode in args.modes:
            mode_dir = outputs_root / mode
            if not mode_dir.is_dir():
                continue
            for sub_dir in sorted(mode_dir.iterdir()):
                if sub_dir.is_dir():
                    run_dirs.append(sub_dir)
    else:
        run_dir_set: set[Path] = set()
        for mode in args.pinning_modes:
            mode_dir = outputs_root / mode
            if not mode_dir.is_dir():
                continue
            for sub_dir in sorted(mode_dir.iterdir()):
                if sub_dir.is_dir():
                    run_dir_set.add(sub_dir)

        for sub_dir in sorted(outputs_root.iterdir()):
            if sub_dir.is_dir() and sub_dir.name.startswith("d2_"):
                run_dir_set.add(sub_dir)

        run_dirs = sorted(run_dir_set)

    if not run_dirs:
        selected_modes = args.modes if track == "d1" else args.pinning_modes
        print(f"[INFO] No run directories found under {outputs_root} for {track} modes={selected_modes}")
        sys.exit(0)

    to_delete: list[Path] = []

    if args.all_except_llm:
        print("[MODE] Full cleanup except stage-1 LLM outputs (*.jsonl).")
        for run_dir in run_dirs:
            full_cleanup_targets: list[Path] = []

            if track == "d1":
                for subdir_name in ("cache", "checkpoints", "paper"):
                    subdir_path = run_dir / subdir_name
                    if subdir_path.exists():
                        full_cleanup_targets.append(subdir_path)

            for file_or_dir in run_dir.iterdir():
                if file_or_dir.is_dir():
                    if track == "d2":
                        full_cleanup_targets.append(file_or_dir)
                    continue
                if file_or_dir.suffix == ".jsonl":
                    continue
                full_cleanup_targets.append(file_or_dir)

            to_delete.extend(sorted(set(full_cleanup_targets)))
    else:
        if not args.stages:
            print("[ERROR] --stages is required when --all-except-llm is not set.", file=sys.stderr)
            sys.exit(1)

        stages = sorted(set(args.stages))
        if track == "d2":
            invalid = [s for s in stages if s < 1 or s > 6]
            if invalid:
                print(f"[ERROR] Invalid D2 stages: {invalid}. Valid range is 1-6.", file=sys.stderr)
                sys.exit(1)

        per_stage_counts: dict[int, int] = {s: 0 for s in stages}
        print(f"[MODE] Stage-based cleanup for track={track}, stages={stages}.")
        for run_dir in run_dirs:
            py_subdirs = []
            if track in ("d1", "d2"):
                py_subdirs = sorted(
                    sub_dir for sub_dir in run_dir.iterdir() if sub_dir.is_dir() and sub_dir.name.startswith("py")
                )

            for stage in stages:
                stage_paths: list[Path] = []
                if track == "d1":
                    if stage == 6:
                        for py_dir in py_subdirs:
                            for summary_name in ("metrics_summary.json", "metrics_summary.csv"):
                                summary_path = py_dir / summary_name
                                if summary_path.exists():
                                    stage_paths.append(summary_path)
                    elif stage in (2, 3, 4, 5):
                        stage_pattern = f"m{stage}_*"
                        checkpoint_name = D2_STAGE_CHECKPOINT_FILES.get(stage)
                        paper_summary_name = D1_STAGE_PAPER_SUMMARY_FILES.get(stage)
                        for py_dir in py_subdirs:
                            stage_paths.extend(sorted(py_dir.glob(stage_pattern)))
                            if checkpoint_name:
                                checkpoint_path = py_dir / "checkpoints" / checkpoint_name
                                if checkpoint_path.exists():
                                    stage_paths.append(checkpoint_path)
                            if paper_summary_name:
                                paper_summary_path = py_dir / "paper" / paper_summary_name
                                if paper_summary_path.exists():
                                    stage_paths.append(paper_summary_path)
                    else:
                        for pattern in (f"m{stage}_*", f"m{stage}.*"):
                            stage_paths.extend(run_dir.glob(pattern))
                elif stage == 1:
                    stage_paths.extend(sorted(run_dir.glob("*.jsonl")))
                elif stage in (2, 3, 4, 5):
                    stage_pattern = f"m{stage}_*"
                    checkpoint_name = D2_STAGE_CHECKPOINT_FILES.get(stage)
                    for py_dir in py_subdirs:
                        stage_paths.extend(sorted(py_dir.glob(stage_pattern)))
                        if checkpoint_name:
                            checkpoint_path = py_dir / "checkpoints" / checkpoint_name
                            if checkpoint_path.exists():
                                stage_paths.append(checkpoint_path)
                else:  # stage == 6
                    for py_dir in py_subdirs:
                        for summary_name in ("metrics_summary.json", "metrics_summary.csv"):
                            summary_path = py_dir / summary_name
                            if summary_path.exists():
                                stage_paths.append(summary_path)

                unique_stage_paths = sorted(set(stage_paths))
                to_delete.extend(unique_stage_paths)
                per_stage_counts[stage] += len(unique_stage_paths)

        print("Per-stage path counts:")
        for stage in stages:
            print(f"  - stage {stage}: {per_stage_counts[stage]} paths")

    to_delete = sorted(set(to_delete))

    print(f"\n[INFO] track        = {track}")
    print(f"[INFO] outputs_root = {outputs_root}")
    if track == "d1":
        print(f"[INFO] modes        = {args.modes}")
    else:
        print(f"[INFO] pinning_modes = {args.pinning_modes}")
    print(f"[INFO] discovered {len(run_dirs)} run dirs, {len(to_delete)} paths to delete in total.\n")

    if not to_delete:
        print("[INFO] Nothing to delete for the specified options.")
        sys.exit(0)

    print("Sample paths to be deleted (up to 20):")
    for path in to_delete[:20]:
        print(f"  {path}")

    print(
        f"\n[WARNING] This will permanently delete the above files/directories under {outputs_root}.\n"
        "          Confirm you have backups before proceeding."
    )

    if args.dry_run:
        print("\n[DRY-RUN] No files were deleted. Re-run without --dry-run to actually delete.")
        sys.exit(0)

    confirm = input("\nType YES to proceed with deletion: ").strip()
    if confirm != "YES":
        print("[INFO] Aborted by user. No files were deleted.")
        sys.exit(0)

    removed = 0
    for path in to_delete:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[WARN] Failed to delete {path}: {exc}", file=sys.stderr)

    print(f"\n[DONE] Deleted {removed} paths for track={track} (all-except-llm={args.all_except_llm}).")


if __name__ == "__main__":
    main()
