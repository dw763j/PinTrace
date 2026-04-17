#!/usr/bin/env python3
"""
Scan pipeline output tree for LLM inference JSONL files and summarize progress.

Supports both D1 and D2 with shared scanning logic:
- D1 inference line: has ``task_id`` and ``llm_output``
- D2 inference line: has ``record_id`` and ``llm_output``

Summary groups by: track x variant_mode x context_mode
- D1: variant_mode is prompt-mode-like value (inline / requirements.txt / ...)
- D2: variant_mode is pinning_mode (inline / requirements.txt / ...)

Usage:
  python scripts/summarize_inference_jsonl.py
  python -m scripts.summarize_inference_jsonl --incomplete-only
  python -m scripts.summarize_inference_jsonl --track d1 --outputs-d1 outputs/d1
  python -m scripts.summarize_inference_jsonl --track d2 --outputs-d2 outputs/d2 --format json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWN_D1_VARIANT_DIRS = frozenset({"inline", "inline_no_vuln", "requirements.txt", "inline_safe_version", "inline_api_rag"})
KNOWN_D2_VARIANT_DIRS = frozenset({"inline", "inline_no_vuln", "requirements.txt", "inline_safe_version", "inline_api_rag"})


@dataclass
class FileStat:
    path: Path
    rel_path: str
    track: str
    records: int
    variant_mode: str
    context_mode: str
    model_name: str | None


@dataclass
class GroupTotals:
    files: int = 0
    records: int = 0
    by_model: dict[str, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize D1/D2 inference JSONL counts by mode.")
    p.add_argument("--track", choices=["d1", "d2", "both"], default="both", help="Which track(s) to scan.")
    p.add_argument("--outputs-d1", type=Path, default=PROJECT_ROOT / "outputs" / "d1", help="Root outputs/d1 directory.")
    p.add_argument("--outputs-d2", type=Path, default=PROJECT_ROOT / "outputs" / "d2", help="Root outputs/d2 directory.")
    p.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    p.add_argument("--incomplete-only", action="store_true", help="Only show incomplete experiments (D1: not 813/975, D2: not 1000).")
    return p.parse_args()


def is_completed(track: str, records: int) -> bool:
    """Check if an experiment is completed based on track and record count."""
    if track == "d1":
        return records in (813, 975)
    elif track == "d2":
        return records == 1000
    return False


def main() -> None:
    args = parse_args()

    requested = [args.track] if args.track in {"d1", "d2"} else ["d1", "d2"]
    roots: dict[str, Path] = {}
    all_stats: list[FileStat] = []
    groups: dict[tuple[str, str, str], GroupTotals] = defaultdict(GroupTotals)

    for track in requested:
        root = Path(args.outputs_d1).resolve() if track == "d1" else Path(args.outputs_d2).resolve()
        roots[track] = root
        if not root.is_dir():
            continue

        for path in sorted(root.rglob("*.jsonl")):
            if not path.is_file():
                continue

            first_obj: dict[str, Any] | None = None
            detected_track: str | None = None
            n = 0
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        raw = line.strip()
                        if not raw:
                            continue
                        n += 1
                        if first_obj is not None:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            first_obj = None
                            detected_track = None
                            break
                        if not isinstance(obj, dict):
                            first_obj = None
                            detected_track = None
                            break
                        if bool(obj.get("task_id")) and "llm_output" in obj:
                            detected_track = "d1"
                        elif bool(obj.get("record_id")) and "llm_output" in obj:
                            detected_track = "d2"
                        else:
                            first_obj = None
                            detected_track = None
                            break
                        first_obj = obj
            except OSError:
                continue

            if first_obj is None or detected_track != track:
                continue

            if track == "d2":
                mode_raw = first_obj.get("pinning_mode")
                known_dirs = KNOWN_D2_VARIANT_DIRS
            else:
                mode_raw = first_obj.get("prompt_mode")
                known_dirs = KNOWN_D1_VARIANT_DIRS
            if isinstance(mode_raw, str) and mode_raw.strip():
                variant = mode_raw.strip()
            else:
                try:
                    rel_for_mode = path.resolve().relative_to(root.resolve())
                    variant = rel_for_mode.parts[0] if rel_for_mode.parts and rel_for_mode.parts[0] in known_dirs else ""
                except ValueError:
                    variant = ""
                if not variant:
                    run_dir_name = path.parent.name
                    if "inline_no_vuln" in run_dir_name:
                        variant = "inline_no_vuln"
                    elif "requirements_txt" in run_dir_name:
                        variant = "requirements.txt"
                    elif "inline" in run_dir_name:
                        variant = "inline"
                    elif track == "d1" and "requirements" in run_dir_name:
                        variant = "requirements.txt"
                    else:
                        variant = "unknown"

            context_raw = first_obj.get("prompt_mode")
            if isinstance(context_raw, str) and context_raw.strip():
                context = context_raw.strip()
            else:
                stem = path.stem
                run_dir_name = path.parent.name
                if stem.endswith("_hint") or "_hint_" in run_dir_name or run_dir_name.endswith("_hint"):
                    context = "hint"
                elif stem.endswith("_blind") or "_blind_" in run_dir_name or run_dir_name.endswith("_blind"):
                    context = "blind"
                elif track == "d1":
                    context = variant
                else:
                    context = "unknown"

            model: str | None = None
            cfg = first_obj.get("llm_config")
            if isinstance(cfg, dict):
                model_raw = cfg.get("api_model")
                if isinstance(model_raw, str) and model_raw.strip():
                    model = model_raw.strip()

            try:
                rel_path = str(path.resolve().relative_to(root))
            except ValueError:
                rel_path = str(path)

            stat = FileStat(
                path=path,
                rel_path=rel_path,
                track=track,
                records=n,
                variant_mode=variant,
                context_mode=context,
                model_name=model,
            )
            all_stats.append(stat)
            g = groups[(track, variant, context)]
            g.files += 1
            g.records += n
            model_key = model or "(unknown)"
            g.by_model[model_key] = g.by_model.get(model_key, 0) + n

    if args.incomplete_only:
        all_stats = [s for s in all_stats if not is_completed(s.track, s.records)]
        groups = defaultdict(GroupTotals)
        for stat in all_stats:
            g = groups[(stat.track, stat.variant_mode, stat.context_mode)]
            g.files += 1
            g.records += stat.records
            model_key = stat.model_name or "(unknown)"
            g.by_model[model_key] = g.by_model.get(model_key, 0) + stat.records

    if args.format == "json":
        payload = {
            "summary_by_mode": [
                {
                    "track": track,
                    "variant_mode": variant,
                    "context_mode": context,
                    "jsonl_files": g.files,
                    "records": g.records,
                    "by_model": dict(sorted(g.by_model.items())),
                }
                for (track, variant, context), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
            ],
            "files": [
                {
                    "track": s.track,
                    "rel_path": s.rel_path,
                    "records": s.records,
                    "variant_mode": s.variant_mode,
                    "context_mode": s.context_mode,
                    "model_name": s.model_name,
                }
                for s in sorted(all_stats, key=lambda x: (x.track, x.rel_path))
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    for track in ("d1", "d2"):
        if track in roots:
            print(f"Root ({track}): {roots[track]}")
    if args.incomplete_only:
        print("\n*** Showing only incomplete experiments (D1: not 813/975, D2: not 1000) ***\n")
    if not all_stats:
        print("No inference JSONL found.")
        return

    print("\n=== By track x variant_mode x context_mode ===")
    for (track, variant, context) in sorted(groups.keys(), key=lambda x: (x[0], x[1], x[2])):
        g = groups[(track, variant, context)]
        print(f"  {track:2s}  {variant!r} + {context!r}:  runs={g.files}  records={g.records}")

    print("\n=== Per file ===")
    for s in sorted(all_stats, key=lambda x: (x.track, x.variant_mode, x.context_mode, x.rel_path)):
        model = s.model_name or "?"
        print(f"  {s.records:5d}  {s.track:2s}  {s.variant_mode:18s}  {s.context_mode:12s}  {model:40s}  {s.rel_path}")

    print("\n=== Totals ===")
    print(f"  JSONL files: {len(all_stats)}")
    print(f"  Records:     {sum(s.records for s in all_stats)}")


if __name__ == "__main__":
    main()
