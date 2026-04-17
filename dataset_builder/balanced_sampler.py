"""Fair sampling across ``target_packages`` for fixed-size datasets.

Usage: imported by ``dataset_builder.build_stackoverflow_dataset`` (no CLI).

    from dataset_builder.balanced_sampler import sample_balanced_records
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger
from tqdm import tqdm


def sample_balanced_records(
    *,
    run_dir: Path,
    input_jsonl: str,
    output_jsonl: str,
    stats_json: str,
    target_total: int = 5000,
    seed: int = 42,
) -> dict:
    in_path = run_dir / input_jsonl
    out_path = run_dir / output_jsonl
    stats_path = run_dir / stats_json
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    random.seed(seed)
    records: list[dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Load records for balanced sampling"):
            obj = json.loads(line)
            tpls = sorted(
                set([x for x in (obj.get("target_packages") or []) if isinstance(x, str) and x.strip()])
            )
            if tpls:
                obj["_tpls"] = tpls
                records.append(obj)

    tpl_set = sorted({tpl for r in records for tpl in r["_tpls"]})
    k = len(tpl_set)
    if k == 0:
        out_path.write_text("", encoding="utf-8")
        stats = {
            "source": str(in_path),
            "output": str(out_path),
            "target_total": target_total,
            "actual_total": 0,
            "tpl_count": 0,
        }
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return stats

    base = target_total // k
    rem = target_total % k
    quota = {tpl: base for tpl in tpl_set}
    for tpl in tpl_set[:rem]:
        quota[tpl] += 1

    tpl_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        for tpl in r["_tpls"]:
            tpl_to_indices[tpl].append(i)
    for tpl in tpl_set:
        random.shuffle(tpl_to_indices[tpl])

    selected: set[int] = set()
    assigned_tpl = Counter()

    # Phase1: fill each tpl quota as much as possible.
    for tpl in tpl_set:
        need = quota[tpl]
        if need <= 0:
            continue
        for idx in tpl_to_indices[tpl]:
            if assigned_tpl[tpl] >= need:
                break
            if idx in selected:
                continue
            selected.add(idx)
            assigned_tpl[tpl] += 1

    # Phase2: fill target_total by assigning to most underfilled tpl among candidates.
    all_indices = list(range(len(records)))
    random.shuffle(all_indices)
    while len(selected) < target_total:
        progressed = False
        for idx in all_indices:
            if idx in selected:
                continue
            tpl_candidates = records[idx]["_tpls"]
            best_tpl = min(
                tpl_candidates,
                key=lambda t: (
                    (assigned_tpl[t] / quota[t]) if quota[t] > 0 else 1e9,
                    assigned_tpl[t],
                    t,
                ),
            )
            selected.add(idx)
            assigned_tpl[best_tpl] += 1
            progressed = True
            if len(selected) >= target_total:
                break
        if not progressed:
            break

    selected_indices = sorted(selected)[:target_total]
    selected_records = [records[i] for i in selected_indices]
    with out_path.open("w", encoding="utf-8") as w:
        for r in selected_records:
            r.pop("_tpls", None)
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    vals = [assigned_tpl[t] for t in tpl_set]
    mean_v = sum(vals) / len(vals) if vals else 0.0
    var_v = sum((x - mean_v) ** 2 for x in vals) / len(vals) if vals else 0.0
    std_v = math.sqrt(var_v)
    stats = {
        "source": str(in_path),
        "output": str(out_path),
        "target_total": target_total,
        "actual_total": len(selected_records),
        "tpl_count": k,
        "quota_base": base,
        "quota_remainder": rem,
        "seed": seed,
        "min_assigned": min(vals) if vals else 0,
        "max_assigned": max(vals) if vals else 0,
        "mean_assigned": mean_v,
        "std_assigned": std_v,
        "top10_assigned_tpl": assigned_tpl.most_common(10),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Balanced sampling completed: {stats}")
    return stats

