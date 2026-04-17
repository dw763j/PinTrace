"""Shared OSV parsing utilities (alias deduplication, etc.).

Usage (import-only)::

    from stages.osv_utils import build_osv_alias_canonical_map
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def build_osv_alias_canonical_map(osv_dir: str) -> dict[str, str]:
    """Build id -> canonical_id mapping. Aliases (id + aliases from same advisory) are the same vuln.
    Uses union-find to merge equivalence classes across files, then pick min id as canonical."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for name in os.listdir(osv_dir):
        path = os.path.join(osv_dir, name)
        if not os.path.isfile(path) or not name.endswith(".json"):
            continue
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        osv_id = data.get("id")
        if not isinstance(osv_id, str) or not osv_id:
            continue
        ids_in_advisory = {osv_id}
        for a in data.get("aliases", []) or []:
            if isinstance(a, str) and a.strip():
                ids_in_advisory.add(a.strip())
        for aid in ids_in_advisory:
            union(osv_id, aid)

    canonical: dict[str, str] = {}
    classes: dict[str, set[str]] = {}
    for x in parent:
        root = find(x)
        classes.setdefault(root, set()).add(x)
    for members in classes.values():
        rep = min(members)
        for m in members:
            canonical[m] = rep
    return canonical
