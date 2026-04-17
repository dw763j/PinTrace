"""Shared model display names and ordering for plots.

Usage (import-only; plotting scripts call into this module)::

    from plots.model_display import normalize_model_key, paper_display_label

This module centralizes:
- Canonical model keys (as used in outputs / metrics).
- Paper-friendly display labels.
- A stable figure ordering.
- Helpers to infer model names from D2 run directory structure.

All model-key matching is done via ``normalize_model_key`` (lower, strip).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


# Paper-friendly short labels (keep insertion order as the canonical figure order).
MODEL_LABEL: dict[str, str] = {
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4-6": "Claude-Sonnet-4.6",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro",
    "DeepSeek-V3.2": "DeepSeek-V3.2",
    "kimi-k2.5": "Kimi-K2.5",
    "Qwen3.5-397B-A17B": "Qwen3.5-397B",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen3-235B",
    "Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B",
    "MiniMax-M2.5": "MiniMax-M2.5",
    "meta-llama_llama-4-scout": "Llama-4-Scout",
}


# D2 plots may want to hide certain runs by default.
EXCLUDED_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "gpt-5.2",
        "gemini-3-pro-preview"
    }
)


def normalize_model_key(name: str) -> str:
    return name.strip().lower()


def order_model_keys(model_keys: Iterable[str]) -> list[str]:
    """Order keys by MODEL_LABEL insertion order; leftovers are appended alphabetically."""
    present = {normalize_model_key(k) for k in model_keys}
    out: list[str] = []
    for canon in MODEL_LABEL:
        mk = normalize_model_key(canon)
        if mk in present:
            out.append(mk)
    rest = sorted(present - set(out), key=str.lower)
    out.extend(rest)
    return out


def is_excluded_model(mk: str, raw_name: str) -> bool:
    mkn = normalize_model_key(mk)
    if mkn in EXCLUDED_MODEL_KEYS:
        return True
    return normalize_model_key(raw_name) in EXCLUDED_MODEL_KEYS


def read_model_name_from_metrics(run_dir: Path) -> str | None:
    """Read model_name from metrics_summary.json under a run's py-tag directory."""
    ms = run_dir / "metrics_summary.json"
    if not ms.is_file():
        return None
    try:
        obj = json.loads(ms.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mn = obj.get("model_name")
    return str(mn).strip() if mn else None


def _short_model_name(run_dir_name: str) -> str:
    name = run_dir_name
    if name.startswith("d2_"):
        name = name[3:]
    for suffix in ("_blind_inline", "_blind_requirements", "_inline", "_requirements"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def infer_run_dir_name_from_m4_path(m4_path: Path) -> str:
    """Infer run dir name from ``.../<run_dir>/py312/m4_vuln_records.json``."""
    p = m4_path.parent
    n = p.name
    if len(n) >= 4 and n.lower().startswith("py") and n[2:].isdigit():
        return m4_path.parents[1].name
    return n


def infer_model_raw_name_from_m4(m4_path: Path) -> str:
    """Prefer metrics_summary.json; otherwise fall back to run dir name."""
    from_metrics = read_model_name_from_metrics(m4_path.parent)
    if from_metrics:
        return from_metrics
    return _short_model_name(infer_run_dir_name_from_m4_path(m4_path))


def paper_display_label(mk: str, raw_name: str) -> str:
    """Map internal model key/name to paper label; fallback to raw_name."""
    mk_norm = normalize_model_key(mk)
    for key, label in MODEL_LABEL.items():
        if normalize_model_key(key) == mk_norm:
            return label
    raw_norm = normalize_model_key(raw_name)
    for key, label in MODEL_LABEL.items():
        if normalize_model_key(key) == raw_norm:
            return label
    compact = raw_name.lower().replace("-", "").replace("_", "").replace("/", "")
    for key, label in MODEL_LABEL.items():
        kn = normalize_model_key(key).replace("_", "").replace("-", "")
        if len(kn) >= 6 and kn in compact:
            return label
    return raw_name

