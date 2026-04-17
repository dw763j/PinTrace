"""Helpers for locating uv caches and subprocess-friendly paths.

Usage (import-only)::

    from stages.uv_runtime import get_uv_cache_dir
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from loguru import logger

_LAST_CACHE_CHECK_TS = 0.0


def get_uv_cache_dir() -> Path:
    uv_cache = os.environ.get("UV_CACHE_DIR")
    if uv_cache:
        return Path(uv_cache).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser().resolve() / "uv"
    return Path.home() / ".cache" / "uv"


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True)


def _get_size_from_uv_command(cache_dir: Path) -> int | None:
    """Use uv native cache size command (fast path)."""
    args = [
        "uv",
        "cache",
        "size",
        "--preview-features",
        "cache-size",
        "--cache-dir",
        str(cache_dir),
    ]
    proc = _run_command(args)
    if proc.returncode != 0:
        logger.debug(f"[uv-cache] uv size command failed: stderr={proc.stderr[:300]}")
        return None
    text = proc.stdout.strip()
    # uv cache size outputs bytes as plain integer in default mode.
    if text.isdigit():
        return int(text)
    return None


def _get_size_from_du(cache_dir: Path) -> int | None:
    """Use system du as a fallback before Python traversal."""
    proc = _run_command(["du", "-sb", str(cache_dir)])
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    token = proc.stdout.strip().split()[0]
    return int(token) if token.isdigit() else None


def get_cache_size_bytes(cache_dir: Path) -> int:
    """Best-effort cache size with native uv command first."""
    if not cache_dir.exists():
        return 0
    size = _get_size_from_uv_command(cache_dir)
    if size is not None:
        return size
    size = _get_size_from_du(cache_dir)
    if size is not None:
        return size

    # Last resort Python traversal (slower on large trees).
    total = 0
    for p in cache_dir.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _run_uv_cache_cmd(command: str) -> None:
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(f"[uv-cache] command failed: {command}; stderr={proc.stderr[:300]}")
    else:
        logger.info(f"[uv-cache] command ok: {command}")


def enforce_uv_cache_limit(
    *,
    max_gb: int = 300,
    check_interval_seconds: int = 300,
) -> None:
    """Keep uv cache under limit; prune then clean when still over limit."""
    global _LAST_CACHE_CHECK_TS
    now = time.time()
    if now - _LAST_CACHE_CHECK_TS < check_interval_seconds:
        return
    _LAST_CACHE_CHECK_TS = now

    cache_dir = get_uv_cache_dir()
    current = get_cache_size_bytes(cache_dir)
    limit_bytes = max_gb * 1024 * 1024 * 1024
    logger.info(f"[uv-cache] size={current / (1024**3):.2f}GB limit={max_gb}GB dir={cache_dir}")
    if current <= limit_bytes:
        return

    logger.warning("[uv-cache] cache exceeds limit, running prune.")
    _run_uv_cache_cmd("uv cache prune")
    after_prune = get_cache_size_bytes(cache_dir)
    logger.info(f"[uv-cache] after prune size={after_prune / (1024**3):.2f}GB")
    if after_prune <= limit_bytes:
        return

    logger.warning("[uv-cache] cache still exceeds limit, running clean.")
    _run_uv_cache_cmd("uv cache clean")
    after_clean = get_cache_size_bytes(cache_dir)
    logger.info(f"[uv-cache] after clean size={after_clean / (1024**3):.2f}GB")

