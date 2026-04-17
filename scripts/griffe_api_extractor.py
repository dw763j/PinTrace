"""Extract package API surface using griffe with cache and resume support.

Usage (import from repository code; no ``python -m`` CLI)::

    from scripts.griffe_api_extractor import ...
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger
from tqdm import tqdm

import griffe

_GRIFFE_LOAD_LOCK = threading.Lock()



@dataclass
class ParamInfo:
    name: str
    kind: str
    annotation: str | None
    has_default: bool


@dataclass
class APIEntry:
    name: str
    kind: str
    parameters: list[ParamInfo]
    return_type: str | None
    docstring: str | None


def _extract_docstring(node: dict[str, Any]) -> str | None:
    """Extract normalized docstring text from griffe JSON node."""
    raw = node.get("docstring")
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, dict):
        # griffe JSON can encode docstring as {"value": "...", ...}.
        value = raw.get("value")
        if isinstance(value, str):
            text = value.strip()
            return text or None
    return None


def _entry_from_griffe_json(qname: str, node: dict[str, Any]) -> APIEntry | None:
    kind = node.get("kind")
    if kind not in {"function", "class", "attribute"}:
        return None

    parameters = []
    for param in node.get("parameters", []):
        parameters.append(
            ParamInfo(
                name=param.get("name", ""),
                kind=param.get("kind", "positional_or_keyword"),
                annotation=str(param.get("annotation")) if param.get("annotation") is not None else None,
                has_default=param.get("default") is not None,
            )
        )

    returns = node.get("returns")
    return APIEntry(
        name=qname,
        kind=kind,
        parameters=parameters,
        return_type=str(returns) if returns is not None else None,
        docstring=_extract_docstring(node),
    )


def _walk_members(prefix: str, node: dict[str, Any], surface: dict[str, APIEntry]) -> None:
    if not isinstance(node, dict):
        return
    entry = _entry_from_griffe_json(prefix, node)
    if entry is not None:
        surface[prefix] = entry

    for child_name, child in (node.get("members") or {}).items():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        _walk_members(child_prefix, child, surface)


def _cache_path(cache_dir: str, package: str, version: str) -> str:
    return os.path.join(cache_dir, package, f"{version}.json")


def _failure_marker_path(
    cache_dir: str,
    package: str,
    version: str,
    distribution: str,
) -> str:
    safe_dist = distribution.replace("/", "_")
    return os.path.join(cache_dir, package, f"{version}__{safe_dist}.failed.json")


def _read_failure_marker(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {"error_type": "unknown", "error_message": "failed marker unreadable"}


def _write_failure_marker(
    path: str,
    *,
    package: str,
    distribution: str,
    version: str,
    error_type: str,
    error_message: str,
) -> None:
    payload = {
        "package": package,
        "distribution": distribution,
        "version": version,
        "error_type": error_type,
        "error_message": error_message[:500],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_api_surface_cache_path(package: str, version: str, *, cache_dir: str = "cache/api_surface") -> str:
    return _cache_path(cache_dir, package, version)


def _load_module_from_pypi(
    package: str,
    version: str,
    *,
    distribution: str | None = None,
    submodules: bool = True,
) -> Any:
    """Load module metadata from PyPI via griffe with explicit options."""
    dist = distribution or package
    # griffe.load_pypi touches import resolution state and is not thread-safe enough
    # under heavy parallel extraction. Serialize this call to avoid transient
    # false-negative checks like "install with [pypi]".
    with _GRIFFE_LOAD_LOCK:
        return griffe.load_pypi(
            package=package,
            distribution=dist,
            version_spec=f"=={version}",
            submodules=submodules,
        )


def _extract_api_surface_internal(
    package: str,
    version: str,
    *,
    distribution: str | None = None,
    cache_dir: str,
) -> tuple[dict[str, dict], str, bool]:
    """Extract API surface and return (surface, extraction_mode, cache_hit)."""
    path = _cache_path(cache_dir, package, version)
    dist = distribution or package
    failure_path = _failure_marker_path(cache_dir, package, version, dist)
    cache_hit = os.path.exists(path)
    if cache_hit:
        logger.info(f"[griffe] API surface cache hit: {package}=={version}")
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        # Backward compatibility: old cache files may not include docstring.
        # Rebuild once so downstream retrieval can use richer textual evidence.
        if cached and not all(
            isinstance(v, dict) and "docstring" in v
            for v in cached.values()
        ):
            logger.info(f"[griffe] Cache schema upgrade needed (docstring): {package}=={version}")
        else:
            return cached, "cache", True

    failed = _read_failure_marker(failure_path)
    if failed is not None:
        logger.warning(
            "[griffe] Skip extraction due to failure marker: "
            f"{package} (dist={dist})=={version}, "
            f"error_type={failed.get('error_type')}"
        )
        return {}, "failed_marker", cache_hit

    if griffe is None:
        raise RuntimeError("griffe is not installed. Please add griffe to requirements and install dependencies.")

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        os.makedirs(os.path.join(xdg_cache_home, "griffe"), exist_ok=True)

    target = f"{package} (dist={dist})" if dist != package else package
    logger.info(f"[griffe] Extracting API surface: {target}=={version}")
    extraction_mode = "full"
    try:
        module = _load_module_from_pypi(
            package, version, distribution=dist, submodules=True
        )
    except AttributeError as e:
        error_msg = str(e)
        if "'Attribute' object has no attribute 'parameters'" in error_msg:
            # Known griffe issue in stub merge path. Fall back to top-level only extraction.
            logger.warning(f"[griffe] Known bug encountered with {target}=={version}: {error_msg}")
            logger.warning(
                f"[griffe] Retrying with submodules=False as fallback: "
                f"{target}=={version}"
            )
            try:
                module = _load_module_from_pypi(
                    package, version, distribution=dist, submodules=False
                )
                extraction_mode = "top_level_fallback"
            except Exception as fallback_exc:
                _write_failure_marker(
                    failure_path,
                    package=package,
                    distribution=dist,
                    version=version,
                    error_type=type(fallback_exc).__name__,
                    error_message=str(fallback_exc),
                )
                logger.error(
                    f"[griffe] Fallback load failed for {target}=={version}: "
                    f"{fallback_exc}"
                )
                logger.warning(f"[griffe] Returning empty API surface for {target}=={version}")
                return {}, extraction_mode, cache_hit
        else:
            raise
    except Exception as e:
        _write_failure_marker(
            failure_path,
            package=package,
            distribution=dist,
            version=version,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        logger.error(f"[griffe] Unexpected error loading {target}=={version}: {e}")
        # Return empty surface to avoid crashing the pipeline
        logger.warning(f"[griffe] Returning empty API surface for {target}=={version}")
        return {}, extraction_mode, cache_hit
    raw = json.loads(module.as_json(full=True))

    surface: dict[str, APIEntry] = {}
    if isinstance(raw, dict) and "kind" in raw and "members" in raw:
        # Griffe may serialize the loaded module directly as an object dict.
        _walk_members(package, raw, surface)
    elif package in raw:
        _walk_members(package, raw[package], surface)
    else:
        # Fallback for unexpected root key naming.
        for root_name, root_node in raw.items():
            _walk_members(root_name, root_node, surface)

    serializable = {k: asdict(v) for k, v in surface.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    # Success supersedes any old failure marker for this (package, version, dist).
    try:
        if os.path.exists(failure_path):
            os.remove(failure_path)
    except Exception:
        pass
    logger.info(f"[griffe] API surface extracted: {package}=={version}, symbols={len(serializable)}")
    return serializable, extraction_mode, cache_hit


def extract_api_surface(
    package: str,
    version: str,
    *,
    distribution: str | None = None,
    cache_dir: str = "cache/api_surface",
) -> dict[str, dict]:
    """Return API surface keyed by qualified symbol name."""
    surface, _, _ = _extract_api_surface_internal(
        package, version, distribution=distribution, cache_dir=cache_dir
    )
    return surface


def extract_api_surface_with_meta(
    package: str,
    version: str,
    *,
    distribution: str | None = None,
    cache_dir: str = "cache/api_surface",
) -> tuple[dict[str, dict], dict[str, object]]:
    """Extract API surface and return lightweight metadata for experiment tracing."""
    cache_path = get_api_surface_cache_path(package, version, cache_dir=cache_dir)
    surface, extraction_mode, cache_hit = _extract_api_surface_internal(
        package, version, distribution=distribution, cache_dir=cache_dir
    )
    meta = {
        "package": package,
        "distribution": distribution or package,
        "version": version,
        "cache_hit": cache_hit,
        "cache_path": cache_path,
        "extraction_mode": extraction_mode,
        "symbol_count": len(surface),
    }
    return surface, meta  # type: ignore[return-value]


def extract_api_surface_batch(
    package_versions: list[tuple[str, str]],
    *,
    cache_dir: str = "cache/api_surface",
    checkpoint_path: str = "cache/api_surface_checkpoint.json",
    max_workers: int = 4,
) -> dict[str, dict[str, dict]]:
    """Extract API surfaces in parallel and persist progress for resuming."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    results: dict[str, dict[str, dict]] = {}
    done: set[str] = set()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        results = saved.get("results", {})
        done = set(saved.get("done", []))
        logger.info(f"Resumed API extractor checkpoint: {len(done)} finished.")

    pending = [(p, v) for p, v in package_versions if f"{p}=={v}" not in done]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_api_surface, p, v, cache_dir=cache_dir): (p, v) for p, v in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extract API surface"):
            p, v = futures[future]
            surface = future.result()
            results.setdefault(p, {})[v] = surface
            done.add(f"{p}=={v}")

            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"results": results, "done": sorted(done)}, f, ensure_ascii=False, indent=2)

    return results

