"""Ablation experiment prompt builders for D1/D2 datasets.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m stages.ablation_prompts --help
    python -m stages.ablation_prompts --ablation 2 --tpls pandas numpy  # print a single-demo tier breakdown
    python -m stages.ablation_prompts --summarize-ablation2-jsonl path/to/inline_safe_version.jsonl
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import threading
from html.parser import HTMLParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import argparse
from loguru import logger
from packaging.version import InvalidVersion, Version
from scripts.griffe_api_extractor import extract_api_surface
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Reuse name-normalisation from osv_version_matrix ─────────────────────────
from stages.osv_version_matrix import _candidate_pkg_names

# ── Reuse version-fetch utilities from version_resolver ──────────────────────
from stages.version_resolver import (
    _build_release_time_index,
    _load_pypi_json_from_cache,
)

# ── Shared prompt constants – kept in sync with evaluate/d2/prompt_builder.py ─
_SELF_CONTAINED_ANCHOR = "You should write self-contained code starting with:"

_SUFFIX_INLINE = (
    "Ensure every import statement for the utilized third-party libraries "
    "(except python stdlib) includes a trailing comment of the form "
    "'# VERSION=x.y.z' with the version you select. "
    "Attach the comment directly to the import line "
    "(e.g., 'import xxyyzz  # VERSION=x.y.z'). "
    "GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
)
_SUFFIX_REQUIREMENTS_TXT = (
    "Provide a requirements.txt file that lists all the third-party libraries "
    "you use to solve the task within a block of ```requirements.txt ...```. "
    "GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
)

# Matches the PinningMode literal in evaluate/d2/prompt_builder.py (inline / req.txt).
# inline_no_vuln is handled by Ablation 1 as a separate ablation axis.
PinningMode = Literal["inline", "requirements.txt"]

VersionReason = Literal["safe", "min_vuln", "latest", "not_found"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared injection helper
# ─────────────────────────────────────────────────────────────────────────────

def _augment_before_anchor(base_prompt: str, content: str) -> str:
    """Insert *content* immediately before the self-contained anchor.

    Mirrors the injection pattern of
    ``evaluate.d2.prompt_builder.augment_prompt_with_pinning``:
    falls back to appending when the anchor is not found.
    """
    idx = base_prompt.find(_SELF_CONTAINED_ANCHOR)
    if idx != -1:
        return f"{base_prompt[:idx]}{content}{base_prompt[idx:]}"
    return f"{base_prompt}{content}"


def _pinning_suffix(pinning_mode: PinningMode) -> str:
    return _SUFFIX_REQUIREMENTS_TXT if pinning_mode == "requirements.txt" else _SUFFIX_INLINE


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 1
# ─────────────────────────────────────────────────────────────────────────────

def build_ablation1_prompt(
    base_prompt: str,
    pinning_mode: PinningMode = "inline",
) -> str:
    """Ablation 1: add a standalone vulnerability-avoidance instruction."""
    content = "Do not use vulnerable version.\n\n" + _pinning_suffix(pinning_mode)
    return _augment_before_anchor(base_prompt, content)

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 2: safe-version selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_sorted_versions(
    pypi_name: str,
    pypi_info_dir: str,
    reference_date: str | None = None,
) -> list[str]:
    """Sorted (ascending PEP 440) released versions from local PyPI cache.

    Uses ``_candidate_pkg_names`` (osv_version_matrix) for name variants,
    and ``_load_pypi_json_from_cache`` + ``_build_release_time_index``
    (version_resolver) for yanked-aware version listing.
    """
    for candidate in _candidate_pkg_names(pypi_name):
        data = _load_pypi_json_from_cache(candidate, pypi_info_dir)
        if data is None:
            continue
        release_times = _build_release_time_index(data, exclude_yanked=True)
        if reference_date:
            cutoff = datetime.fromisoformat(reference_date.replace("Z", "+00:00"))
            release_times = {v: t for v, t in release_times.items() if t <= cutoff}
        parsed: list[tuple[Version, str]] = []
        for v in release_times:
            try:
                parsed.append((Version(v), v))
            except InvalidVersion:
                pass
        parsed.sort()
        return [s for _, s in parsed]
    return []


def _pick_latest(versions: list[str]) -> str | None:
    """Latest PEP 440 version from a plain list (no datetime lookup needed)."""
    parsed: list[tuple[Version, str]] = []
    for v in versions:
        try:
            parsed.append((Version(v), v))
        except InvalidVersion:
            pass
    return max(parsed, key=lambda x: x[0])[1] if parsed else None


def load_osv_version_matrix(matrix_path: str | Path | None = None) -> dict[str, Any]:
    """Load the ``"packages"`` section of the pre-built OSV version matrix.

    Defaults to ``global_cache/osv_version_matrix.json``.
    Build it first with: ``python -m stages.osv_version_matrix``
    """
    if matrix_path is None:
        from paths import GLOBAL_CACHE
        matrix_path = GLOBAL_CACHE / "osv_version_matrix.json"
    path = Path(matrix_path)
    if not path.exists():
        logger.warning(
            f"OSV version matrix not found at {path}. "
            "Run: python -m stages.osv_version_matrix"
        )
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("packages", {})


def find_safe_version(
    pypi_name: str,
    osv_matrix: dict[str, Any],
    all_versions: list[str],
) -> tuple[str | None, VersionReason]:
    """Select the safest available version for a package.

    Priority:

    1. **safe** – latest version absent from the OSV vulnerability matrix.
    2. **min_vuln** – latest version with the smallest ``vuln_count``.
    3. **latest** – last entry in ``all_versions``.
    4. **not_found** – when ``all_versions`` is empty.
    """
    if not all_versions:
        return None, "not_found"

    # Look up vulnerability data, trying name variants
    pkg_vuln_versions: dict[str, dict] = {}
    for candidate in _candidate_pkg_names(pypi_name):
        entry = osv_matrix.get(candidate)
        if entry:
            pkg_vuln_versions = entry.get("versions") or {}
            break

    # 1. Safe: not listed in OSV
    safe = [v for v in all_versions if v not in pkg_vuln_versions]
    if safe:
        best = _pick_latest(safe)
        if best:
            return best, "safe"

    # 2. Min-vuln fallback
    def _vuln_count(v: str) -> int:
        return pkg_vuln_versions.get(v, {}).get("vuln_count", 0)

    min_count = min(_vuln_count(v) for v in all_versions)
    min_vuln = [v for v in all_versions if _vuln_count(v) == min_count]
    best = _pick_latest(min_vuln)
    if best:
        return best, "min_vuln"

    # 3. Latest known
    best = _pick_latest(all_versions)
    return (best, "latest") if best else (None, "not_found")


def resolve_tpl_versions(
    tpls: list[str],
    osv_matrix: dict[str, Any],
    pypi_info_dir: str,
    reference_date: str | None = None,
) -> dict[str, tuple[str | None, VersionReason]]:
    """Resolve a safe version for each TPL import/package name.

    Falls back to the import→PyPI-name mapping from ``stages.utils.get_pkg_name``
    when the local PyPI cache has no entry for the raw name.
    """
    results: dict[str, tuple[str | None, VersionReason]] = {}
    get_pkg_name = None
    pkg_mapping: dict[str, str] | None = None
    try:
        from stages.utils import get_pkg_name as _get_pkg_name, load_mapping
        pkg_mapping, _ = load_mapping()
        get_pkg_name = _get_pkg_name
    except Exception:
        get_pkg_name = None
        pkg_mapping = None

    for tpl in tpls:
        raw_name = tpl.strip().lower()
        mapped_name = raw_name
        if get_pkg_name is not None:
            try:
                mapped_name = str(get_pkg_name(tpl, mapping=pkg_mapping)).strip().lower()
            except Exception:
                mapped_name = raw_name

        candidates: list[str] = []
        for name in (mapped_name, raw_name):
            if name and name not in candidates:
                candidates.append(name)

        pypi_name = raw_name
        versions: list[str] = []
        for candidate in candidates:
            got = _get_sorted_versions(candidate, pypi_info_dir, reference_date)
            if got:
                versions = got
                pypi_name = candidate
                break
        results[tpl] = find_safe_version(pypi_name, osv_matrix, versions)
    return results


def _accumulate_ablation2_reasons_from_version_map(
    version_map: Any,
    counter: Counter[str],
) -> None:
    """Accumulate ``VersionReason`` counts from one record's ``version_map`` (tuple or list pairs)."""
    if not isinstance(version_map, dict):
        return
    for _tpl, pair in version_map.items():
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            reason = str(pair[1])
        else:
            continue
        counter[reason] += 1


def aggregate_ablation2_reasons_from_jsonl(path: str | Path) -> Counter[str]:
    """Scan Ablation-2 JSONL rows (each with ``version_map``) and tally resolution tiers."""
    path = Path(path)
    out: Counter[str] = Counter()
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            _accumulate_ablation2_reasons_from_version_map(rec.get("version_map"), out)
    return out


def summarize_ablation2_version_tier_distribution(counter: Counter[str]) -> dict[str, Any]:
    """Summarize Ablation-2 ``VersionReason`` counts, fractions, and percentages.

    Each listed TPL in a task counts once (same granularity as ``resolve_tpl_versions``).
    """
    order = ("safe", "min_vuln", "latest", "not_found")
    total = sum(int(counter.get(k, 0)) for k in order)
    counts = {k: int(counter.get(k, 0)) for k in order}
    if total <= 0:
        return {
            "total_tpl_slots": 0,
            "counts": counts,
            "fraction": {k: 0.0 for k in order},
            "percent": {k: 0.0 for k in order},
        }
    fr = {k: round(counts[k] / total, 6) for k in order}
    pct = {k: 100.0 * counts[k] / total for k in order}
    return {
        "total_tpl_slots": total,
        "counts": counts,
        "fraction": fr,
        "percent": {k: round(pct[k], 4) for k in order},
    }


def _format_version_constraints_block(
    version_map: dict[str, tuple[str | None, VersionReason]],
) -> str:
    """Human-readable block listing the safe version for each TPL."""
    _reason_label = {
        "safe": "no known vulnerabilities",
        "min_vuln": "fewest known vulnerabilities",
        "latest": "latest known",
        "not_found": "version unknown",
    }
    lines = [
        "For the following third-party libraries, use the specified safe versions:",
        "",
    ]
    for tpl, (ver, reason) in sorted(version_map.items()):
        label = _reason_label.get(reason, reason)
        if ver:
            lines.append(f"- {tpl}: use version {ver}  [{label}]")
        else:
            lines.append(f"- {tpl}: no specific version recommended  [{label}]")
    lines.append("")
    return "\n".join(lines)


def build_ablation2_prompt(
    base_prompt: str,
    tpls: list[str],
    osv_matrix: dict[str, Any],
    pypi_info_dir: str,
    reference_date: str | None = None,
    pinning_mode: PinningMode = "inline",
) -> tuple[str, dict[str, tuple[str | None, VersionReason]]]:
    """Ablation 2: inject safe version constraints for all TPLs.

    Resolves a safe version for each TPL via the OSV matrix, then injects the
    version table + pinning instruction before the anchor – matching the structure of
    ``evaluate/d2/prompt_builder.augment_prompt_with_pinning``.

    Returns
    -------
    tuple[str, dict]
        ``(augmented_prompt, {tpl: (version, reason)})``
    """
    version_map = resolve_tpl_versions(tpls, osv_matrix, pypi_info_dir, reference_date)
    content = _format_version_constraints_block(version_map) + "\n" + _pinning_suffix(pinning_mode)
    return _augment_before_anchor(base_prompt, content), version_map


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 3: griffe API extraction + keyword BM25 RAG
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens from arbitrary text."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _build_bm25_index(
    documents: list[list[str]],
) -> tuple[dict[str, float], list[Counter[str]], list[int], float]:
    """Build BM25 index state: idf, per-doc term freq, doc lengths, avg length."""
    n_docs = len(documents)
    term_freqs: list[Counter[str]] = [Counter(doc) for doc in documents]
    doc_lengths = [len(doc) for doc in documents]
    avg_doc_len = (sum(doc_lengths) / n_docs) if n_docs else 0.0

    df: dict[str, int] = {}
    for doc in documents:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    # Okapi BM25 IDF (RSJ) with +1 in log to keep non-negative values.
    idf = {t: math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    return idf, term_freqs, doc_lengths, avg_doc_len


def _bm25_score(
    query_tokens: list[str],
    doc_tf: Counter[str],
    doc_len: int,
    avg_doc_len: float,
    idf: dict[str, float],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute Okapi BM25 score for one query/document pair."""
    if not query_tokens or not doc_tf or avg_doc_len <= 0:
        return 0.0

    query_tf = Counter(query_tokens)
    score = 0.0
    norm = k1 * (1.0 - b + b * (doc_len / avg_doc_len))
    for term, qf in query_tf.items():
        tf = doc_tf.get(term, 0)
        if tf <= 0:
            continue
        term_idf = idf.get(term)
        if term_idf is None:
            continue
        score += qf * term_idf * ((tf * (k1 + 1.0)) / (tf + norm))
    return score


def _is_public_api_qname(qname: str) -> bool:
    """Heuristic filter to drop obviously internal/test-qualified names before prompting."""
    lowered = qname.lower()
    blocked_parts = (
        ".tests.",
        ".testing.",
        "._testing.",
        ".bench",
        ".benchmark",
        ".conftest",
        "._",
        ".core.internals.",
        "._libs.",
        ".compat.",
    )
    return not any(part in lowered for part in blocked_parts)


def _compact_qname_for_prompt(qname: str, pkg_name: str) -> str:
    """Convert verbose qualified name to a code-facing display name."""
    if qname.startswith(f"{pkg_name}."):
        tail = qname[len(pkg_name) + 1 :]
    else:
        tail = qname
    parts = [p for p in tail.split(".") if p]
    if not parts:
        return qname

    # Prefer class-oriented notation (e.g., DataFrame.aggregate).
    class_idx = [i for i, p in enumerate(parts) if p[:1].isupper()]
    if class_idx:
        i = class_idx[-1]
        return ".".join(parts[i:])

    # Fallback to concise module/function suffix.
    return ".".join(parts[-2:]) if len(parts) > 1 else parts[0]


def _simplify_type_repr(value: Any) -> str | None:
    """Make griffe expression dict/string readable for prompt context."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s[:1] in {"{", "["}:
            try:
                parsed = ast.literal_eval(s)
                parsed_s = _simplify_type_repr(parsed)
                if parsed_s:
                    return parsed_s
            except Exception:
                pass
        # Keep prompt concise when raw annotation is too verbose.
        return (s[:56] + "...") if len(s) > 60 else s
    if isinstance(value, dict):
        op = value.get("operator")
        if op in {"|", "&"} and "left" in value and "right" in value:
            left = _simplify_type_repr(value.get("left")) or "Any"
            right = _simplify_type_repr(value.get("right")) or "Any"
            return f"{left} {op} {right}"
        if "name" in value and isinstance(value["name"], str):
            return value["name"]
        if "member" in value and isinstance(value["member"], str):
            return value["member"]
        if "elements" in value and isinstance(value["elements"], list):
            elems = [e for e in (_simplify_type_repr(v) for v in value["elements"]) if e]
            return f"[{', '.join(elems)}]" if elems else None
        if "slice" in value:
            return _simplify_type_repr(value.get("slice"))
        return None
    if isinstance(value, (list, tuple)):
        elems = [e for e in (_simplify_type_repr(v) for v in value) if e]
        return f"[{', '.join(elems)}]" if elems else None
    return str(value)


def _summarize_docstring(text: Any, max_tokens: int = 18) -> str | None:
    """Take first sentence/line of docstring as short retrieval evidence."""
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return None
    sentence = re.split(r"(?<=[.!?])\s+|\n", cleaned, maxsplit=1)[0].strip()
    words = sentence.split()
    if len(words) > max_tokens:
        sentence = " ".join(words[:max_tokens]) + " ..."
    return sentence


def rag_retrieve_api_info(
    question: str,
    api_surfaces: dict[str, dict[str, dict]],
    top_k: int = 20,
) -> list[dict]:
    """Retrieve the most relevant API entries for *question* via BM25 RAG.

    Parameters
    ----------
    question:
        Plain-text programming question used as the retrieval query.
    api_surfaces:
        ``{package_name: {qualified_name: api_entry_dict}}`` as produced by
        the griffe-based extractor.  Each entry has keys ``kind``,
        ``parameters``, ``return_type``, ``docstring``.
    top_k:
        Maximum number of API entries to return (sorted by relevance).
    """
    if top_k <= 0:
        return []

    entries: list[dict] = []
    documents: list[list[str]] = []

    for pkg_name, surface in api_surfaces.items():
        for qname, info in surface.items():
            if not isinstance(info, dict):
                continue
            tokens = _tokenize(qname)
            for param in info.get("parameters") or []:
                if isinstance(param, dict):
                    tokens += _tokenize(param.get("name") or "")
                    ann = param.get("annotation")
                    if ann:
                        tokens += _tokenize(str(ann))
            ret = info.get("return_type")
            if ret:
                tokens += _tokenize(str(ret))
            docstring = info.get("docstring")
            if docstring:
                tokens += _tokenize(str(docstring))
            entries.append({
                "package": pkg_name,
                "name": qname,
                "display_name": _compact_qname_for_prompt(qname, pkg_name),
                "kind": info.get("kind", "unknown"),
                "parameters": info.get("parameters") or [],
                "return_type": ret,
                "docstring": docstring,
                "public_api": _is_public_api_qname(qname),
                "score": 0.0,
            })
            documents.append(tokens)

    if not documents:
        return []

    idf, term_freqs, doc_lengths, avg_doc_len = _build_bm25_index(documents)
    query_tokens = _tokenize(question)

    for i, entry in enumerate(entries):
        base_score = _bm25_score(
            query_tokens,
            term_freqs[i],
            doc_lengths[i],
            avg_doc_len,
            idf,
        )
        # Prefer public, code-facing symbols while retaining fallback coverage.
        entry["score"] = base_score if entry.get("public_api", True) else base_score * 0.3

    positive = [e for e in entries if e["score"] > 0.0]
    public_positive = [e for e in positive if e.get("public_api", True)]
    internal_positive = [e for e in positive if not e.get("public_api", True)]
    ranked = sorted(public_positive, key=lambda e: e["score"], reverse=True)
    ranked.extend(sorted(internal_positive, key=lambda e: e["score"], reverse=True))

    # Fallback when no positive matches are found.
    if not ranked:
        ranked = sorted(entries, key=lambda e: e["score"], reverse=True)
    return ranked[:top_k]


def _format_api_entry(entry: dict) -> str:
    """Concise function/class/attribute signature string."""
    name = entry.get("display_name") or entry["name"]
    ret = _simplify_type_repr(entry.get("return_type"))
    if entry.get("kind") == "attribute":
        line = f"{name}: {ret}" if ret else name
        doc = _summarize_docstring(entry.get("docstring"))
        return f"{line}  -- {doc}" if doc else line

    param_strs: list[str] = []
    for p in entry.get("parameters") or []:
        if not isinstance(p, dict):
            continue
        pname = p.get("name", "")
        ann = _simplify_type_repr(p.get("annotation"))
        pstr = f"{pname}: {ann}" if ann else pname
        if p.get("has_default"):
            pstr += "=..."
        param_strs.append(pstr)

    sig = f"{name}({', '.join(param_strs)})"
    line = f"{sig} -> {ret}" if ret else sig
    doc = _summarize_docstring(entry.get("docstring"))
    return f"{line}  -- {doc}" if doc else line


def _format_api_block(
    entries: list[dict],
    version_map: dict[str, tuple[str | None, VersionReason]],
) -> str:
    """Format retrieved API entries grouped by package."""
    lines = ["Relevant API reference for the selected package versions:", ""]
    by_pkg: dict[str, list[dict]] = {}
    for entry in entries:
        by_pkg.setdefault(entry["package"], []).append(entry)

    # Build a quick lookup: normalised pkg_name → version string
    ver_lookup: dict[str, str] = {}
    for tpl, (ver, _) in version_map.items():
        if ver:
            for c in _candidate_pkg_names(tpl):
                ver_lookup[c] = ver

    for pkg_name in sorted(by_pkg):
        ver_str = ver_lookup.get(pkg_name, "")
        header = f"# {pkg_name}" + (f" v{ver_str}" if ver_str else "")
        lines.append(header)
        for entry in by_pkg[pkg_name]:
            lines.append(f"  {_format_api_entry(entry)}")
        lines.append("")
    return "\n".join(lines)


def extract_api_surface_for_versions(
    version_map: dict[str, tuple[str | None, VersionReason]],
    cache_dir: str,
) -> dict[str, dict[str, dict]]:
    """Extract griffe API surfaces for all resolved (package, version) pairs.

    which provides griffe-based extraction with file-system caching.

    Returns ``{package_name: {qualified_symbol: api_entry_dict}}``.
    """

    surfaces: dict[str, dict[str, dict]] = {}
    pkg_mapping: dict[str, str] | None = None
    get_pkg_name = None
    try:
        from stages.utils import get_pkg_name as _get_pkg_name, load_mapping
        pkg_mapping, _ = load_mapping()
        get_pkg_name = _get_pkg_name
    except Exception:
        pkg_mapping = None
        get_pkg_name = None

    for tpl, (version, _) in version_map.items():
        if version is None:
            continue

        raw_name = tpl.strip().lower()
        mapped_name = raw_name
        if get_pkg_name is not None:
            try:
                mapped_name = str(get_pkg_name(tpl, mapping=pkg_mapping)).strip().lower()
            except Exception:
                mapped_name = raw_name

        # Candidate pairs are (import_name, distribution_name).
        candidates: list[tuple[str, str]] = []
        preferred = (raw_name, mapped_name)
        fallback = (raw_name, raw_name)
        for pair in (preferred, fallback):
            if pair[0] and pair[1] and pair not in candidates:
                candidates.append(pair)

        loaded = False
        for import_name, distribution_name in candidates:
            try:
                surface = extract_api_surface(
                    import_name, version, distribution=distribution_name, cache_dir=cache_dir
                )
                if surface:
                    surfaces[import_name] = surface
                    loaded = True
                    break
            except Exception as exc:
                logger.warning(
                    f"API extraction failed for {import_name} "
                    f"(dist={distribution_name})=={version}: {exc}"
                )

        if not loaded:
            logger.warning(
                f"API extraction yielded empty surface for tpl={tpl}, "
                f"version={version}, candidates={candidates}"
            )
    return surfaces


def build_ablation3_prompt(
    base_prompt: str,
    question: str,
    tpls: list[str],
    osv_matrix: dict[str, Any],
    pypi_info_dir: str,
    api_surface_cache_dir: str | None = None,
    reference_date: str | None = None,
    top_k_apis: int = 20,
    pinning_mode: PinningMode = "inline",
) -> tuple[str, dict[str, Any]]:
    """Ablation 3: safe versions + griffe API extraction + keyword RAG.

    Builds on Ablation 2 by additionally:

    1. Extracting griffe API surfaces for the resolved (package, version) pairs.
    2. Retrieving the ``top_k_apis`` most relevant entries via BM25 similarity
       against ``question``.
    3. Prepending the retrieved API snippets to the version-constraint block,
       then injecting the combined section before the anchor.

    Returns
    -------
    tuple[str, dict]
        ``(augmented_prompt, metadata)`` – metadata contains ``version_map``,
        ``api_counts`` (symbols per package), ``retrieved_count``.
    """
    if api_surface_cache_dir is None:
        from paths import GLOBAL_CACHE
        api_surface_cache_dir = str(GLOBAL_CACHE / "api_surface")

    version_map = resolve_tpl_versions(tpls, osv_matrix, pypi_info_dir, reference_date)
    api_surfaces = extract_api_surface_for_versions(version_map, api_surface_cache_dir)
    relevant = rag_retrieve_api_info(question, api_surfaces, top_k=top_k_apis)

    parts: list[str] = []
    if relevant:
        parts.append(_format_api_block(relevant, version_map))
        parts.append("\n")
    parts.append(_format_version_constraints_block(version_map))
    parts.append("\n")
    parts.append(_pinning_suffix(pinning_mode))

    augmented = _augment_before_anchor(base_prompt, "".join(parts))
    return augmented, {
        "version_map": {t: (v, r) for t, (v, r) in version_map.items()},
        "api_counts": {pkg: len(surf) for pkg, surf in api_surfaces.items()},
        "retrieved_count": len(relevant),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch build for D1 + D2
# ─────────────────────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {"script", "style"}
        self._block_tags = {
            "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
            "blockquote", "pre", "tr", "td", "th",
        }
        self._current_skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t in self._skip_tags:
            self._current_skip += 1
        if t in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._skip_tags and self._current_skip > 0:
            self._current_skip -= 1
        if t in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._current_skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def _strip_html(html_str: str | None) -> str:
    if not html_str:
        return ""
    parser = _HTMLStripper()
    parser.feed(html_str)
    return parser.get_text()


def _strip_known_pinning_suffix(base_prompt: str) -> str:
    idx = base_prompt.find(_SELF_CONTAINED_ANCHOR)
    if idx == -1:
        return base_prompt
    prefix = base_prompt[:idx]
    for suffix in (_SUFFIX_INLINE, _SUFFIX_REQUIREMENTS_TXT):
        if prefix.endswith(suffix):
            return f"{prefix[:-len(suffix)]}{base_prompt[idx:]}"
    return base_prompt


def _iter_d1_records(dataset: str, split: str, max_examples: int | None) -> list[dict]:
    from evaluate.inference import load_dataset_split

    ds = load_dataset_split(dataset, split)
    items: list[dict] = []
    for i, item in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        items.append(item)
    return items


def _load_d2_records(path: str | Path, max_examples: int | None) -> list[dict]:
    p = Path(path)
    records: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_d2_base_prompt(record: dict, context_mode: Literal["blind", "hint"]) -> str:
    from evaluate.d2.prompt_builder import build_prompt

    # Build with inline pinning then strip suffix to get a true "base prompt".
    prompt = build_prompt(record, mode=context_mode, pinning_mode="inline")
    return _strip_known_pinning_suffix(prompt)


def _build_d2_question(record: dict) -> str:
    title = (record.get("question_title") or "").strip()
    body = _strip_html(record.get("question_text") or "")
    if title and body:
        return f"{title}\n\n{body}"
    return title or body or "Solve the given Python programming task."


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_written_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            idx = rec.get("sample_index")
            if isinstance(idx, int):
                done.add(idx)
    return done


def _load_batch_checkpoint(path: Path) -> tuple[set[int], set[int]]:
    if not path.exists():
        return set(), set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        d1 = {int(x) for x in payload.get("d1_done", [])}
        d2 = {int(x) for x in payload.get("d2_done", [])}
        return d1, d2
    except Exception:
        return set(), set()


def _save_batch_checkpoint(path: Path, d1_done: set[int], d2_done: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "d1_done": sorted(d1_done),
        "d2_done": sorted(d2_done),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_all_ablation_prompts_d1_d2(
    *,
    d1_dataset: str,
    d1_split: str,
    d2_dataset_jsonl: str | Path,
    output_dir: str | Path,
    pypi_info_dir: str,
    pinning_mode: PinningMode = "inline",
    d2_context_mode: Literal["blind", "hint"] = "blind",
    max_examples: int | None = None,
    reference_date: str | None = None,
    top_k_apis: int = 20,
    api_surface_cache_dir: str | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Batch-build Ablation 1/2/3 prompts for both D1 and D2 datasets.

    Results are persisted incrementally to JSONL files and a checkpoint file,
    so interrupted runs can resume without losing completed work.
    """
    osv_matrix = load_osv_version_matrix()
    out_root = Path(output_dir)

    d1_records = _iter_d1_records(d1_dataset, d1_split, max_examples)
    d2_records = _load_d2_records(d2_dataset_jsonl, max_examples)

    def _build_d1_one(item: dict) -> tuple[dict, dict, dict]:
        base_prompt = item.get("instruct_prompt") or ""
        try:
            from evaluate.inference import parse_libs
            tpls = [lib.strip() for lib in parse_libs(item.get("libs")) if lib.strip()]
        except Exception:
            raw_libs = item.get("libs")
            if isinstance(raw_libs, (list, tuple, set)):
                tpls = [str(lib).strip() for lib in raw_libs if str(lib).strip()]
            else:
                tpls = []
        task_id = item.get("task_id")

        prompt1 = build_ablation1_prompt(base_prompt, pinning_mode=pinning_mode)
        prompt2, vmap2 = build_ablation2_prompt(
            base_prompt, tpls, osv_matrix, pypi_info_dir, reference_date, pinning_mode
        )
        prompt3, meta3 = build_ablation3_prompt(
            base_prompt=base_prompt,
            question=base_prompt,
            tpls=tpls,
            osv_matrix=osv_matrix,
            pypi_info_dir=pypi_info_dir,
            api_surface_cache_dir=api_surface_cache_dir,
            reference_date=reference_date,
            top_k_apis=top_k_apis,
            pinning_mode=pinning_mode,
        )
        row1 = {"dataset": "d1", "task_id": task_id, "tpls": tpls, "prompt": prompt1}
        row2 = {"dataset": "d1", "task_id": task_id, "tpls": tpls, "prompt": prompt2, "version_map": vmap2}
        row3 = {"dataset": "d1", "task_id": task_id, "tpls": tpls, "prompt": prompt3, "meta": meta3}
        return row1, row2, row3

    def _build_d2_one(record: dict) -> tuple[dict, dict, dict]:
        base_prompt = _build_d2_base_prompt(record, d2_context_mode)
        question = _build_d2_question(record)
        tpls = [str(x).strip() for x in (record.get("target_packages") or []) if str(x).strip()]
        record_id = record.get("record_id")
        question_id = record.get("question_id")

        prompt1 = build_ablation1_prompt(base_prompt, pinning_mode=pinning_mode)
        prompt2, vmap2 = build_ablation2_prompt(
            base_prompt, tpls, osv_matrix, pypi_info_dir, reference_date, pinning_mode
        )
        prompt3, meta3 = build_ablation3_prompt(
            base_prompt=base_prompt,
            question=question,
            tpls=tpls,
            osv_matrix=osv_matrix,
            pypi_info_dir=pypi_info_dir,
            api_surface_cache_dir=api_surface_cache_dir,
            reference_date=reference_date,
            top_k_apis=top_k_apis,
            pinning_mode=pinning_mode,
        )
        common = {
            "dataset": "d2",
            "record_id": record_id,
            "question_id": question_id,
            "tpls": tpls,
            "context_mode": d2_context_mode,
        }
        row1 = {**common, "prompt": prompt1}
        row2 = {**common, "prompt": prompt2, "version_map": vmap2}
        row3 = {**common, "prompt": prompt3, "meta": meta3}
        return row1, row2, row3

    d1_out = out_root / "d1"
    d2_out = out_root / "d2"
    d1_paths = {
        1: d1_out / "inline_no_vuln.jsonl",
        2: d1_out / "inline_safe_version.jsonl",
        3: d1_out / "inline_api_rag.jsonl",
    }
    d2_paths = {
        1: d2_out / "inline_no_vuln.jsonl",
        2: d2_out / "inline_safe_version.jsonl",
        3: d2_out / "inline_api_rag.jsonl",
    }
    checkpoint_path = out_root / "batch_checkpoint.json"

    d1_done_ckpt, d2_done_ckpt = _load_batch_checkpoint(checkpoint_path)
    d1_done_files = _load_written_indices(d1_paths[1])
    d2_done_files = _load_written_indices(d2_paths[1])
    d1_done = d1_done_ckpt | d1_done_files
    d2_done = d2_done_ckpt | d2_done_files

    d1_todo = [(i, item) for i, item in enumerate(d1_records) if i not in d1_done]
    d2_todo = [(i, rec) for i, rec in enumerate(d2_records) if i not in d2_done]

    write_lock = threading.Lock()
    progress = {"d1": 0, "d2": 0}
    total_todo = {"d1": len(d1_todo), "d2": len(d2_todo)}

    d1_out.mkdir(parents=True, exist_ok=True)
    d2_out.mkdir(parents=True, exist_ok=True)
    d1_fhs = {k: d1_paths[k].open("a", encoding="utf-8") for k in (1, 2, 3)}
    d2_fhs = {k: d2_paths[k].open("a", encoding="utf-8") for k in (1, 2, 3)}

    def _write_triplet(dataset: Literal["d1", "d2"], idx: int, rows: tuple[dict, dict, dict]) -> None:
        rows_idx = tuple({**row, "sample_index": idx} for row in rows)
        fhs = d1_fhs if dataset == "d1" else d2_fhs
        done_set = d1_done if dataset == "d1" else d2_done
        with write_lock:
            for ablation, row in zip((1, 2, 3), rows_idx):
                fhs[ablation].write(json.dumps(row, ensure_ascii=False) + "\n")
                fhs[ablation].flush()
            done_set.add(idx)
            _save_batch_checkpoint(checkpoint_path, d1_done, d2_done)
            progress[dataset] += 1

    max_workers = max(1, workers)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, item in d1_todo:
                futures[executor.submit(_build_d1_one, item)] = ("d1", idx)
            for idx, rec in d2_todo:
                futures[executor.submit(_build_d2_one, rec)] = ("d2", idx)

            for fut in as_completed(list(futures.keys())):
                dataset, idx = futures[fut]
                rows = fut.result()
                _write_triplet(dataset, idx, rows)
    finally:
        for fh in list(d1_fhs.values()) + list(d2_fhs.values()):
            try:
                fh.close()
            except Exception:
                pass

    d1_ab2_counter = aggregate_ablation2_reasons_from_jsonl(d1_paths[2])
    d2_ab2_counter = aggregate_ablation2_reasons_from_jsonl(d2_paths[2])
    combined_ab2 = d1_ab2_counter + d2_ab2_counter

    return {
        "output_dir": str(out_root),
        "pinning_mode": pinning_mode,
        "d2_context_mode": d2_context_mode,
        "workers": max_workers,
        "checkpoint": str(checkpoint_path),
        "counts": {
            "d1": len(d1_records),
            "d2": len(d2_records),
        },
        "todo": total_todo,
        "written_this_run": progress,
        "completed_total": {
            "d1": len(d1_done),
            "d2": len(d2_done),
        },
        "files": {
            "d1": [str(d1_paths[1]), str(d1_paths[2]), str(d1_paths[3])],
            "d2": [str(d2_paths[1]), str(d2_paths[2]), str(d2_paths[3])],
        },
        "ablation2_version_tier_distribution": {
            "d1": summarize_ablation2_version_tier_distribution(d1_ab2_counter),
            "d2": summarize_ablation2_version_tier_distribution(d2_ab2_counter),
            "combined": summarize_ablation2_version_tier_distribution(combined_ab2),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo(ablation: int, tpls: list[str]) -> None:
    from paths import PYPI_INFO, ensure_dirs
    ensure_dirs()

    sample = (
        "Given a DataFrame, compute the rolling mean of a numeric column.\n\n"
        f"{_SELF_CONTAINED_ANCHOR}\n"
        "```\n"
        "def task_func():\n"
        "```"
    )

    if ablation == 1:
        print("=== Ablation 1 ===")
        print(build_ablation1_prompt(sample))

    elif ablation == 2:
        result, version_map = build_ablation2_prompt(
            sample, tpls=tpls,
            osv_matrix=load_osv_version_matrix(),
            pypi_info_dir=str(PYPI_INFO),
        )
        print("=== Ablation 2 – version map ===")
        for tpl, (ver, reason) in version_map.items():
            print(f"  {tpl}: {ver} ({reason})")
        demo_ctr: Counter[str] = Counter()
        _accumulate_ablation2_reasons_from_version_map(version_map, demo_ctr)
        print()
        print("=== Ablation 2 – tier distribution (this demo) ===")
        print(json.dumps(summarize_ablation2_version_tier_distribution(demo_ctr), ensure_ascii=False, indent=2))
        print()
        print(result)

    elif ablation == 3:
        result, meta = build_ablation3_prompt(
            sample, question=sample, tpls=tpls,
            osv_matrix=load_osv_version_matrix(),
            pypi_info_dir=str(PYPI_INFO),
        )
        print("=== Ablation 3 – metadata ===")
        print(json.dumps(meta, indent=2, default=str))
        print()
        print(result)


def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(description="Ablation prompt builder demo.")
    parser.add_argument("--ablation", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--tpls", nargs="+", default=["pandas", "numpy"])
    parser.add_argument(
        "--summarize-ablation2-jsonl",
        action="append",
        metavar="PATH",
        help="Repeatable. Scan Ablation-2 JSONL (with version_map) and print tier JSON, then exit.",
    )
    parser.add_argument("--run-batch", action="store_true", help="Batch build all 3 ablations for D1 and D2.")
    parser.add_argument("--d1-dataset", default="bigcode/bigcodebench", help="D1 dataset id/path (for --run-batch).")
    parser.add_argument("--d1-split", default="test", help="D1 split name (for --run-batch).")
    parser.add_argument("--d2-dataset", default=None, help="Path to D2 dataset JSONL (for --run-batch).")
    parser.add_argument("--output-dir", default=None, help="Output directory for batched prompts.")
    parser.add_argument("--pinning-mode", default="inline", choices=["inline", "requirements.txt"])
    parser.add_argument("--d2-context-mode", default="blind", choices=["blind", "hint"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--top-k-apis", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.summarize_ablation2_jsonl:
        merged: Counter[str] = Counter()
        for p in args.summarize_ablation2_jsonl:
            merged.update(aggregate_ablation2_reasons_from_jsonl(p))
        print(json.dumps(summarize_ablation2_version_tier_distribution(merged), ensure_ascii=False, indent=2))
        sys.exit(0)
    if args.run_batch:
        from paths import OUTPUTS, PYPI_INFO, ensure_dirs

        ensure_dirs()
        if not args.d2_dataset:
            raise ValueError("--run-batch requires --d2-dataset")
        out_dir = Path(args.output_dir) if args.output_dir else (OUTPUTS / "ablation_prompts")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = build_all_ablation_prompts_d1_d2(
            d1_dataset=args.d1_dataset,
            d1_split=args.d1_split,
            d2_dataset_jsonl=args.d2_dataset,
            output_dir=out_dir,
            pypi_info_dir=str(PYPI_INFO),
            pinning_mode=args.pinning_mode,
            d2_context_mode=args.d2_context_mode,
            max_examples=args.max_examples,
            reference_date=args.reference_date,
            top_k_apis=args.top_k_apis,
            workers=args.workers,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _demo(args.ablation, args.tpls)
