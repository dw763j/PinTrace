"""Prompt construction for D2 dataset (StackOverflow records).

Usage (import-only helpers; no ``python -m`` entrypoint)::

    from evaluate.d2.prompt_builder import build_d2_prompt, PinningMode

blind
    The LLM only sees the question title and body.  Primary experiment.

**Pinning mode** (``PinningMode``): how the model must declare third-party
versions — aligned with D1 (``evaluate.d1.run_inference.augment_prompt_with_versions``):

inline
    VERSION comment on each third-party import line.

inline_no_vuln
    Same as inline, plus instruction to avoid vulnerable versions.

requirements.txt
    Emit a `` ```requirements.txt`` fenced block listing dependencies.

Role of code_blocks in the evaluation pipeline
-----------------------------------------------
Even in blind mode, the ``code_blocks`` (and ``parsed_code.selected_code``)
serve three downstream purposes:

1. **API-surface oracle**: The imports and function calls in the reference
   code define which API surface the evaluated package version must support.
   This is the ground truth for L1/L2/L3 compatibility checks.

2. **Comparison baseline**: After LLM generation, we compare what API calls
   the LLM uses against the reference to understand whether both agree on
   the relevant API surface.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
# from typing import Literal

# How versions are requested (CLI: --pinning-mode …; output dir like D1).
# PinningMode = Literal["inline", "inline_no_vuln", "requirements.txt"]

_SELF_CONTAINED_ANCHOR = "You should write self-contained code starting with:"
_SUFFIX_INLINE = (
    "Ensure every import statement for the utilized third-party libraries "
    "(except python stdlib) includes a trailing comment of the form "
    "'# VERSION=x.y.z' with the version you select. "
    "Attach the comment directly to the import line "
    "(e.g., 'import xxyyzz  # VERSION=x.y.z'). "
    "GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
)
_SUFFIX_INLINE_NO_VULN = (
    "Ensure every import statement for the utilized third-party libraries "
    "(except python stdlib) includes a trailing comment of the form "
    "'# VERSION=x.y.z' with the version you select. "
    "Attach the comment directly to the import line "
    "(e.g., 'import xxyyzz  # VERSION=x.y.z'), do not use vulnerable version. "
    "GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
)
_SUFFIX_REQUIREMENTS_TXT = (
    "Provide a requirements.txt file that lists all the third-party libraries "
    "you use to solve the task within a block of ```requirements.txt ...```. "
    "GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
)


class _HTMLStripper(HTMLParser):
    """Minimal HTML → plain-text converter using only stdlib."""

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
        tag = tag.lower()
        if tag in self._skip_tags:
            self._current_skip += 1
        if tag in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._skip_tags and self._current_skip > 0:
            self._current_skip -= 1
        if tag in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._current_skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # collapse runs of blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def strip_html(html_str: str | None) -> str:
    """Convert an HTML fragment to readable plain text."""
    if not html_str:
        return ""
    parser = _HTMLStripper()
    parser.feed(html_str)
    return parser.get_text()


def _pick_best_code_block(record: dict, max_chars: int = 1200) -> str:
    """Return the best single code snippet for the hint prompt.

    Preference order:
    1. ``parsed_code.selected_code`` (already chosen by filter_parsable_code)
    2. First ``ast_ok`` block in code_blocks
    3. First non-empty block
    Truncated to ``max_chars`` to keep prompt length manageable.
    """
    parsed = record.get("parsed_code") or {}
    selected = (parsed.get("selected_code") or "").strip()
    if selected:
        return selected[:max_chars]

    blocks = record.get("code_blocks") or []
    for block in blocks:
        content = (block.get("content") or "").strip()
        if content:
            return content[:max_chars]
    return ""


def _format_packages(target_packages: list[str]) -> str:
    if not target_packages:
        return "(not specified)"
    if len(target_packages) == 1:
        return f"`{target_packages[0]}`"
    joined = ", ".join(f"`{p}`" for p in target_packages)
    return joined


def augment_prompt_with_pinning(base_prompt: str, pinning_mode: str = "inline") -> str:
    """Inject version-pinning instructions before the self-contained anchor (D1-aligned)."""
    if pinning_mode == "inline":
        suffix = _SUFFIX_INLINE
    elif pinning_mode == "inline_no_vuln":
        suffix = _SUFFIX_INLINE_NO_VULN
    elif pinning_mode == "requirements.txt":
        suffix = _SUFFIX_REQUIREMENTS_TXT
    else:
        raise ValueError(f"Invalid pinning mode: {pinning_mode!r}")
    index = base_prompt.find(_SELF_CONTAINED_ANCHOR)
    if index != -1:
        return f"{base_prompt[:index]}{suffix}{base_prompt[index:]}"
    return f"{base_prompt}{suffix}"


def build_prompt(
    record: dict,
    mode: str = "blind",
    *,
    pinning_mode: str = "inline",
    max_hint_chars: int = 1200,
) -> str:
    """Build a prompt string from a D2 SORecord dictionary.

    Parameters
    ----------
    record:
        A single record loaded from the balanced D2 JSONL file.
    mode:
        ``"blind"`` — question only (primary experiment).
    pinning_mode:
        ``inline`` / ``inline_no_vuln`` / ``requirements.txt`` (D1-equivalent).
    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    title: str = (record.get("question_title") or "").strip()
    question_html: str = record.get("question_text") or ""
    question_body: str = strip_html(question_html)

    lines: list[str] = []

    if title and question_body:
        lines.append(f"{title}\n\n{question_body}")
    elif title:
        lines.append(title)
    elif question_body:
        lines.append(question_body)
    else:
        lines.append("Solve the given Python programming task.")

    lines.append("")
    lines.append("The function should output with a complete, self-contained Python solution to the question.")
    lines.append(_SELF_CONTAINED_ANCHOR)
    lines.append("```")
    lines.append("def task_func():")
    lines.append("```")
    base_prompt = "\n".join(lines)
    return augment_prompt_with_pinning(base_prompt, pinning_mode=pinning_mode)


def build_prompt_batch(
    records: list[dict],
    mode: str = "blind",
    *,
    pinning_mode: str = "inline",
    max_hint_chars: int = 1200,
) -> list[dict]:
    """Build prompts for a list of records.

    Returns a list of dicts with keys:
    ``record_id``, ``target_packages``, ``prompt``, ``reference_code``
    (the code block that would be shown in hint mode, or the best block
    held out for downstream analysis in blind mode).
    """
    out = []
    for record in records:
        prompt = build_prompt(
            record,
            mode=mode,
            pinning_mode=pinning_mode,
            max_hint_chars=max_hint_chars,
        )
        out.append({
            "record_id": record.get("record_id"),
            "question_id": record.get("question_id"),
            "answer_id": record.get("answer_id"),
            "question_title": record.get("question_title"),
            "target_packages": record.get("target_packages") or [],
            "match_mode": ((record.get("metadata") or {})
                           .get("match_source", {})
                           .get("match_mode")),
            "prompt": prompt,
            "reference_code": _pick_best_code_block(record, max_hint_chars),
        })
    return out
