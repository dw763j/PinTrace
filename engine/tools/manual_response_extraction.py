"""Extract saved model responses with source locations and explicit ambiguity.

The returned Python text is copied from the response. Named companion files are
kept as separate files. Conflicting implementations remain review candidates.
"""

from __future__ import annotations

import ast
import hashlib
import re
import warnings
from pathlib import PurePosixPath
from textwrap import dedent

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


_FENCE = re.compile(r"(?m)^[ \t]*```([^\n]*)\n(.*?)(?:^[ \t]*```[ \t]*$)", re.DOTALL)
_FILE_NAME = re.compile(r"(?<![\w/])([\w.-]+(?:[/\\][\w.-]+)*\.(?:py|txt))(?!\w)", re.IGNORECASE)
_REQUIREMENTS_HEADER = re.compile(r"^\s*[#*`\s]*(?:file\s*[:=]\s*)?requirements\.txt\s*[:`*]*\s*$", re.IGNORECASE)
_EMBEDDED_MANIFEST = re.compile(
    r"(?m)^[ \t]*#[ \t]*(?:file\s*[:=]\s*)?requirements\.txt[ \t]*:?[ \t]*\n"
    r"[ \t]*(?P<quote>'''|\"\"\")(?P<body>[\s\S]*?)(?P=quote)",
    re.IGNORECASE,
)


def _filename(text: str) -> str | None:
    match = _FILE_NAME.search(text)
    if not match:
        return None
    prefix = text[:match.start()].strip(" #*`()")
    if prefix and not re.fullmatch(r"(?i)(?:(?:file|filename|save as|save to|python|python3|py)\s*[:=]?\s*|\d+[.)]\s*)", prefix):
        return None
    candidate = match.group(1).replace("\\", "/")
    path = PurePosixPath(candidate)
    if path.is_absolute() or ".." in path.parts:
        return None
    return str(path)


def _code_info(code: str) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return {"parses": False, "has_task_func": bool(re.search(r"(?m)^\s*(?:async\s+)?def task_func\s*\(", code)), "placeholder": False, "definitions": []}
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    functions = [node for node in definitions if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "task_func"]
    placeholder = False
    if functions:
        body = [node for node in functions[0].body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
        placeholder = bool(body) and all(
            isinstance(node, ast.Pass)
            or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis
            for node in body
        )
        substantive_others = [node for node in tree.body if node not in functions and not isinstance(node, (ast.Import, ast.ImportFrom))]
        placeholder = placeholder and not substantive_others
    return {"parses": True, "has_task_func": bool(functions), "placeholder": placeholder, "definitions": [node.name for node in definitions]}


def _normalized_code(code: str) -> str:
    # Leading indentation and comments carry meaning. Only trailing whitespace
    # and blank outer lines are ignored for duplicate blocks.
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _requirement_list(text: str) -> bool:
    found = False
    for line in text.splitlines():
        value = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if not value or value.startswith(("#", "-")):
            continue
        try:
            Requirement(value)
        except InvalidRequirement:
            return False
        found = True
    return found


def _fenced_blocks(content: str):
    whole_code = _code_info(content.strip())
    if whole_code["parses"] and whole_code["has_task_func"]:
        # Markdown inside a valid, unfenced Python wrapper is string content.
        yield "python", content, 0, 0, len(content), False
        return
    consumed = 0
    while match := _FENCE.search(content, consumed):
        body = match.group(2)
        end = match.end()
        recovered_nested_fence = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(body)
        except SyntaxError as error:
            for nested in re.finditer(r"(?m)^[ \t]*```(?:requirements(?:\.txt)?|python|text|txt)[ \t]*\n", body):
                prefix = body[:nested.start()]
                if prefix.strip() and _code_info(prefix)["parses"]:
                    # A newly opened labelled block supplies an unambiguous
                    # boundary when the preceding Python block lacks its close.
                    body = prefix
                    end = match.start(2) + nested.start()
                    recovered_nested_fence = True
                    break
            if "unterminated triple-quoted" in error.msg:
                for closing in re.finditer(r"(?m)^[ \t]*```[ \t]*$", content[match.end():]):
                    candidate_end = match.end() + closing.start()
                    candidate = content[match.start(2):candidate_end]
                    if _code_info(candidate)["parses"]:
                        body = candidate
                        end = match.end() + closing.end()
                        recovered_nested_fence = True
                        break
        consumed = end
        yield match.group(1), body, match.start(), match.start(2), end, recovered_nested_fence


def _literal_text(node: ast.AST) -> str | None:
    """Read string constants and literal dedent/strip forms without execution."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"strip", "lstrip", "rstrip"} and not node.args:
        value = _literal_text(node.func.value)
        if value is not None:
            return getattr(value, node.func.attr)()
    is_dedent = isinstance(node.func, ast.Name) and node.func.id == "dedent" or isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "textwrap" and node.func.attr == "dedent"
    if is_dedent and len(node.args) == 1:
        value = _literal_text(node.args[0])
        return dedent(value) if value is not None else None
    return None


def _manifest_conflicts(sources: list[dict]) -> list[str]:
    declarations: dict[str, set[str]] = {}
    for source in sources:
        for line in source["text"].splitlines():
            text = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            if not text or text.startswith(("#", "-")):
                continue
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                continue
            requirement.name = canonicalize_name(requirement.name)
            declarations.setdefault(canonicalize_name(requirement.name), set()).add(str(requirement))
    return sorted(name for name, values in declarations.items() if len(values) > 1)


def extract_response(raw_response: str, *, legacy_code: str | None = None) -> dict:
    """Return code, files, manifests, provenance, and review decisions.

    ``review_required`` means the source contains competing implementations or
    conflicting file/manifest definitions. ``code`` is then only a provisional
    source candidate. Callers must keep unresolved responses out of automatic
    correction application until a source-selection decision is recorded.
    """
    content_start = raw_response.rfind("</think>") + len("</think>") if "</think>" in raw_response else 0
    content = raw_response[content_start:]
    manifests = []
    blocks = []
    previous_end = 0
    pending_manifest_heading = False
    for number, (info, body, block_start, body_start, block_end, nested_fence) in enumerate(_fenced_blocks(content), 1):
        info = info.strip()
        body = body.strip()
        preceding_text = content[previous_end:block_start].strip()
        preceding = preceding_text.splitlines()
        heading = preceding[-1] if preceding else ""
        first_line = body.splitlines()[0] if body else ""
        filename = _filename(info) or _filename(first_line if first_line.lstrip().startswith("#") else "") or _filename(heading)
        language = info.lower().split()[0] if info else ""
        manifest_language = language in {"", "text", "txt", "plaintext", "requirements", "requirements.txt"}
        has_manifest_heading = ("requirements.txt" in heading.lower() or pending_manifest_heading and not preceding) and _requirement_list(body)
        is_manifest = language in {"requirements", "requirements.txt"} or bool(manifest_language and (has_manifest_heading or filename and PurePosixPath(filename).name.lower() == "requirements.txt"))
        body_lines = body.splitlines()
        if body_lines and _REQUIREMENTS_HEADER.fullmatch(body_lines[0]):
            manifest_lines = body_lines[1:]
            if manifest_lines and re.fullmatch(r"\s*[=-]{3,}\s*", manifest_lines[0]):
                manifest_lines = manifest_lines[1:]
            manifest_body = "\n".join(manifest_lines).strip()
            has_python_statements = bool(re.search(r"(?m)^\s*(?:async\s+def|def|class|from|import)\s+", manifest_body))
            is_manifest = not has_python_statements
            if has_python_statements:
                comment_lines = []
                for line in manifest_lines:
                    value = line.strip()
                    if not value:
                        continue
                    if not value.startswith("#"):
                        break
                    value = value.lstrip("# ")
                    if not _requirement_list(value):
                        break
                    comment_lines.append(value)
                if comment_lines:
                    manifests.append({"block": number, "source_line": raw_response[:content_start + body_start].count("\n") + 2, "format": "labelled_comments", "text": "\n".join(comment_lines)})
                filename = None
        else:
            manifest_body = body
        line_number = raw_response[:content_start + body_start].count("\n") + 1
        block = {
            "block": number,
            "language": language,
            "filename": filename,
            "source_line": line_number,
            "text": body,
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "manifest": is_manifest,
            "nested_fence_recovered": nested_fence,
            "preceding_text": preceding_text,
        }
        if is_manifest:
            manifests.append({"block": number, "source_line": line_number, "format": "fenced", "text": manifest_body})
        else:
            code_info = _code_info(body)
            python_language = language in {"", "text", "txt", "python", "python3", "py"} or language.endswith(".py")
            explicit_python = language in {"python", "python3", "py"} or bool(python_language and filename and filename.endswith(".py"))
            block["python"] = explicit_python or not language and code_info["parses"]
            block.update(code_info)
        blocks.append(block)
        pending_manifest_heading = is_manifest and not manifest_body
        previous_end = block_end

    for match in _EMBEDDED_MANIFEST.finditer(content):
        manifest_body = match.group("body").strip()
        fenced_manifest = re.fullmatch(r"```(?:requirements\.txt|text|txt)?\s*\n([\s\S]*?)\n```", manifest_body)
        if fenced_manifest:
            manifest_body = fenced_manifest.group(1).strip()
        manifests.append({
            "block": None,
            "source_line": raw_response[:content_start + match.start("body")].count("\n") + 1,
            "format": "labelled_string",
            "text": manifest_body,
        })

    for block in blocks:
        if not block.get("python") or not block.get("parses"):
            continue
        if "requirements" not in block["text"]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(block["text"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # A literal printed document may contain its own manifest fence.
                for match in re.finditer(r"```requirements\.txt\s*\n([\s\S]*?)\n[ \t]*```", node.value):
                    manifests.append({"block": block["block"], "source_line": block["source_line"] + node.lineno - 1, "format": "literal_document_fence", "text": match.group(1).strip()})
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not set(names) & {"requirements", "requirements_txt", "requirements_content", "requirements_txt_content", "requirements_block"}:
                continue
            value = _literal_text(node.value)
            if value is not None:
                manifests.append({"block": block["block"], "source_line": block["source_line"] + node.lineno - 1, "format": "named_literal_string", "text": value.strip()})

    unique_manifests = []
    seen_manifests = set()
    for source in manifests:
        fenced_manifest = re.fullmatch(r"```(?:requirements\.txt|text|txt)?\s*\n([\s\S]*?)\n```", source["text"])
        if fenced_manifest:
            source = {**source, "text": fenced_manifest.group(1).strip()}
        lines = source["text"].splitlines()
        if lines and _REQUIREMENTS_HEADER.fullmatch(lines[0]):
            lines = lines[1:]
        if lines and re.fullmatch(r"\s*[=-]{3,}\s*", lines[0]):
            lines = lines[1:]
        source = {**source, "text": "\n".join(lines).strip()}
        normalized = _normalized_code(source["text"])
        if normalized and normalized not in seen_manifests:
            unique_manifests.append(source)
            seen_manifests.add(normalized)
    manifests = unique_manifests

    candidates = [block for block in blocks if block.get("python")]
    if not candidates:
        fallback = legacy_code if legacy_code is not None else content.strip()
        candidates = [{"block": 0, "filename": None, "text": fallback, "source_line": 1, **_code_info(fallback)}]
    unique_candidates = []
    seen_code = set()
    for block in candidates:
        normalized = _normalized_code(block["text"])
        if normalized not in seen_code:
            unique_candidates.append(block)
            seen_code.add(normalized)
    candidates = unique_candidates

    substantive = [block for block in candidates if not block["placeholder"]]
    task_candidates = [block for block in substantive if block["has_task_func"]]
    named = [block for block in substantive if block.get("filename") and block["filename"].endswith(".py")]
    review_reasons = []
    if len(task_candidates) == 1:
        selected = task_candidates[0]
        selection_reason = "unique_substantive_task_func"
    elif len(task_candidates) > 1:
        selected = next((block for block in task_candidates if _normalized_code(block["text"]) == _normalized_code(legacy_code or "")), task_candidates[0])
        selection_reason = "competing_task_func_implementations"
        review_reasons.append("Multiple distinct substantive task_func implementations require a source-selection decision.")
    elif named:
        preferred = [block for block in named if PurePosixPath(block["filename"]).name in {"main.py", "app.py", "solution.py"}]
        selected = preferred[0] if preferred else named[0]
        selection_reason = "named_python_files"
    elif len(substantive) == 1:
        selected = substantive[0]
        selection_reason = "single_substantive_python_block"
    else:
        pool = substantive or candidates
        selected = next((block for block in pool if _normalized_code(block["text"]) == _normalized_code(legacy_code or "")), pool[0])
        selection_reason = "multiple_python_blocks" if substantive else "placeholder_only"
        if len(substantive) > 1:
            review_reasons.append("Multiple Python blocks without one task_func require a primary-solution decision.")

    primary_filename = selected.get("filename") or "solution.py"
    files = {primary_filename: selected["text"]}
    for block in named:
        filename = block["filename"]
        if filename in files and _normalized_code(files[filename]) != _normalized_code(block["text"]):
            review_reasons.append(f"Multiple distinct definitions of {filename} require a file-selection decision.")
        else:
            files[filename] = block["text"]
    conflicts = _manifest_conflicts(manifests) if len(manifests) > 1 else []
    if conflicts:
        review_reasons.append("Multiple manifest blocks give different constraints for " + ", ".join(conflicts) + ".")
    local_modules = sorted({PurePosixPath(filename).parts[0].removesuffix(".py") for filename in files if filename != "solution.py"})
    selected_code = selected["text"]
    if legacy_code is not None and _normalized_code(selected_code) == _normalized_code(legacy_code):
        selected_code = legacy_code
        files[primary_filename] = legacy_code
    return {
        "code": selected_code,
        "primary_filename": primary_filename,
        "files": files,
        "manifest_text": "\n".join(source["text"] for source in manifests),
        "manifest_sources": manifests,
        "local_modules": local_modules,
        "selection_reason": selection_reason,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "placeholder_blocks": [block["block"] for block in candidates if block["placeholder"]],
        "selected_block": selected["block"],
        "blocks": blocks,
    }
