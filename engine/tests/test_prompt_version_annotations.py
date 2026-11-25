from __future__ import annotations

import json

import pytest

from evaluate.annotate_prompt_versions import (
    audit_judgment,
    build_llm_prompt,
    detect_mentions,
    finalize_llm,
    ground_evidence,
    mention_is_contaminating,
    parse_llm_judgment,
)


def test_detects_python_and_target_package_versions():
    row = {
        "question_title": "requests==2.31.0 on Python 3.12",
        "question_body": "Why does this fail?",
        "target_packages": ["requests"],
    }
    kinds = {item["kind"] for item in detect_mentions(row)}
    assert kinds == {"python_version", "pep508_specifier"}


def test_ignores_unrelated_package_specifier_when_targets_known():
    row = {
        "question_title": "Use urllib3==2.0 with requests",
        "target_packages": ["requests"],
    }
    assert detect_mentions(row) == []


def test_llm_prompt_contains_only_frozen_classification_inputs():
    prompt = build_llm_prompt(
        {
            "record_id": "task-1",
            "question_title": "requests 2.31 on Python 3.12",
            "question_text": "Example",
            "target_packages": ["requests"],
            "answer_text": "must not be classified",
        }
    )
    assert "pintrace-version-mention-v1" in prompt
    assert "requests 2.31" in prompt
    assert "must not be classified" not in prompt


def test_parse_llm_judgment_requires_evidence_for_contamination():
    result = parse_llm_judgment(
        '```json\n{"label":"contaminated","mentions":[{"subject":"python","name":"Python","version_text":"3.12","quote":"Python 3.12"}],"reason":"runtime"}\n```'
    )
    assert result["label"] == "contaminated"
    with pytest.raises(ValueError, match="must contain evidence"):
        parse_llm_judgment('{"label":"contaminated","mentions":[],"reason":""}')


def test_finalize_llm_requires_complete_one_label_per_task(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    labels = tmp_path / "labels.jsonl"
    clean = tmp_path / "clean.jsonl"
    contaminated = tmp_path / "contaminated.jsonl"
    dataset.write_text(
        '\n'.join(json.dumps({"record_id": task_id}) for task_id in ("a", "b")) + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        '\n'.join(
            json.dumps({"task_id": task_id, "label": label})
            for task_id, label in (("a", "clean"), ("b", "contaminated"))
        )
        + "\n",
        encoding="utf-8",
    )
    finalize_llm(dataset, labels, clean, contaminated)
    assert json.loads(clean.read_text(encoding="utf-8"))["task_id"] == "a"
    assert json.loads(contaminated.read_text(encoding="utf-8"))["task_id"] == "b"


def test_ground_evidence_accepts_html_normalized_version_mention():
    source = "My <strong>Python</strong> version is <code>3.12.6</code>."
    mention = {
        "subject": "python",
        "name": "Python",
        "version_text": "3.12.6",
        "quote": "Python version is 3.12.6",
    }
    assert ground_evidence(mention, source) == "normalized_quote"
    mention["version_text"] = "9.99"
    mention["quote"] = "Python 9.99"
    assert ground_evidence(mention, source) is None


def test_bare_dependency_and_python3_only_are_not_contaminating():
    assert mention_is_contaminating(
        {"subject": "dependency", "name": "feedparser", "version_text": "unknown", "quote": "feedparser"}
    ) is False
    assert mention_is_contaminating(
        {"subject": "python", "name": "python", "version_text": "3", "quote": "python3"}
    ) is False
    assert mention_is_contaminating(
        {"subject": "python", "name": "python", "version_text": "3", "quote": "Python 3"}
    ) is False
    assert mention_is_contaminating(
        {"subject": "python", "name": "python", "version_text": "3.12", "quote": "Python 3.12"}
    ) is True
    assert mention_is_contaminating(
        {"subject": "python", "name": "python", "version_text": "37", "quote": "--runtime python37"}
    ) is True
    assert mention_is_contaminating(
        {
            "subject": "dependency",
            "name": "docutils",
            "version_text": "Python38",
            "quote": r"Python\Python38\site-packages\docutils",
        }
    ) is True
    audited = audit_judgment(
        {
            "label": "contaminated",
            "mentions": [{"subject": "python", "name": "python", "version_text": "3", "quote": "python3"}],
            "reason": "runtime",
        }
    )
    assert audited["label"] == "clean"
    assert audited["mentions"] == []
