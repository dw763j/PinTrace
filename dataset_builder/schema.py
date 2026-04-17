"""Dataclasses describing StackExchange records and embedded code blocks.

Usage (import-only)::

    from dataset_builder.schema import SORecord, CodeBlock
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CodeBlock:
    language: str | None
    content: str


@dataclass
class SORecord:
    record_id: str
    source: str
    question_id: str | None
    answer_id: str | None
    question_title: str | None
    question_text: str | None
    answer_text: str | None
    code_blocks: list[CodeBlock] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    accepted: bool = False
    score: int | None = None
    target_packages: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["code_blocks"] = [asdict(block) for block in self.code_blocks]
        return data

