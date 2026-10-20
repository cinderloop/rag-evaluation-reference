"""Evaluation case schema (one JSON object per line in data/*.jsonl)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    """Evidence is labelled at document/passage level, never by chunk ID, because
    chunking changes under mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    quote: str = Field(min_length=1, description="Verbatim span from the document")


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    question: str = Field(min_length=1)
    answerable: bool
    expected_answer: str | None = None
    evidence: tuple[Evidence, ...] = ()
    dimensions: dict[str, str] = Field(default_factory=dict[str, str])

    @model_validator(mode="after")
    def _answerable_has_answer_and_evidence(self) -> Self:
        if self.answerable and (not self.expected_answer or not self.evidence):
            raise ValueError(f"{self.id}: answerable cases need expected_answer and evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable cases must not have evidence")
        return self


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(Case.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case IDs: {duplicates}")
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases
