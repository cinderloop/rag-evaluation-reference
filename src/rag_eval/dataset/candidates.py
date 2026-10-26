"""Candidate questions: machine-generated, awaiting review.

A candidate is never a dataset item by itself. It becomes one only when it
is approved (possibly after editing) in the review step.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag_eval.dataset.schema import Evidence


class GeneratorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_requested: str
    model_served: str
    prompt_sha256: str
    response_id: str
    attempt: int


class Checks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lexical_overlap: float | None = None
    too_easy: bool = False
    chunk_boundary: str | None = None  # within_chunk | cross_chunk, under baseline chunking
    duplicate_of: str | None = None
    # Where the evidence actually is (report section), which can differ from the plan.
    evidence_location: str | None = None


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    batch_id: str
    doc_ids: list[str]
    answerable: bool
    dimensions: dict[str, str]
    question: str
    expected_answer: str
    evidence: list[Evidence] = Field(default_factory=list[Evidence])
    # For false-premise questions: the passage(s) contradicting the premise (review aid).
    premise_evidence: list[Evidence] = Field(default_factory=list[Evidence])
    notes: str = ""
    generator: GeneratorInfo
    checks: Checks = Field(default_factory=Checks)


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(c.model_dump_json() + "\n" for c in candidates), encoding="utf-8")


def load_candidates(path: Path) -> list[Candidate]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Candidate.model_validate_json(line) for line in lines if line.strip()]
