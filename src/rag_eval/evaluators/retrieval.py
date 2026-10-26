"""Deterministic retrieval metrics (no LLM): hit rate, recall@k and MRR.

Computed for answerable cases only, by matching retrieved passages to the case's
labelled evidence (rules in ``rag_eval.evaluators.matching``), at two levels:

- **Passage level.** A retrieved passage is *relevant* if it holds any part of an
  evidence quote from its own document. A quote is *recalled* within the top k when
  those k passages contain the whole quote (in one passage, or across adjacent chunks).
- **Document level.** A passage is relevant if its document holds any evidence; a
  document is recalled within the top k when any of those k passages comes from it.

Per case, at cutoff k (default: every retrieved passage):

- ``hit``: 1 if any of the top k passages is relevant, else 0.
- ``recall``: share of evidence quotes (or evidence documents) recalled in the top k.
- ``reciprocal_rank``: 1 / rank of the first relevant passage, 0 if none.

Hit rate and MRR are the means of ``hit`` and ``reciprocal_rank`` over cases.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.schema import Case
from rag_eval.evaluators.matching import covers, touches
from rag_eval.runner.records import RetrievedPassage


class LevelScores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hit: int
    recall: float
    reciprocal_rank: float


class RetrievalScores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int
    passage: LevelScores
    document: LevelScores


def _first_relevant_rank(flags: Sequence[bool]) -> int | None:
    return next((i + 1 for i, flag in enumerate(flags) if flag), None)


def _level(flags: Sequence[bool], recalled: int, total: int) -> LevelScores:
    rank = _first_relevant_rank(flags)
    return LevelScores(
        hit=int(rank is not None),
        recall=recalled / total,
        reciprocal_rank=0.0 if rank is None else 1 / rank,
    )


def retrieval_scores(
    case: Case, retrieved: Sequence[RetrievedPassage], k: int | None = None
) -> RetrievalScores | None:
    """Scores for one answerable case, or None for an unanswerable one (no evidence)."""
    if not case.answerable:
        return None
    top = list(retrieved if k is None else retrieved[:k])
    evidence = case.evidence
    passage_flags = [touches(p, evidence) for p in top]
    quotes_recalled = sum(covers(e, top) for e in evidence)

    evidence_docs = {e.doc_id for e in evidence}
    doc_flags = [p.doc_id in evidence_docs for p in top]
    docs_recalled = len(evidence_docs & {p.doc_id for p in top})

    return RetrievalScores(
        k=len(top) if k is None else k,
        passage=_level(passage_flags, quotes_recalled, len(evidence)),
        document=_level(doc_flags, docs_recalled, len(evidence_docs)),
    )
