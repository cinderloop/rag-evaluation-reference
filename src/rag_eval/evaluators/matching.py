"""Matching labelled evidence quotes to retrieved passages.

Evidence is labelled as verbatim quotes per document, never by chunk ID, because
chunking changes under mutation. Both texts are normalized (whitespace and quote
characters) and compared as overlapping 6-word shingles, so a quote split across two
adjacent chunks is still recognised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rag_eval.dataset.corpus_text import normalize
from rag_eval.dataset.schema import Evidence
from rag_eval.runner.records import RetrievedPassage

SHINGLE_WORDS = 6
EvidenceStatus = Literal["retrieved", "partly", "not_retrieved"]


def shingles(text: str) -> set[str]:
    words = normalize(text).split(" ")
    if len(words) <= SHINGLE_WORDS:
        return {" ".join(words)}
    return {" ".join(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


def evidence_status(evidence: Evidence, retrieved: Sequence[RetrievedPassage]) -> EvidenceStatus:
    """Whether a quote is in one retrieved passage of its document, partly, or not at all."""
    passages = [normalize(p.text) for p in retrieved if p.doc_id == evidence.doc_id]
    quote = normalize(evidence.quote)
    if any(quote in p for p in passages):
        return "retrieved"
    joined = " ".join(passages)
    if passages and any(s in joined for s in shingles(quote)):
        return "partly"
    return "not_retrieved"


def covers(evidence: Evidence, passages: Sequence[RetrievedPassage]) -> bool:
    """True if the passages of the quote's document contain the whole quote: in one
    passage, or (for a quote cut by a chunk boundary) with every word of the quote
    inside some 6-word shingle that occurs in one of the passages. A piece shorter than
    six words on either side of a cut cannot be matched, so such a quote is not counted."""
    texts = [normalize(p.text) for p in passages if p.doc_id == evidence.doc_id]
    quote = normalize(evidence.quote)
    if any(quote in t for t in texts):
        return True
    words = quote.split(" ")
    if not texts or len(words) <= SHINGLE_WORDS:
        return False
    covered = [False] * len(words)
    for i in range(len(words) - SHINGLE_WORDS + 1):
        if any(" ".join(words[i : i + SHINGLE_WORDS]) in t for t in texts):
            covered[i : i + SHINGLE_WORDS] = [True] * SHINGLE_WORDS
    return all(covered)


def touches(passage: RetrievedPassage, evidence: Sequence[Evidence]) -> bool:
    """True if the passage holds any part (one shingle or more) of an evidence quote
    from its own document."""
    text = normalize(passage.text)
    return any(
        e.doc_id == passage.doc_id and any(s in text for s in shingles(e.quote)) for e in evidence
    )
