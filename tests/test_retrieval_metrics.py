from __future__ import annotations

import pytest

from rag_eval.dataset.schema import Case, Evidence
from rag_eval.evaluators.matching import covers
from rag_eval.evaluators.retrieval import retrieval_scores
from rag_eval.runner.records import RetrievedPassage

QUOTE_A = "the barge struck the lock gate at a speed of 1.4 mph"
QUOTE_B = "the crew released carbon dioxide into the engine room about 1630"


def _passage(doc: str, text: str) -> RetrievedPassage:
    return RetrievedPassage(doc_id=doc, chunk_id=None, text=text, score=None)


def _case(*evidence: Evidence) -> Case:
    return Case(
        id="q1",
        split="dev",
        question="?",
        answerable=True,
        expected_answer="a",
        evidence=evidence,
    )


A = Evidence(doc_id="A.pdf", quote=QUOTE_A)
B = Evidence(doc_id="B.pdf", quote=QUOTE_B)


def test_quote_split_across_two_chunks_counts_as_recalled() -> None:
    first = _passage("A.pdf", "Earlier text. the barge struck the lock gate")
    second = _passage("A.pdf", "at a speed of 1.4 mph and stopped.")
    assert covers(A, [first, second])
    assert not covers(A, [first])


def test_quote_from_wrong_document_does_not_count() -> None:
    assert not covers(A, [_passage("B.pdf", QUOTE_A)])


def test_scores_by_hand() -> None:
    # Ranks: 1 unrelated, 2 other passage of doc A, 3 holds quote A, 4 holds quote B.
    retrieved = [
        _passage("C.pdf", "nothing relevant here at all"),
        _passage("A.pdf", "a different part of report A"),
        _passage("A.pdf", f"Before. {QUOTE_A}. After."),
        _passage("B.pdf", QUOTE_B),
    ]
    scores = retrieval_scores(_case(A, B), retrieved)
    assert scores is not None
    assert scores.k == 4
    assert scores.passage.hit == 1
    assert scores.passage.recall == 1.0
    assert scores.passage.reciprocal_rank == pytest.approx(1 / 3)
    assert scores.document.recall == 1.0
    assert scores.document.reciprocal_rank == pytest.approx(1 / 2)

    top2 = retrieval_scores(_case(A, B), retrieved, k=2)
    assert top2 is not None
    assert (top2.passage.hit, top2.passage.recall, top2.passage.reciprocal_rank) == (0, 0.0, 0.0)
    assert top2.document.recall == 0.5


def test_unanswerable_case_has_no_retrieval_scores() -> None:
    case = Case(id="q2", split="dev", question="?", answerable=False)
    assert retrieval_scores(case, [_passage("A.pdf", QUOTE_A)]) is None
