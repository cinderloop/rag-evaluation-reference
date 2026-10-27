from __future__ import annotations

import pytest

from rag_eval.dataset.schema import Case, Evidence
from rag_eval.mutations.compare import compare_metric
from rag_eval.mutations.knowledge_edit import (
    CaseEdit,
    KnowledgeEdit,
    apply_edit,
    edited_cases,
    text_pdf,
)

EDIT = KnowledgeEdit(
    doc_id="A.pdf",
    find="repair yard in Paducah, Kentucky",
    replace="repair yard in Cape Girardeau, Missouri",
    cases=(
        CaseEdit(
            id="q1",
            expected_answer="Cape Girardeau",
            evidence=(Evidence(doc_id="A.pdf", quote="repair yard in Cape Girardeau, Missouri"),),
        ),
    ),
)


def test_edit_tolerates_whitespace_and_needs_exactly_one_match() -> None:
    text = "It was towed to a repair yard in Paducah,\n Kentucky, where it was dry docked."
    assert "Cape Girardeau, Missouri, where" in apply_edit(text, EDIT)
    with pytest.raises(ValueError, match="occurs 0 times"):
        apply_edit("nothing here", EDIT)
    with pytest.raises(ValueError, match="occurs 2 times"):
        apply_edit("repair yard in Paducah, Kentucky; repair yard in Paducah, Kentucky", EDIT)


def test_edited_cases_replace_expected_answer_and_evidence() -> None:
    original = Case(
        id="q1",
        split="test",
        question="Where?",
        answerable=True,
        expected_answer="Paducah",
        evidence=(Evidence(doc_id="A.pdf", quote="repair yard in Paducah, Kentucky"),),
    )
    edited = edited_cases({"q1": original}, EDIT)["q1"]
    assert edited.expected_answer == "Cape Girardeau"
    assert edited.evidence[0].quote.endswith("Missouri")
    assert edited.question == "Where?"


def test_text_pdf_is_deterministic_and_well_formed() -> None:
    first = text_pdf("Line (one) with a back\\slash and \u2019quotes\u2019.\n" * 130)
    assert first == text_pdf("Line (one) with a back\\slash and \u2019quotes\u2019.\n" * 130)
    assert first.startswith(b"%PDF-1.4")
    assert first.rstrip().endswith(b"%%EOF")
    assert first.count(b"/Type /Page ") == 3  # 130 lines at 60 per page
    assert b"\\(one\\)" in first


def _runs(pattern: list[int]) -> list[float]:
    return [float(x) for x in pattern]


def test_drop_in_expected_direction_is_detected() -> None:
    # 12 questions pass in all three baseline runs; under the mutation 8 of them fail.
    base = {f"q{i}": _runs([1, 1, 1]) for i in range(12)}
    mut = {f"q{i}": _runs([0, 0, 0] if i < 8 else [1, 1, 1]) for i in range(12)}
    result = compare_metric("correctness", "down", base, mut)
    # McNemar on majority outcomes: b = 8, c = 0, p = 2 / 2^8 = 0.0078125.
    assert result.p_value == pytest.approx(0.0078125)
    assert result.difference == pytest.approx(-8 / 12)
    assert result.outcome == "detected"
    wrong_way = compare_metric("correctness", "up", base, mut)
    assert wrong_way.outcome == "not detected"


def test_small_change_is_as_expected_when_none_predicted() -> None:
    base = {f"q{i}": _runs([1, 1, 1]) for i in range(30)}
    mut = {f"q{i}": _runs([0, 0, 0] if i < 2 else [1, 1, 1]) for i in range(30)}
    result = compare_metric("correctness", "none", base, mut)
    # b = 2, c = 0: p = 2 * 1/4 = 0.5, not significant.
    assert result.p_value == pytest.approx(0.5)
    assert result.outcome == "as expected"
    assert result.mde is not None
    assert result.mde > abs(result.difference)
    assert compare_metric("correctness", "any", base, mut).outcome == "not pre-registered"
