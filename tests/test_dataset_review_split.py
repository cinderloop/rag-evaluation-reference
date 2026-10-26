import json
from pathlib import Path

import pytest
from dataset_fixtures import REPORTS, candidate

from rag_eval.dataset.candidates import Candidate
from rag_eval.dataset.review import DecisionImportError, candidates_sha256, import_decisions
from rag_eval.dataset.schema import Case, Evidence
from rag_eval.dataset.split import stratified_split

GOOD = "Damage to the vessel was estimated at $1 million."


def _decisions(
    tmp_path: Path,
    candidates: list[Candidate],
    decisions: dict[str, dict[str, object]],
    sha: str | None = None,
) -> Path:
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            {
                "reviewer": "reviewer-a",
                "candidates_sha256": sha or candidates_sha256(candidates),
                "decisions": decisions,
            }
        ),
        encoding="utf-8",
    )
    return path


def _decision(
    decision: str = "approve", quote: str = GOOD, question: str = "Edited question?"
) -> dict[str, object]:
    return {
        "decision": decision,
        "question": question,
        "expected_answer": "About $1 million.",
        "evidence": [{"doc_id": "A.pdf", "quote": quote}],
        "reason": "",
    }


def test_approved_items_use_reviewer_edits_and_rejects_are_dropped(tmp_path: Path) -> None:
    items = [candidate("b001-1"), candidate("b001-2", "Different question about Alpha's captain?")]
    path = _decisions(tmp_path, items, {"b001-1": _decision(), "b001-2": _decision("reject")})
    parsed, approved = import_decisions(path, items, REPORTS)
    assert parsed.reviewer == "reviewer-a"
    assert [c.id for c in approved] == ["q-b001-1"]
    assert approved[0].question == "Edited question?"
    assert approved[0].dimensions["chunk_boundary"] == "within_chunk"


def test_edited_quote_must_still_be_verbatim(tmp_path: Path) -> None:
    items = [candidate("b001-1")]
    path = _decisions(tmp_path, items, {"b001-1": _decision(quote="Damage was huge.")})
    with pytest.raises(DecisionImportError, match="not found verbatim"):
        import_decisions(path, items, REPORTS)


def test_every_candidate_needs_a_decision(tmp_path: Path) -> None:
    items = [candidate("b001-1"), candidate("b001-2", "Another distinct question on Alpha's crew?")]
    path = _decisions(tmp_path, items, {"b001-1": _decision()})
    with pytest.raises(DecisionImportError, match="no decision"):
        import_decisions(path, items, REPORTS)


def test_decisions_must_match_the_candidates_file(tmp_path: Path) -> None:
    items = [candidate("b001-1")]
    path = _decisions(tmp_path, items, {"b001-1": _decision()}, sha="0" * 64)
    with pytest.raises(DecisionImportError, match="different candidates file"):
        import_decisions(path, items, REPORTS)


def _case(i: int, qtype: str) -> Case:
    answerable = qtype != "unanswerable"
    return Case(
        id=f"q-{i:03d}",
        split="dev",
        question=f"Q{i}?",
        answerable=answerable,
        expected_answer="A" if answerable else None,
        evidence=(Evidence(doc_id="A.pdf", quote=GOOD),) if answerable else (),
        dimensions={
            "question_type": qtype,
            **({"unanswerable_kind": "out_of_corpus"} if not answerable else {}),
        },
    )


def test_split_is_stratified_deterministic_and_complete() -> None:
    cases = [_case(i, "factual") for i in range(20)] + [
        _case(100 + i, "unanswerable") for i in range(10)
    ]
    dev, test = stratified_split(cases, 0.4, seed=3)
    assert sum(c.dimensions["question_type"] == "factual" for c in dev) == 8
    assert sum(c.dimensions["question_type"] == "unanswerable" for c in dev) == 4
    assert {c.id for c in dev} | {c.id for c in test} == {c.id for c in cases}
    assert not {c.id for c in dev} & {c.id for c in test}
    assert all(c.split == "dev" for c in dev)
    assert all(c.split == "test" for c in test)
    assert stratified_split(cases, 0.4, seed=3) == (dev, test)
