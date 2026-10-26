import json
from pathlib import Path

import pytest

from rag_eval.dataset.closed_book import (
    MarksImportError,
    baseline_report,
    export_marking_page,
    load_marks,
    looks_abstained,
    traces_sha256,
    wilson,
)
from rag_eval.dataset.schema import Case, Evidence
from rag_eval.runner.records import TraceRecord

QUOTE = "Damage to the vessel was estimated at $1 million."


def _case(cid: str, answerable: bool = True) -> Case:
    return Case(
        id=cid,
        split="test",
        question=f"{cid}?",
        answerable=answerable,
        expected_answer="About $1 million." if answerable else None,
        evidence=(Evidence(doc_id="A.pdf", quote=QUOTE),) if answerable else (),
        dimensions={"question_type": "factual" if answerable else "unanswerable"}
        | ({} if answerable else {"unanswerable_kind": "out_of_corpus"}),
    )


def _trace(cid: str, answer: str) -> TraceRecord:
    return TraceRecord(
        run_id="r",
        case_id=cid,
        repeat=0,
        split="test",
        question=f"{cid}?",
        dimensions={"question_type": "factual"},
        mode="chat",
        workspace_slug="w",
        thread_slug="t",
        answer=answer,
        reasoning=None,
        response_type="textResponse",
        refusal_path=None,
        retrieved=[],
        latency_s=1.0,
        tokens=None,
        model="qwen3:8b",
        error=None,
        started_at="2026-10-23T00:00:00+00:00",
    )


CASES = {c.id: c for c in (_case("q-1"), _case("q-2"), _case("q-3", answerable=False))}
TRACES = [
    _trace("q-1", "About $1 million."),
    _trace("q-2", "I don't know the details of that casualty."),
    _trace("q-3", "I do not know."),
]


def test_wilson_matches_known_interval() -> None:
    low, high = wilson(36, 45)
    assert (round(low, 2), round(high, 2)) == (0.66, 0.89)


def test_abstentions_are_detected() -> None:
    assert looks_abstained("I don't know the details.")
    assert looks_abstained(None)
    assert not looks_abstained("The pilot fell asleep.")


def test_page_lists_only_answerable_items_and_prefills_abstentions(tmp_path: Path) -> None:
    out = tmp_path / "page.html"
    assert export_marking_page(TRACES, CASES, out) == 2
    page = out.read_text(encoding="utf-8")
    assert '"id": "q-3"' not in page
    assert '"prefill": "abstained"' in page


def _marks(tmp_path: Path, marks: dict[str, dict[str, object]]) -> Path:
    path = tmp_path / "marks.json"
    payload = {"marker": "reviewer-a", "traces_sha256": traces_sha256(TRACES), "marks": marks}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_marks_must_cover_every_item(tmp_path: Path) -> None:
    with pytest.raises(MarksImportError, match="no mark"):
        load_marks(_marks(tmp_path, {"q-1": {"mark": "correct"}}), TRACES, CASES)


def test_report_gives_rates_with_intervals(tmp_path: Path) -> None:
    path = _marks(
        tmp_path,
        {
            "q-1": {"mark": "correct", "guessable": True},
            "q-2": {"mark": "abstained", "prefilled": True},
        },
    )
    report = baseline_report(load_marks(path, TRACES, CASES), CASES, ["r"])
    assert "- Correct: 1/2 = 50%" in report
    assert "excluding questions a guess could get right: 0/1" in report
    assert "- q-1 (guessable)" in report
