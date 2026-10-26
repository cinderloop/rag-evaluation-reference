import json
from pathlib import Path

import pytest

from rag_eval.dataset.schema import Case, Evidence
from rag_eval.review.traces import (
    OpenCodingImportError,
    evidence_status,
    export_coding_page,
    load_open_coding,
    trace_key,
    traces_sha256,
)
from rag_eval.runner.records import RetrievedPassage, TraceRecord

QUOTE = "Damage to the vessel was estimated at $1 million after the grounding."
CASE = Case(
    id="q-1",
    split="dev",
    question="How much damage?",
    answerable=True,
    expected_answer="About $1 million.",
    evidence=(Evidence(doc_id="A.pdf", quote=QUOTE),),
    dimensions={"question_type": "factual"},
)


def _passage(text: str, doc_id: str = "A.pdf") -> RetrievedPassage:
    return RetrievedPassage(doc_id=doc_id, chunk_id="c", text=text, score=0.5)


def _trace(run_id: str, retrieved: list[RetrievedPassage]) -> TraceRecord:
    return TraceRecord(
        run_id=run_id,
        case_id="q-1",
        repeat=0,
        split="dev",
        question=CASE.question,
        dimensions=CASE.dimensions,
        mode="query",
        workspace_slug="w",
        thread_slug="t",
        answer="About $1 million.",
        reasoning=None,
        response_type="textResponse",
        refusal_path=None,
        retrieved=retrieved,
        latency_s=1.0,
        tokens=None,
        model="m",
        error=None,
        started_at="2026-10-23T00:00:00+00:00",
    )


def test_evidence_status_distinguishes_full_partial_and_missing() -> None:
    evidence = CASE.evidence[0]
    full = [
        _passage(
            "Earlier text. Damage to the vessel was\nestimated at $1 million after the grounding."
        )
    ]
    assert evidence_status(evidence, full) == "retrieved"
    part = [_passage("Damage to the vessel was estimated at $1 million")]
    assert evidence_status(evidence, part) == "partly"
    other_doc = [_passage(QUOTE, doc_id="B.pdf")]
    assert evidence_status(evidence, other_doc) == "not_retrieved"


def test_trace_keys_hide_the_run_and_differ_per_run() -> None:
    a, b = _trace("run-baseline", []), _trace("run-weak", [])
    assert trace_key(a) != trace_key(b)
    assert "baseline" not in trace_key(a)


def test_page_hides_run_ids_and_highlights_evidence(tmp_path: Path) -> None:
    traces = [_trace("20261023T000000Z-ea-weak-retrieval", [_passage(f"Before. {QUOTE}")])]
    out = tmp_path / "traces.html"
    assert export_coding_page(traces, {"q-1": CASE}, out, seed=1) == 1
    page = out.read_text(encoding="utf-8")
    assert "weak-retrieval" not in page
    assert '"status": "retrieved"' in page
    assert '"spans": [[8,' in page


def _notes(tmp_path: Path, traces: list[TraceRecord], notes: dict[str, object]) -> Path:
    path = tmp_path / "notes.json"
    payload = {"coder": "reviewer-a", "traces_sha256": traces_sha256(traces), "notes": notes}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_every_trace_needs_a_note_or_an_ok(tmp_path: Path) -> None:
    traces = [_trace("r1", []), _trace("r2", [])]
    k1, k2 = (trace_key(t) for t in traces)
    with pytest.raises(OpenCodingImportError, match="no note"):
        load_open_coding(_notes(tmp_path, traces, {k1: {"ok": True}}), traces)
    with pytest.raises(OpenCodingImportError, match="neither marked OK nor noted"):
        load_open_coding(_notes(tmp_path, traces, {k1: {"ok": True}, k2: {"note": " "}}), traces)
    coding = load_open_coding(
        _notes(tmp_path, traces, {k1: {"ok": True}, k2: {"note": "wrong report retrieved"}}),
        traces,
    )
    assert coding.notes[k2].note == "wrong report retrieved"
