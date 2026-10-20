"""Trace building, using real AnythingLLM 1.18.0 responses captured during the spike."""

import json
from pathlib import Path

from rag_eval.client.models import ChatResponse
from rag_eval.dataset.schema import Case
from rag_eval.runner.records import TokenUsage, TraceRecord, build_trace

EVIDENCE = Path("spike/evidence")
CASE = Case(id="c1", split="dev", question="Q?", answerable=False, dimensions={"type": "factual"})


def _trace(name: str, refusal: str = "NO_RELEVANT_CONTEXT") -> TraceRecord:
    response = ChatResponse.model_validate(
        json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    )
    return build_trace(
        run_id="r",
        case=CASE,
        repeat=0,
        mode="query",
        refusal_response=refusal,
        workspace_slug="ws",
        thread_slug="t",
        response=response,
        latency_s=1.5,
        started_at="2026-10-21T00:00:00+00:00",
    )


def test_answer_with_sources() -> None:
    trace = _trace("10-chat-sources.json")
    assert trace.answer == (
        "The Zorvath Model 7 pressure valve requires replacement of the primary diaphragm every "
        "**1,850 operating hours**, or sooner if the indicator ring turns amber."
    )
    assert trace.reasoning is not None
    assert trace.reasoning.startswith("Okay")
    assert trace.refusal_path is None
    assert len(trace.retrieved) == 2
    first = trace.retrieved[0]
    assert first.doc_id == "zorvath-valve.txt"
    assert first.text.startswith("Zorvath Model 7 Pressure Valve")
    assert "<document_metadata>" not in first.text
    assert first.score is not None
    assert first.score > 0.7
    assert trace.tokens == TokenUsage(prompt=466, completion=237, total=703)
    assert trace.model == "qwen3:8b"
    assert trace.dimensions == {"type": "factual"}


def test_deterministic_refusal_is_flagged() -> None:
    trace = _trace("07-query-mode-no-context.json")
    assert trace.answer == "NO_RELEVANT_CONTEXT"
    assert trace.refusal_path == "deterministic"
    assert trace.retrieved == []
    assert trace.tokens == TokenUsage(prompt=None, completion=None, total=None)


def test_refusal_text_from_a_different_config_is_not_flagged() -> None:
    assert _trace("07-query-mode-no-context.json", refusal="Something else").refusal_path is None


def test_llm_abstention_with_context_is_not_a_deterministic_refusal() -> None:
    trace = _trace("13-unanswerable-query.json")
    assert trace.refusal_path is None
    assert trace.answer == (
        "The weight of the Zorvath Model 7 valve is not specified in the provided context."
    )


def test_abort_keeps_the_error() -> None:
    trace = _trace("16-workspace-model-override.json")
    assert trace.response_type == "abort"
    assert trace.error is not None
    assert "not found" in trace.error
    assert trace.answer is None
