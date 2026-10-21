"""Generation orchestration with a fake model: validation, retry, budget, cache use."""

import json
from pathlib import Path
from typing import Any

import pytest
from dataset_fixtures import REPORTS

from rag_eval.dataset.dimensions import load_dimensions
from rag_eval.dataset.generate import (
    BudgetExceededError,
    CallResult,
    GeneratedItem,
    Usage,
    build_request,
    request_key,
    run_generation,
    validate_item,
)
from rag_eval.dataset.plan import Batch, Plan, Slot
from rag_eval.dataset.schema import Evidence

SPEC = load_dimensions(Path("data/dimensions.yaml"))
TITLES = {"A.pdf": "Grounding of Vessel Alpha", "B.pdf": "Fire aboard Vessel Bravo"}
GOOD_QUOTE = "Damage to the vessel was estimated at $1 million."


def _slot(sid: str, qtype: str = "factual", kind: str | None = None) -> Slot:
    return Slot(slot_id=sid, question_type=qtype, unanswerable_kind=kind)


def _item(
    sid: str, quotes: list[tuple[str, str]], question: str = "How much damage?"
) -> dict[str, Any]:
    return {
        "slot_id": sid,
        "question": question,
        "expected_answer": "About $1 million.",
        "evidence": [{"doc_id": d, "quote": q} for d, q in quotes],
        "notes": "n",
    }


class FakeCaller:
    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> CallResult:
        self.requests.append(request)
        items = self.responses[len(self.requests) - 1]
        return CallResult(
            text=json.dumps({"items": items}),
            model_served="fake-model",
            response_id=f"r{len(self.requests)}",
            usage=Usage(input_tokens=1000, output_tokens=500),
            cached=False,
        )


def _plan(*batches: Batch) -> Plan:
    return Plan(seed=0, n_slots=sum(len(b.slots) for b in batches), batches=list(batches))


SINGLE = Batch(
    batch_id="b001", kind="single", doc_ids=["A.pdf"], slots=[_slot("b001-1"), _slot("b001-2")]
)


def _run(caller: FakeCaller, plan: Plan, max_cost: float = 10.0):
    return run_generation(
        plan,
        SPEC,
        REPORTS,
        TITLES,
        caller,
        max_cost_usd=max_cost,
        estimate_per_batch=lambda _b: 0.01,
    )


def test_valid_items_become_candidates_with_provenance() -> None:
    caller = FakeCaller(
        [[_item("b001-1", [("A.pdf", GOOD_QUOTE)]), _item("b001-2", [("A.pdf", GOOD_QUOTE)])]]
    )
    outcome = _run(caller, _plan(SINGLE))
    assert [c.candidate_id for c in outcome.candidates] == ["b001-1", "b001-2"]
    assert outcome.failures == {}
    first = outcome.candidates[0]
    assert first.generator.model_served == "fake-model"
    assert first.generator.prompt_sha256 == request_key(caller.requests[0])
    # 1,000 input tokens at $2/M plus 500 output tokens at $10/M.
    assert outcome.cost_usd == pytest.approx(0.007)


def test_invalid_quote_is_retried_once_with_feedback_then_accepted() -> None:
    caller = FakeCaller(
        [
            [
                _item("b001-1", [("A.pdf", GOOD_QUOTE)]),
                _item("b001-2", [("A.pdf", "Damage was estimated at one million dollars.")]),
            ],
            [_item("b001-2", [("A.pdf", GOOD_QUOTE)], question="What did it cost?")],
        ]
    )
    outcome = _run(caller, _plan(SINGLE))
    assert len(caller.requests) == 2
    retry_task = caller.requests[1]["messages"][0]["content"][1]["text"]
    assert "not found verbatim" in retry_task
    assert '"b001-1"' not in retry_task  # only the failed slot is retried
    assert {c.candidate_id: c.generator.attempt for c in outcome.candidates} == {
        "b001-1": 1,
        "b001-2": 2,
    }
    assert outcome.failures == {}


def test_slot_failing_twice_is_dropped_and_recorded() -> None:
    bad = [_item("b001-1", [("A.pdf", "invented text that is not in the report")])]
    caller = FakeCaller([[*bad, _item("b001-2", [("A.pdf", GOOD_QUOTE)])], bad])
    outcome = _run(caller, _plan(SINGLE))
    assert [c.candidate_id for c in outcome.candidates] == ["b001-2"]
    assert "b001-1" in outcome.failures


def test_budget_cap_stops_before_overspending() -> None:
    caller = FakeCaller([[]])
    with pytest.raises(BudgetExceededError):
        run_generation(
            _plan(SINGLE),
            SPEC,
            REPORTS,
            TITLES,
            caller,
            max_cost_usd=0.001,
            estimate_per_batch=lambda _b: 0.01,
        )
    assert caller.requests == []


@pytest.mark.parametrize(
    ("slot", "batch_docs", "evidence", "message"),
    [
        (_slot("s"), ["A.pdf"], [], "evidence is required"),
        (
            _slot("s"),
            ["A.pdf"],
            [("B.pdf", "A fire started in the engine room at 0300.")],
            "not one of",
        ),
        (_slot("s"), ["A.pdf"], [("A.pdf", "Damage")], "too short"),
        (
            _slot("s", "unanswerable", "missing_detail"),
            ["A.pdf"],
            [("A.pdf", GOOD_QUOTE)],
            "no evidence",
        ),
        (_slot("s", "comparison"), ["A.pdf", "B.pdf"], [("A.pdf", GOOD_QUOTE)], "both reports"),
    ],
)
def test_validation_rules(
    slot: Slot, batch_docs: list[str], evidence: list[tuple[str, str]], message: str
) -> None:
    batch = Batch(
        batch_id="b",
        kind="pair" if len(batch_docs) == 2 else "single",
        doc_ids=batch_docs,
        slots=[slot],
    )
    item = GeneratedItem(
        slot_id="s",
        question="Q?",
        expected_answer="A.",
        evidence=[Evidence(doc_id=d, quote=q) for d, q in evidence],
        notes="",
    )
    assert any(message in e for e in validate_item(slot, item, batch, REPORTS))


def test_false_premise_evidence_is_kept_as_premise_evidence() -> None:
    batch = Batch(
        batch_id="b001",
        kind="single",
        doc_ids=["A.pdf"],
        slots=[_slot("b001-1", "unanswerable", "false_premise")],
    )
    caller = FakeCaller(
        [[_item("b001-1", [("A.pdf", GOOD_QUOTE)], question="Why did Alpha catch fire?")]]
    )
    candidate = _run(caller, _plan(batch)).candidates[0]
    assert candidate.evidence == []
    assert candidate.premise_evidence[0].quote == GOOD_QUOTE
    assert not candidate.answerable


def test_request_is_stable_and_caches_report_text() -> None:
    a = build_request(SINGLE, SINGLE.slots, spec=SPEC, reports=REPORTS, titles=TITLES)
    b = build_request(SINGLE, SINGLE.slots, spec=SPEC, reports=REPORTS, titles=TITLES)
    assert request_key(a) == request_key(b)
    report_block = a["messages"][0]["content"][0]
    assert report_block["cache_control"] == {"type": "ephemeral"}
    assert TITLES["A.pdf"] in report_block["text"]
    assert a["output_config"]["format"]["type"] == "json_schema"
