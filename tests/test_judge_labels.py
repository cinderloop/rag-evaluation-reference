from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.dataset.schema import Case, Evidence
from rag_eval.evaluators.judges.judges import Verdict
from rag_eval.review.judge_labels import (
    JudgeLabelsImportError,
    LabelledCase,
    agreement,
    export_label_page,
    judge_split,
    labelable_judges,
    labelled_cases,
    load_judge_labels,
)
from rag_eval.review.traces import trace_key, traces_sha256
from rag_eval.runner.records import RetrievedPassage, TraceRecord

ANSWERABLE = Case(
    id="q1",
    split="dev",
    question="How fast?",
    answerable=True,
    expected_answer="1.4 mph",
    evidence=(Evidence(doc_id="A.pdf", quote="at 1.4 mph"),),
)
UNANSWERABLE = Case(id="q2", split="dev", question="Who?", answerable=False)
CASES = {"q1": ANSWERABLE, "q2": UNANSWERABLE}


def _trace(case: Case, run: str = "r", refusal: bool = False) -> TraceRecord:
    return TraceRecord(
        run_id=run,
        case_id=case.id,
        repeat=0,
        split="dev",
        question=case.question,
        dimensions={},
        mode="query",
        workspace_slug="w",
        thread_slug=None,
        answer="It was 1.4 mph.",
        reasoning=None,
        response_type=None,
        refusal_path="deterministic" if refusal else None,
        retrieved=[RetrievedPassage(doc_id="A.pdf", chunk_id=None, text="at 1.4 mph", score=0.8)],
        latency_s=None,
        tokens=None,
        model=None,
        error=None,
        started_at="t",
    )


def test_split_is_by_question_and_covers_both_sides() -> None:
    splits = {judge_split(f"q-{i}") for i in range(40)}
    assert splits == {"judge-dev", "judge-test"}
    assert judge_split("q-7") == judge_split("q-7")


def test_judges_offered_by_answerability_and_never_for_rule_decided_traces() -> None:
    assert labelable_judges(_trace(ANSWERABLE), ANSWERABLE) == [
        "groundedness",
        "correctness",
        "refusal",
    ]
    assert labelable_judges(_trace(UNANSWERABLE), UNANSWERABLE) == ["groundedness", "abstention"]
    assert labelable_judges(_trace(ANSWERABLE, refusal=True), ANSWERABLE) == []


def test_page_hides_run_ids(tmp_path: Path) -> None:
    traces = [_trace(ANSWERABLE, run="secret-config-run")]
    out = tmp_path / "page.html"
    counts = export_label_page(traces, CASES, out, seed=1)
    assert counts["correctness"] == 1
    assert counts["abstention"] == 0
    assert "secret-config-run" not in out.read_text()


def test_import_requires_every_applicable_mark(tmp_path: Path) -> None:
    traces = [_trace(ANSWERABLE), _trace(UNANSWERABLE)]
    key_a, key_u = trace_key(traces[0]), trace_key(traces[1])
    labels = {
        key_a: {"marks": {"groundedness": "Pass", "correctness": "Fail", "refusal": "skip"}},
        key_u: {"marks": {"groundedness": "Pass"}},
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"traces_sha256": traces_sha256(traces), "labels": labels}))
    with pytest.raises(JudgeLabelsImportError, match="1 mark"):
        load_judge_labels(path, traces, CASES)
    labels[key_u]["marks"]["abstention"] = "Fail"
    path.write_text(json.dumps({"traces_sha256": traces_sha256(traces), "labels": labels}))
    loaded = load_judge_labels(path, traces, CASES)
    rows = labelled_cases(loaded, traces)
    assert len(rows) == 4  # the skip is dropped


def _verdict(result: str | None) -> Verdict:
    return Verdict.model_validate(
        {
            "judge": "groundedness",
            "result": result,
            "reason": "",
            "source": "judge" if result else "error",
            "prompt_sha256": "x",
        }
    )


def test_tpr_and_tnr_by_hand() -> None:
    # Human Pass x3 (judge passes 2), human Fail x2 (judge fails 1), one judge error.
    human = ["Pass", "Pass", "Pass", "Fail", "Fail", "Fail"]
    judged = ["Pass", "Pass", "Fail", "Fail", "Pass", None]
    rows = [
        LabelledCase(key=f"t{i}", judge="groundedness", human=h, split="judge-dev")  # pyright: ignore[reportArgumentType]
        for i, h in enumerate(human)
    ]
    verdicts = {(f"t{i}", "groundedness"): _verdict(v) for i, v in enumerate(judged)}
    result = agreement("groundedness", "judge-dev", rows, verdicts)  # pyright: ignore[reportArgumentType]
    assert (result.human_pass, result.human_fail, result.errors) == (3, 2, 1)
    assert result.tpr == pytest.approx(2 / 3)
    assert result.tnr == pytest.approx(1 / 2)
    assert result.tpr_ci[0] < 2 / 3 < result.tpr_ci[1]
