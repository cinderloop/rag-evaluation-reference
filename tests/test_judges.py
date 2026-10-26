from __future__ import annotations

import json
import stat
from pathlib import Path

import httpx
import pytest

from rag_eval.dataset.schema import Case, Evidence
from rag_eval.evaluators.evaluate import CostCapReachedError, Spend, plan_jobs
from rag_eval.evaluators.judges.backends import (
    CachedBackend,
    ClaudeCodeBackend,
    JudgeConfig,
    JudgeReply,
    OpenAICompatibleBackend,
)
from rag_eval.evaluators.judges.judges import (
    JUDGES,
    BinaryJudgeMetric,
    JudgeOutputError,
    parse_verdict,
    to_test_case,
    user_prompt,
)
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


def _trace(case: Case, answer: str = "1.4 mph", refusal: bool = False) -> TraceRecord:
    return TraceRecord(
        run_id="r",
        case_id=case.id,
        repeat=0,
        split="dev",
        question=case.question,
        dimensions={},
        mode="query",
        workspace_slug="w",
        thread_slug=None,
        answer=answer,
        reasoning=None,
        response_type=None,
        refusal_path="deterministic" if refusal else None,
        retrieved=[]
        if refusal
        else [
            RetrievedPassage(doc_id="A.pdf", chunk_id=None, text="It hit at 1.4 mph.", score=0.8)
        ],
        latency_s=None,
        tokens=None,
        model=None,
        error=None,
        started_at="t",
    )


class FakeBackend:
    def __init__(self, text: str) -> None:
        self.config = JudgeConfig(provider="claude_code", model="fake-model")
        self.text = text
        self.calls = 0

    def complete(self, system: str, user: str) -> JudgeReply:
        self.calls += 1
        return JudgeReply(
            text=self.text,
            model_served="fake-model",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        )


def test_parse_verdict_accepts_fenced_json_and_rejects_other_results() -> None:
    assert parse_verdict('```json\n{"result": "Fail", "reason": "x"}\n```') == ("Fail", "x")
    with pytest.raises(JudgeOutputError):
        parse_verdict('{"result": "Maybe", "reason": "x"}')
    with pytest.raises(JudgeOutputError):
        parse_verdict("Pass")


def test_prompts_show_only_what_each_judge_needs() -> None:
    case = to_test_case(_trace(ANSWERABLE), ANSWERABLE)
    grounded = user_prompt(JUDGES["groundedness"], case)
    assert "[1] A.pdf: It hit at 1.4 mph." in grounded
    assert "Expected" not in grounded
    correct = user_prompt(JUDGES["correctness"], case)
    assert "Expected answer: 1.4 mph" in correct
    assert "Passages" not in correct


def test_deterministic_refusal_is_decided_by_rule_without_a_call() -> None:
    backend = FakeBackend('{"result": "Pass", "reason": "x"}')
    case = to_test_case(_trace(ANSWERABLE, answer="no", refusal=True), ANSWERABLE)
    metric = BinaryJudgeMetric(JUDGES["correctness"], backend)
    assert metric.measure(case) == 0.0
    assert metric.verdict is not None
    assert metric.verdict.source == "rule"
    assert backend.calls == 0


def test_metric_scores_one_or_zero_and_flags_unusable_replies() -> None:
    case = to_test_case(_trace(ANSWERABLE), ANSWERABLE)
    passing = BinaryJudgeMetric(
        JUDGES["groundedness"], FakeBackend('{"result":"Pass","reason":"ok"}')
    )
    assert passing.measure(case) == 1.0
    assert passing.is_successful()
    broken = BinaryJudgeMetric(JUDGES["groundedness"], FakeBackend("not json"))
    assert broken.measure(case) == 0.0
    assert broken.error is not None
    assert not broken.is_successful()


def test_judges_apply_by_answerability_and_skip_errored_traces() -> None:
    errored = _trace(ANSWERABLE).model_copy(update={"error": "timeout"})
    traces = [_trace(ANSWERABLE), _trace(UNANSWERABLE), errored]
    jobs = plan_jobs(traces, {"q1": ANSWERABLE, "q2": UNANSWERABLE}, list(JUDGES.values()))
    names = sorted((j.index, j.spec.name) for j in jobs)
    assert names == [
        (0, "correctness"),
        (0, "groundedness"),
        (0, "refusal"),
        (1, "abstention"),
        (1, "groundedness"),
    ]


def test_cache_sends_each_prompt_once(tmp_path: Path) -> None:
    inner = FakeBackend('{"result":"Pass","reason":"ok"}')
    backend = CachedBackend(inner, tmp_path)
    first = backend.complete("system", "user")
    second = backend.complete("system", "user")
    assert inner.calls == 1
    assert (first.cached, second.cached) == (False, True)


def test_spend_cap_stops_before_exceeding() -> None:
    spend = Spend(cap_usd=0.25, per_call_usd=0.1)
    spend.reserve()
    spend.reserve()
    with pytest.raises(CostCapReachedError):
        spend.reserve()


def test_openai_compatible_backend_reads_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "k")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer k"
        body = json.loads(request.content)
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        return httpx.Response(
            200,
            json={
                "model": "m-served",
                "choices": [{"message": {"content": '{"result":"Pass","reason":"r"}'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            },
        )

    config = JudgeConfig(
        provider="openai_compatible",
        model="m",
        base_url="https://example.test/v1",
        api_key_env="TEST_KEY",
        price_input_per_mtok=1.0,
        price_output_per_mtok=10.0,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    reply = OpenAICompatibleBackend(config, client=client).complete("s", "u")
    assert reply.model_served == "m-served"
    assert reply.cost_usd == pytest.approx(0.002)


def test_claude_code_backend_parses_headless_json(tmp_path: Path) -> None:
    script = tmp_path / "claude"
    payload: dict[str, object] = {
        "is_error": False,
        "result": '{"result":"Fail","reason":"r"}',
        "total_cost_usd": 0.0002,
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 500, "output_tokens": 20},
        "modelUsage": {"claude-haiku-5-5": {"costUSD": 0.0002}},
    }
    script.write_text(f"#!/bin/sh\ncat > /dev/null\necho '{json.dumps(payload)}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    config = JudgeConfig(provider="claude_code", model="claude-haiku-5-5")
    reply = ClaudeCodeBackend(config, executable=str(script)).complete("s", "u")
    assert parse_verdict(reply.text) == ("Fail", "r")
    assert (reply.input_tokens, reply.output_tokens, reply.cost_usd) == (502, 20, 0.0002)
