"""Binary LLM judges, one per failure mode, as DeepEval custom metrics.

Each judge asks one Pass/Fail question about one trace. Its prompt
(``prompts/<name>.md``) defines Pass and Fail and carries three examples (a clear
pass, a clear fail and a borderline case) drawn from the error-analysis labels. A rule
decides instead of the judge wherever one can: when AnythingLLM refused without
calling the LLM (no passage cleared the similarity threshold), the verdict follows from
that fact alone.

| Judge | Failure modes (docs/failure-taxonomy.md) | Cases | Sees |
|---|---|---|---|
| groundedness | B guesses, E borrowed facts, F extra claims | all | question, passages, answer |
| correctness | D misreadings, and any wrong answer | answerable | question, expected, answer |
| abstention | E trick questions | unanswerable | question, why unanswerable, answer |
| refusal | C half the answer declined | answerable | question, passages, answer |
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from typing import Literal, cast

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.schema import Case
from rag_eval.evaluators.judges.backends import JudgeBackend, JudgeBackendError
from rag_eval.runner.records import TraceRecord

JudgeName = Literal["groundedness", "correctness", "abstention", "refusal"]
Result = Literal["Pass", "Fail"]
UNANSWERABLE_NOTE = "The question cannot be answered from the reports."


@dataclass(frozen=True)
class JudgeSpec:
    name: JudgeName
    failure_modes: str
    applies_to: Literal["all", "answerable", "unanswerable"]
    shows_passages: bool
    shows_expected: bool
    # Verdict when AnythingLLM refused without an LLM call (no passage retrieved).
    on_deterministic_refusal: Result
    deterministic_reason: str

    @cached_property
    def system_prompt(self) -> str:
        path = resources.files("rag_eval.evaluators.judges") / "prompts" / f"{self.name}.md"
        return path.read_text(encoding="utf-8")

    @cached_property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.system_prompt.encode()).hexdigest()

    def applies(self, answerable: bool) -> bool:
        return self.applies_to == "all" or (self.applies_to == "answerable") == answerable


JUDGES: dict[JudgeName, JudgeSpec] = {
    "groundedness": JudgeSpec(
        "groundedness",
        "B, E, F",
        "all",
        shows_passages=True,
        shows_expected=False,
        on_deterministic_refusal="Pass",
        deterministic_reason="Refused without calling the LLM, so the answer makes no claims.",
    ),
    "correctness": JudgeSpec(
        "correctness",
        "D",
        "answerable",
        shows_passages=False,
        shows_expected=True,
        on_deterministic_refusal="Fail",
        deterministic_reason="Refused without calling the LLM, so no answer was given.",
    ),
    "abstention": JudgeSpec(
        "abstention",
        "E",
        "unanswerable",
        shows_passages=False,
        shows_expected=True,
        on_deterministic_refusal="Pass",
        deterministic_reason="Refused without calling the LLM, which is the right outcome.",
    ),
    "refusal": JudgeSpec(
        "refusal",
        "C",
        "answerable",
        shows_passages=True,
        shows_expected=False,
        on_deterministic_refusal="Pass",
        deterministic_reason="Refused because no passage was retrieved, so nothing was held back.",
    ),
}


# Error-analysis traces whose content appears as few-shot examples in each judge
# prompt. Judge validation must not count these, or it would test the judge on its own
# examples.
FEW_SHOT_TRACES: dict[JudgeName, tuple[str, ...]] = {
    "groundedness": ("t-ca5e306c87", "t-090bb16fe7", "t-0233916026"),
    "correctness": ("t-eef70496b1", "t-5d8e8ab744", "t-8254cd6ea8"),
    "abstention": ("t-9b67a9d4f3", "t-2eaf756e98", "t-8edb7100f3"),
    "refusal": ("t-6f501fe3a5", "t-89abcae936", "t-4ddd8de9e8"),
}


def to_test_case(trace: TraceRecord, case: Case) -> LLMTestCase:
    """A trace as a DeepEval test case; the metadata carries what the rules need."""
    return LLMTestCase(
        input=case.question,
        actual_output=trace.answer or "",
        expected_output=case.expected_answer or (None if case.answerable else UNANSWERABLE_NOTE),
        retrieval_context=[f"{p.doc_id}: {p.text}" for p in trace.retrieved],
        metadata={
            "case_id": case.id,
            "answerable": case.answerable,
            "refusal_path": trace.refusal_path,
        },
    )


def user_prompt(spec: JudgeSpec, test_case: LLMTestCase) -> str:
    parts = [f"Question: {test_case.input}"]
    if spec.shows_expected:
        label = (
            "Expected answer" if spec.applies_to == "answerable" else "Why it cannot be answered"
        )
        parts.append(f"{label}: {test_case.expected_output}")
    if spec.shows_passages:
        context = cast(list[str], test_case.retrieval_context or [])
        passages = "\n".join(f"[{i}] {text}" for i, text in enumerate(context, start=1))
        parts.append(f"Passages:\n{passages or '(none retrieved)'}")
    parts.append(f"Answer: {test_case.actual_output}")
    return "\n".join(parts)


class JudgeOutputError(ValueError):
    pass


def parse_verdict(text: str) -> tuple[Result, str]:
    """Read ``{"result": "Pass"|"Fail", "reason": "..."}`` from a judge reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeOutputError(f"no JSON object in judge reply: {text[:200]!r}")
    try:
        data = cast(dict[str, object], json.loads(match.group(0)))
    except json.JSONDecodeError as error:
        raise JudgeOutputError(f"invalid JSON in judge reply: {text[:200]!r}") from error
    result = data.get("result")
    if result not in ("Pass", "Fail"):
        raise JudgeOutputError(f"result must be Pass or Fail, got {result!r}")
    return result, str(data.get("reason", ""))


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    judge: JudgeName
    result: Result | None  # None when the judge failed to give a usable verdict
    reason: str
    source: Literal["rule", "judge", "error"]
    judge_model: str | None = None
    prompt_sha256: str
    cost_usd: float | None = None
    cached: bool = False


def judge(spec: JudgeSpec, test_case: LLMTestCase, backend: JudgeBackend) -> Verdict:
    metadata = cast(dict[str, object], test_case.metadata or {})  # pyright: ignore[reportUnknownMemberType]
    if metadata.get("refusal_path") == "deterministic":
        return Verdict(
            judge=spec.name,
            result=spec.on_deterministic_refusal,
            reason=spec.deterministic_reason,
            source="rule",
            prompt_sha256=spec.prompt_sha256,
        )
    try:
        reply = backend.complete(spec.system_prompt, user_prompt(spec, test_case))
        result, reason = parse_verdict(reply.text)
    except (JudgeBackendError, JudgeOutputError) as error:
        return Verdict(
            judge=spec.name,
            result=None,
            reason=str(error),
            source="error",
            judge_model=backend.config.model,
            prompt_sha256=spec.prompt_sha256,
        )
    return Verdict(
        judge=spec.name,
        result=result,
        reason=reason,
        source="judge",
        judge_model=reply.model_served,
        prompt_sha256=spec.prompt_sha256,
        cost_usd=reply.cost_usd,
        cached=reply.cached,
    )


class BinaryJudgeMetric(BaseMetric):
    """A DeepEval metric whose score is 1 (Pass) or 0 (Fail), never a scale."""

    def __init__(self, spec: JudgeSpec, backend: JudgeBackend) -> None:
        self.spec = spec
        self.backend = backend
        self.threshold = 1.0
        self.include_reason = True
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_model = backend.config.model
        self.verdict: Verdict | None = None

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        verdict = judge(self.spec, test_case, self.backend)
        self.verdict = verdict
        self.reason = verdict.reason
        self.evaluation_cost = verdict.cost_usd
        if verdict.result is None:
            self.error = verdict.reason
            self.score = 0.0
        else:
            self.error = None
            self.score = 1.0 if verdict.result == "Pass" else 0.0
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.error is None and self.score == 1.0

    @property
    def __name__(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"{self.spec.name.capitalize()} (binary judge)"
