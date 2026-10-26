"""Score a run: retrieval metrics for every answerable trace, binary judges per case.

Writes ``scores.jsonl`` (one record per trace) and ``summary.json`` (rates per metric,
the judge config, model and prompt hashes, and cost) next to the run's traces.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from threading import Lock

from pydantic import BaseModel, ConfigDict

from rag_eval.dataset.schema import Case
from rag_eval.evaluators.judges.backends import CachedBackend, JudgeConfig, price
from rag_eval.evaluators.judges.judges import (
    JudgeName,
    JudgeSpec,
    Verdict,
    judge,
    to_test_case,
    user_prompt,
)
from rag_eval.evaluators.retrieval import RetrievalScores, retrieval_scores
from rag_eval.runner.records import TraceRecord

CHARS_PER_TOKEN = 4  # rough, for estimates only
EST_OUTPUT_TOKENS = 80


class TraceScores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    case_id: str
    repeat: int
    answerable: bool
    retrieval: RetrievalScores | None
    verdicts: dict[JudgeName, Verdict]


def load_traces(run_dir: Path) -> list[TraceRecord]:
    lines = (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    return [TraceRecord.model_validate_json(line) for line in lines if line.strip()]


@dataclass
class Job:
    index: int
    spec: JudgeSpec
    trace: TraceRecord
    case: Case

    def prompt(self) -> tuple[str, str]:
        return self.spec.system_prompt, user_prompt(self.spec, to_test_case(self.trace, self.case))


def plan_jobs(
    traces: Sequence[TraceRecord], cases: dict[str, Case], specs: Sequence[JudgeSpec]
) -> list[Job]:
    """Every (trace, judge) pair the judge applies to; traces with run errors are skipped."""
    jobs: list[Job] = []
    for index, trace in enumerate(traces):
        if trace.error:
            continue
        case = cases[trace.case_id]
        jobs.extend(Job(index, s, trace, case) for s in specs if s.applies(case.answerable))
    return jobs


@dataclass(frozen=True)
class Estimate:
    calls: int
    cached: int
    by_rule: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


def needs_call(job: Job) -> bool:
    return job.trace.refusal_path != "deterministic"


def estimate(jobs: Sequence[Job], backend: CachedBackend) -> Estimate:
    by_rule = sum(not needs_call(j) for j in jobs)
    tokens_in = cached = calls = 0
    for job in jobs:
        if not needs_call(job):
            continue
        system, user = job.prompt()
        if backend.is_cached(system, user):
            cached += 1
            continue
        calls += 1
        tokens_in += (len(system) + len(user)) // CHARS_PER_TOKEN
    tokens_out = calls * EST_OUTPUT_TOKENS
    return Estimate(
        calls, cached, by_rule, tokens_in, tokens_out, price(backend.config, tokens_in, tokens_out)
    )


class CostCapReachedError(RuntimeError):
    pass


@dataclass
class Spend:
    cap_usd: float
    per_call_usd: float
    spent_usd: float = 0.0
    lock: Lock = field(default_factory=Lock)

    def reserve(self) -> None:
        with self.lock:
            if self.spent_usd + self.per_call_usd > self.cap_usd:
                raise CostCapReachedError(
                    f"stopping: next call would exceed the ${self.cap_usd:.2f} cap"
                )
            self.spent_usd += self.per_call_usd

    def settle(self, reserved: float, actual: float | None) -> None:
        with self.lock:
            self.spent_usd += (actual if actual is not None else reserved) - reserved


def run_judges(
    jobs: Sequence[Job],
    backend: CachedBackend,
    spend: Spend,
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> dict[tuple[int, JudgeName], Verdict]:
    verdicts: dict[tuple[int, JudgeName], Verdict] = {}
    done = 0
    lock = Lock()

    def one(job: Job) -> None:
        nonlocal done
        system, user = job.prompt()
        fresh = needs_call(job) and not backend.is_cached(system, user)
        if fresh:
            spend.reserve()
        verdict = judge(job.spec, to_test_case(job.trace, job.case), backend)
        if fresh:
            spend.settle(spend.per_call_usd, verdict.cost_usd)
        with lock:
            verdicts[(job.index, job.spec.name)] = verdict
            done += 1
            if progress:
                progress(done, len(jobs))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(one, j) for j in jobs]:
            future.result()
    return verdicts


def score_traces(
    traces: Sequence[TraceRecord],
    cases: dict[str, Case],
    verdicts: dict[tuple[int, JudgeName], Verdict],
) -> list[TraceScores]:
    scores: list[TraceScores] = []
    for index, trace in enumerate(traces):
        case = cases[trace.case_id]
        scores.append(
            TraceScores(
                run_id=trace.run_id,
                case_id=trace.case_id,
                repeat=trace.repeat,
                answerable=case.answerable,
                retrieval=None if trace.error else retrieval_scores(case, trace.retrieved),
                verdicts={name: v for (i, name), v in verdicts.items() if i == index},
            )
        )
    return scores


def summarize(
    scores: Sequence[TraceScores],
    specs: Sequence[JudgeSpec],
    judge_config: JudgeConfig | None,
    judge_config_path: Path | None,
    spent_usd: float,
) -> dict[str, object]:
    retrieval = [s.retrieval for s in scores if s.retrieval is not None]

    def level(name: str) -> dict[str, float | int]:
        rows = [getattr(r, name) for r in retrieval]
        return {
            "n": len(rows),
            "hit_rate": mean(r.hit for r in rows) if rows else 0.0,
            "recall": mean(r.recall for r in rows) if rows else 0.0,
            "mrr": mean(r.reciprocal_rank for r in rows) if rows else 0.0,
        }

    judges: dict[str, object] = {}
    for spec in specs:
        verdicts = [s.verdicts[spec.name] for s in scores if spec.name in s.verdicts]
        graded = [v for v in verdicts if v.result is not None]
        passed = sum(v.result == "Pass" for v in graded)
        judges[spec.name] = {
            "failure_modes": spec.failure_modes,
            "prompt_sha256": spec.prompt_sha256,
            "n": len(graded),
            "passed": passed,
            "pass_rate": passed / len(graded) if graded else None,
            "by_rule": sum(v.source == "rule" for v in verdicts),
            "errors": sum(v.source == "error" for v in verdicts),
        }
    config_ref = None
    if judge_config_path is not None:
        config_ref = {
            "path": str(judge_config_path),
            "sha256": hashlib.sha256(judge_config_path.read_bytes()).hexdigest(),
        }
    return {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_traces": len(scores),
        "retrieval": {"passage": level("passage"), "document": level("document")},
        "judge_config": config_ref,
        "judge_provider": judge_config.provider if judge_config else None,
        "judge_model": judge_config.model if judge_config else None,
        "judges": judges,
        "judge_cost_usd": round(spent_usd, 6),
    }


def write_scores(run_dir: Path, scores: Sequence[TraceScores], summary: dict[str, object]) -> None:
    (run_dir / "scores.jsonl").write_text(
        "".join(s.model_dump_json() + "\n" for s in scores), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
