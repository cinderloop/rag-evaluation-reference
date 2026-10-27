"""Compare a mutated run with the baseline, metric by metric, against pre-registered
expectations (``data/mutations/expectations.yaml``, written before any mutation run).

Both runs answer the same questions with the same number of repeats, so every
comparison is paired by question:

- **Effect:** difference in the pooled rate (mutation minus baseline) over all runs, with
  a 95% interval from a bootstrap that resamples whole questions (ADR 0011).
- **Test:** exact McNemar on one outcome per question per config: Pass if it passes on
  the majority of its repeats. For a continuous metric (passage recall), the paired
  cluster bootstrap interval alone decides.
- **Detected:** the change goes in the expected direction and is significant (p < 0.05,
  or an interval that excludes zero). For "no change expected" (``none``), the result is
  "as expected" when it is not significant. Metrics without a prediction (``any``) are
  reported but not scored.
- **MDE:** the smallest change this comparison could reliably detect, from the observed
  share of questions whose outcome flips (at least one question).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from rag_eval.evaluators.evaluate import TraceScores
from rag_eval.runner.records import TraceRecord
from rag_eval.stats.intervals import Interval
from rag_eval.stats.paired import mcnemar_exact, paired_cluster_bootstrap_ci
from rag_eval.stats.power import mde_paired

Direction = Literal["down", "up", "none", "any"]
MetricName = Literal[
    "passage_hit",
    "passage_recall",
    "deterministic_refusal",
    "groundedness",
    "correctness",
    "refusal",
    "abstention",
]
ALPHA = 0.05


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    config: str
    change: str
    expect: dict[MetricName, Direction]
    targeted_cases: tuple[str, ...] = ()
    note: str = ""


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mutations: tuple[Expectation, ...]


def load_expectations(path: Path) -> Expectations:
    return Expectations.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


Extractor = Callable[[TraceScores, TraceRecord], float | None]


def _verdict(name: Literal["groundedness", "correctness", "refusal", "abstention"]) -> Extractor:
    def get(scores: TraceScores, _: TraceRecord) -> float | None:
        verdict = scores.verdicts.get(name)
        if verdict is None or verdict.result is None:
            return None
        return 1.0 if verdict.result == "Pass" else 0.0

    return get


def _passage_hit(scores: TraceScores, _: TraceRecord) -> float | None:
    return None if scores.retrieval is None else float(scores.retrieval.passage.hit)


def _passage_recall(scores: TraceScores, _: TraceRecord) -> float | None:
    return None if scores.retrieval is None else scores.retrieval.passage.recall


def _deterministic_refusal(_: TraceScores, trace: TraceRecord) -> float | None:
    return 1.0 if trace.refusal_path == "deterministic" else 0.0


EXTRACTORS: dict[MetricName, Extractor] = {
    "passage_hit": _passage_hit,
    "passage_recall": _passage_recall,
    "deterministic_refusal": _deterministic_refusal,
    "groundedness": _verdict("groundedness"),
    "correctness": _verdict("correctness"),
    "refusal": _verdict("refusal"),
    "abstention": _verdict("abstention"),
}
BINARY: frozenset[MetricName] = frozenset(EXTRACTORS) - {"passage_recall"}


def by_question(
    scores: Sequence[TraceScores], traces: Sequence[TraceRecord], metric: MetricName
) -> dict[str, list[float]]:
    trace_by_key = {(t.case_id, t.repeat): t for t in traces}
    out: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        value = EXTRACTORS[metric](s, trace_by_key[(s.case_id, s.repeat)])
        if value is not None:
            out[s.case_id].append(value)
    return dict(out)


class MetricComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricName
    expected: Direction
    questions: int
    baseline: float
    mutation: float
    difference: float
    interval: Interval
    p_value: float | None
    mde: float | None
    significant: bool
    outcome: Literal[
        "detected", "not detected", "as expected", "unexpected change", "not pre-registered"
    ]


def _majority(values: Sequence[float]) -> bool:
    return sum(values) > len(values) / 2


def compare_metric(
    metric: MetricName,
    expected: Direction,
    base: dict[str, list[float]],
    mut: dict[str, list[float]],
    *,
    seed: int = 0,
) -> MetricComparison:
    keys = sorted(set(base) & set(mut))
    a = {k: base[k] for k in keys}
    b = {k: mut[k] for k in keys}
    difference, interval = paired_cluster_bootstrap_ci(a, b, seed=seed)
    p_value: float | None = None
    mde: float | None = None
    if metric in BINARY:
        only_base = sum(_majority(a[k]) and not _majority(b[k]) for k in keys)
        only_mut = sum(_majority(b[k]) and not _majority(a[k]) for k in keys)
        p_value = mcnemar_exact(only_base, only_mut).p_value
        discordant = max(only_base + only_mut, 1) / len(keys)
        mde = mde_paired(len(keys), discordant)
        significant = p_value < ALPHA
    else:
        significant = interval[0] > 0 or interval[1] < 0
    pooled_a = sum(sum(v) for v in a.values()) / sum(len(v) for v in a.values())
    pooled_b = sum(sum(v) for v in b.values()) / sum(len(v) for v in b.values())
    direction_ok = (expected == "down" and difference < 0) or (expected == "up" and difference > 0)
    if expected == "any":
        outcome = "not pre-registered"
    elif expected == "none":
        outcome = "unexpected change" if significant else "as expected"
    else:
        outcome = "detected" if significant and direction_ok else "not detected"
    return MetricComparison(
        metric=metric,
        expected=expected,
        questions=len(keys),
        baseline=pooled_a,
        mutation=pooled_b,
        difference=difference,
        interval=interval,
        p_value=p_value,
        mde=mde,
        significant=significant,
        outcome=outcome,
    )


class TargetedCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    metric: MetricName
    baseline: list[float]
    mutation: list[float]


def targeted(
    cases: Sequence[str],
    base: dict[MetricName, dict[str, list[float]]],
    mut: dict[MetricName, dict[str, list[float]]],
) -> list[TargetedCase]:
    """Per-run outcomes for the specific questions a mutation targets."""
    return [
        TargetedCase(case_id=c, metric=m, baseline=base[m].get(c, []), mutation=mut[m].get(c, []))
        for c in cases
        for m in base
        if c in base[m] or c in mut[m]
    ]
