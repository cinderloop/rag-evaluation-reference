"""Eval-gate thresholds: derived from a scored baseline run, checked against any run.

A gated metric fails when it moves past its threshold in the bad direction. Thresholds
sit below (or above) the baseline by a margin that one run of this size would rarely
exceed by chance (ADR 0012):

    margin = z * SE,  z = inverse normal CDF of (1 - alpha / k)

- alpha is the family-wise false-alarm rate for one gate run.
- k is the number of gated metrics (a one-sided Bonferroni split).
- SE is the standard error of the baseline rate. With one run per question it is the
  binomial SE, using (x + 0.5) / (n + 1) so a 0% or 100% baseline still gets a margin.
  With repeated runs it is the cluster-robust SE over questions (ADR 0011).

The judges' validation results (docs/judges.md) decide which metrics are gated: an
unvalidated judge is reported but never fails the gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from rag_eval.mutations.compare import MetricName
from rag_eval.stats.clustered import clustered_se

Direction = Literal["min", "max"]

GATED: dict[MetricName, Direction] = {
    "passage_hit": "min",
    "groundedness": "min",
    "correctness": "min",
    "deterministic_refusal": "max",
}
REPORT_ONLY: tuple[MetricName, ...] = ("passage_recall", "refusal", "abstention")


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    direction: Direction
    baseline: float
    questions: int
    margin: float
    threshold: float


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_run: str
    family_alpha: float
    metrics: dict[MetricName, Threshold]
    report_only: tuple[MetricName, ...]


def standard_error(by_question: Mapping[str, Sequence[float]]) -> float:
    values = [v for runs in by_question.values() for v in runs]
    if not values:
        raise ValueError("no observations")
    if any(len(runs) > 1 for runs in by_question.values()):
        return clustered_se(by_question)
    n = len(values)
    p = (sum(values) + 0.5) / (n + 1)
    return math.sqrt(p * (1 - p) / n)


def derive(
    rates: Mapping[MetricName, Mapping[str, Sequence[float]]],
    baseline_run: str,
    family_alpha: float = 0.05,
) -> GateConfig:
    """Thresholds for the gated metrics from the baseline's per-question outcomes."""
    gated: list[MetricName] = [m for m in GATED if rates.get(m)]
    z = NormalDist().inv_cdf(1 - family_alpha / len(gated))
    metrics: dict[MetricName, Threshold] = {}
    for metric in gated:
        by_question = rates[metric]
        values = [v for runs in by_question.values() for v in runs]
        baseline = sum(values) / len(values)
        margin = z * standard_error(by_question)
        direction = GATED[metric]
        bound = baseline - margin if direction == "min" else baseline + margin
        metrics[metric] = Threshold(
            direction=direction,
            baseline=round(baseline, 4),
            questions=len(by_question),
            margin=round(margin, 4),
            threshold=round(min(1.0, max(0.0, bound)), 4),
        )
    return GateConfig(
        baseline_run=baseline_run,
        family_alpha=family_alpha,
        metrics=metrics,
        report_only=REPORT_ONLY,
    )


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricName
    value: float
    threshold: float | None  # None for report-only metrics
    direction: Direction | None
    passed: bool | None  # None for report-only metrics


def check(
    rates: Mapping[MetricName, Mapping[str, Sequence[float]]], config: GateConfig
) -> list[Check]:
    """Each metric's value in the run against its threshold; report-only metrics pass through."""
    out: list[Check] = []
    for metric, t in config.metrics.items():
        values = [v for runs in rates.get(metric, {}).values() for v in runs]
        if not values:
            raise ValueError(f"run has no values for gated metric {metric}")
        value = sum(values) / len(values)
        passed = value >= t.threshold if t.direction == "min" else value <= t.threshold
        out.append(
            Check(
                metric=metric,
                value=value,
                threshold=t.threshold,
                direction=t.direction,
                passed=passed,
            )
        )
    for metric in config.report_only:
        values = [v for runs in rates.get(metric, {}).values() for v in runs]
        if values:
            out.append(
                Check(
                    metric=metric,
                    value=sum(values) / len(values),
                    threshold=None,
                    direction=None,
                    passed=None,
                )
            )
    return out


def load_gate_config(path: Path) -> GateConfig:
    return GateConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def write_gate_config(config: GateConfig, path: Path, header: str) -> None:
    body = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    path.write_text(header + body, encoding="utf-8")
