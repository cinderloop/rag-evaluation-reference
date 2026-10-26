"""Comparing two configs on the same items.

Both configs answer the same questions, so the comparison is paired: only the items
where they disagree carry information about which is better. For pass/fail outcomes
that is the exact McNemar test; for continuous scores, a paired bootstrap of the
per-item differences. With repeated runs, ``paired_cluster_bootstrap_ci`` resamples
whole questions, and McNemar is applied per repeat or to one outcome per question.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from rag_eval.stats.intervals import Interval, percentile


@dataclass(frozen=True)
class McNemar:
    """b: items that pass under A and fail under B; c: fail under A, pass under B."""

    b: int
    c: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.b + self.c


def mcnemar_exact(b: int, c: int) -> McNemar:
    """Two-sided exact McNemar test: under "no difference", the b + c discordant items
    split like fair coin flips, so p = 2 * P(Binomial(b + c, 1/2) <= min(b, c)), capped
    at 1. No discordant items gives p = 1."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return McNemar(b, c, 1.0)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return McNemar(b, c, min(1.0, 2 * tail))


def mcnemar_from_outcomes(a: Sequence[bool], b: Sequence[bool]) -> McNemar:
    """Exact McNemar from paired pass/fail outcomes of configs A and B on the same items."""
    if len(a) != len(b):
        raise ValueError("outcomes must be paired (same length)")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    return mcnemar_exact(only_a, only_b)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, Interval]:
    """Mean of (b - a) over items, with a percentile bootstrap interval that resamples
    items (keeping each item's pair together)."""
    if len(a) != len(b) or not a:
        raise ValueError("need paired, non-empty scores")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    estimates = sorted(fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return fmean(diffs), (percentile(estimates, tail), percentile(estimates, 1 - tail))


def paired_cluster_bootstrap_ci(
    a: Mapping[str, Sequence[float]],
    b: Mapping[str, Sequence[float]],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, Interval]:
    """Difference in pooled means (B - A) for two configs run several times on the same
    questions, with a percentile bootstrap that resamples whole questions: a resampled
    question brings all its runs under both configs, so repeats are not counted as
    independent items."""
    if set(a) != set(b) or not a:
        raise ValueError("need the same, non-empty set of questions under both configs")
    keys = sorted(a)

    def difference(sample: Sequence[str]) -> float:
        runs_a = [x for k in sample for x in a[k]]
        runs_b = [x for k in sample for x in b[k]]
        return fmean(runs_b) - fmean(runs_a)

    rng = random.Random(seed)
    estimates = sorted(difference(rng.choices(keys, k=len(keys))) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return difference(keys), (percentile(estimates, tail), percentile(estimates, 1 - tail))
