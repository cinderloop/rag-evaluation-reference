"""Repeated runs of the same question are not independent observations.

When each question is run several times, the runs of one question tend to succeed or
fail together. Treating 66 questions with 3 runs each as 198 independent cases understates the
uncertainty. Questions are the clusters: the standard error and the bootstrap both work
on whole questions, never on single runs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from rag_eval.stats.intervals import Interval, percentile


def _pooled_mean(clusters: Sequence[Sequence[float]]) -> float:
    total = sum(len(c) for c in clusters)
    if total == 0:
        raise ValueError("no observations")
    return sum(sum(c) for c in clusters) / total


def clustered_mean(clusters: Mapping[str, Sequence[float]]) -> float:
    """Mean over every observation (each run counts once)."""
    return _pooled_mean(list(clusters.values()))


def clustered_se(clusters: Mapping[str, Sequence[float]]) -> float:
    """Cluster-robust standard error of the pooled mean (with the G/(G-1) small-sample
    factor): sqrt(G/(G-1) * sum over clusters of (sum of residuals)^2) / N."""
    groups = [list(c) for c in clusters.values() if c]
    g = len(groups)
    if g < 2:
        raise ValueError("need at least two clusters")
    mean = _pooled_mean(groups)
    n = sum(len(c) for c in groups)
    squared = sum(sum(x - mean for x in c) ** 2 for c in groups)
    return math.sqrt(g / (g - 1) * squared) / n


def cluster_bootstrap_ci(
    clusters: Mapping[str, Sequence[float]],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap that resamples whole questions, keeping all their runs."""
    groups = [list(c) for c in clusters.values() if c]
    if not groups:
        raise ValueError("no observations")
    rng = random.Random(seed)
    estimates = sorted(_pooled_mean(rng.choices(groups, k=len(groups))) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return (percentile(estimates, tail), percentile(estimates, 1 - tail))
