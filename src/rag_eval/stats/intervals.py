"""Confidence intervals for a pass rate or a mean.

Small samples rule out the normal-approximation ("p ± 1.96·SE") interval: it can
extend below 0 or above 1 and is too narrow near 0% and 100%. A pass rate gets a Wilson
score interval; any other statistic gets a percentile bootstrap interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from statistics import NormalDist, fmean

Interval = tuple[float, float]


def z_value(confidence: float) -> float:
    """Two-sided critical value, e.g. 1.96 for 0.95."""
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a proportion; (0, 1) when there are no trials."""
    if n == 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be between 0 and n, got {successes} of {n}")
    z = z_value(confidence)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (q in [0, 1])."""
    if not sorted_values:
        raise ValueError("no values")
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = fmean,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval: resample the values with replacement, compute the
    statistic each time, and take the middle ``confidence`` share of the results."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(statistic(rng.choices(values, k=n)) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return (percentile(estimates, tail), percentile(estimates, 1 - tail))
