"""Statistics, checked against values computed by hand (working shown in comments)."""

from __future__ import annotations

import pytest

from rag_eval.stats.clustered import cluster_bootstrap_ci, clustered_mean, clustered_se
from rag_eval.stats.consistency import consistency
from rag_eval.stats.intervals import bootstrap_ci, percentile, wilson, z_value
from rag_eval.stats.paired import (
    mcnemar_exact,
    mcnemar_from_outcomes,
    paired_bootstrap_ci,
    paired_cluster_bootstrap_ci,
)
from rag_eval.stats.power import mde_paired, mde_two_rates

Z = 1.959964  # z for a two-sided 95% interval


def test_z_value() -> None:
    assert z_value(0.95) == pytest.approx(Z, abs=1e-6)


def test_wilson_zero_successes() -> None:
    # p = 0: centre = half = (z²/2n) / (1 + z²/n), so the upper bound is z² / (n + z²)
    # = 3.841459 / 13.841459 = 0.277533.
    low, high = wilson(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.277533, abs=1e-6)


def test_wilson_half() -> None:
    # p = 0.5, n = 10: centre = 0.5; half = z * sqrt(0.025 + z²/400) / (1 + z²/10)
    # = 1.959964 * 0.186020 / 1.384146 = 0.263407.
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.236593, abs=1e-5)
    assert high == pytest.approx(0.763407, abs=1e-5)


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="between 0 and n"):
        wilson(11, 10)
    assert wilson(0, 0) == (0.0, 1.0)


def test_percentile_interpolates_between_ranks() -> None:
    # Positions q * (n - 1): 0.5 -> 1.5 (between 2 and 3), 0.25 -> 0.75 (between 1 and 2).
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.25) == 1.75


def test_bootstrap_is_reproducible_and_degenerate_for_constant_data() -> None:
    values = [0.0, 1.0] * 25
    first = bootstrap_ci(values, seed=7)
    assert first == bootstrap_ci(values, seed=7)
    assert first[0] < 0.5 < first[1]
    assert bootstrap_ci([0.4] * 10) == pytest.approx((0.4, 0.4))


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        # 2 * P(X <= 1 | n = 7, p = 1/2) = 2 * (1 + 7) / 128
        (1, 6, 0.125),
        # 2 * (1 + 10 + 45) / 1024
        (2, 8, 0.109375),
        # 2 * 1 / 32
        (0, 5, 0.0625),
        # 2 * (1 + 6 + 15 + 20) / 64 = 1.3125, capped at 1
        (3, 3, 1.0),
        # no discordant items
        (0, 0, 1.0),
    ],
)
def test_mcnemar_exact(b: int, c: int, p: float) -> None:
    assert mcnemar_exact(b, c).p_value == pytest.approx(p)


def test_mcnemar_counts_only_discordant_items() -> None:
    a = [True, True, False, False, True]
    b = [True, False, True, True, True]
    result = mcnemar_from_outcomes(a, b)
    assert (result.b, result.c, result.discordant) == (1, 2, 3)


def test_paired_bootstrap_of_constant_difference() -> None:
    mean, interval = paired_bootstrap_ci([0, 0, 1, 1], [1, 1, 2, 2])
    assert mean == 1.0
    assert interval == pytest.approx((1.0, 1.0))


def test_clustered_se_by_hand() -> None:
    # Pooled mean 3/6 = 0.5. Residual sums per question: +1, -1, 0; squared sum 2.
    # SE = sqrt(G/(G-1) * 2) / N = sqrt(3) / 6 = 0.288675, larger than the naive
    # sqrt(0.25 / 6) = 0.204124 that treats the six runs as independent.
    clusters = {"a": [1.0, 1.0], "b": [0.0, 0.0], "c": [1.0, 0.0]}
    assert clustered_mean(clusters) == 0.5
    assert clustered_se(clusters) == pytest.approx(0.288675, abs=1e-6)


def test_cluster_bootstrap_is_wider_when_runs_move_together() -> None:
    # 30 questions, three runs each, all runs of a question agree.
    clusters = {f"q{i}": [float(i % 2)] * 3 for i in range(30)}
    flat = [x for runs in clusters.values() for x in runs]
    low_c, high_c = cluster_bootstrap_ci(clusters, resamples=2000)
    low_f, high_f = bootstrap_ci(flat, resamples=2000)
    assert (high_c - low_c) > (high_f - low_f)


def test_consistency_shares() -> None:
    result = consistency({"a": [True, True, True], "b": [True, False, True], "c": [False] * 3})
    assert (result.items, result.all_pass, result.all_fail, result.mixed) == pytest.approx(
        (3, 1 / 3, 1 / 3, 1 / 3)
    )


def test_mde_paired_by_hand() -> None:
    # (1.959964 + 0.841621) * sqrt(0.2 / 66) = 2.801585 * 0.0550482 = 0.154222
    assert mde_paired(66, 0.2) == pytest.approx(0.154222, abs=1e-5)


def test_mde_two_rates_by_hand() -> None:
    # 2.801585 * sqrt(2 * 0.25 / 100) = 2.801585 * 0.0707107 = 0.198102
    assert mde_two_rates(100, 0.5) == pytest.approx(0.198102, abs=1e-5)


def test_paired_cluster_bootstrap_by_hand() -> None:
    # A passes 2 of 6 runs, B passes 5 of 6: difference 5/6 - 2/6 = 0.5.
    a = {"q1": [1.0, 1.0], "q2": [0.0, 0.0], "q3": [0.0, 0.0]}
    b = {"q1": [1.0, 1.0], "q2": [1.0, 1.0], "q3": [1.0, 0.0]}
    difference, (low, high) = paired_cluster_bootstrap_ci(a, b, resamples=2000)
    assert difference == pytest.approx(0.5)
    assert low <= 0.5 <= high
    # Every question gets better or stays equal under B, so no resample favours A.
    assert low >= 0.0


def test_paired_cluster_bootstrap_needs_the_same_questions() -> None:
    with pytest.raises(ValueError, match="same"):
        paired_cluster_bootstrap_ci({"q1": [1.0]}, {"q2": [1.0]})
