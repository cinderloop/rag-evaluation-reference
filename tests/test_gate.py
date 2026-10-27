"""Gate threshold logic, checked against hand-computed values."""

from __future__ import annotations

import pytest

from rag_eval.gate.thresholds import check, derive, standard_error
from rag_eval.mutations.compare import MetricName


def _questions(passes: int, total: int) -> dict[str, list[float]]:
    return {f"q{i}": [1.0 if i < passes else 0.0] for i in range(total)}


def test_single_gated_metric_by_hand() -> None:
    # 5 of 10 pass. p = (5 + 0.5) / 11 = 0.5, SE = sqrt(0.25 / 10) = 0.158114.
    # One gated metric: z = inverse normal CDF of 0.95 = 1.644854; margin = 0.260074.
    config = derive({"correctness": _questions(5, 10)}, "base")
    t = config.metrics["correctness"]
    assert t.baseline == 0.5
    assert t.margin == pytest.approx(0.2601, abs=1e-4)
    assert t.threshold == pytest.approx(0.2399, abs=1e-4)


def test_zero_baseline_still_gets_a_margin() -> None:
    # 0 of 10 refused. p = 0.5 / 11 = 0.045455, SE = sqrt(0.045455 * 0.954545 / 10)
    # = 0.065870; z = 1.644854; margin = 0.108347, so the limit is 10.8%.
    config = derive({"deterministic_refusal": _questions(0, 10)}, "base")
    t = config.metrics["deterministic_refusal"]
    assert t.direction == "max"
    assert t.threshold == pytest.approx(0.1083, abs=1e-4)


def test_margin_splits_alpha_across_gated_metrics() -> None:
    # Two gated metrics: z = inverse normal CDF of 0.975 = 1.959964;
    # margin = 1.959964 * 0.158114 = 0.309898.
    config = derive({"correctness": _questions(5, 10), "groundedness": _questions(5, 10)}, "base")
    assert config.metrics["correctness"].margin == pytest.approx(0.3099, abs=1e-4)


def test_repeated_runs_use_the_clustered_standard_error() -> None:
    # Pooled mean 0.5; residual sums +1, -1, 0; SE = sqrt(1.5 * 2) / 6 = 0.288675.
    clusters = {"a": [1.0, 1.0], "b": [0.0, 0.0], "c": [1.0, 0.0]}
    assert standard_error(clusters) == pytest.approx(0.288675, abs=1e-6)


def test_check_fails_only_gated_metrics_past_their_threshold() -> None:
    config = derive(
        {"correctness": _questions(5, 10), "deterministic_refusal": _questions(0, 10)}, "base"
    )
    # Thresholds with two gated metrics: correctness >= 0.5 - 0.3099 = 19.0%; refusals <= 12.9%.
    rates: dict[MetricName, dict[str, list[float]]] = {
        "correctness": _questions(1, 10),  # 10%, below the threshold
        "deterministic_refusal": _questions(0, 10),  # 0%, within its limit
        "abstention": _questions(1, 10),  # report only: never fails
    }
    results = {r.metric: r for r in check(rates, config)}
    assert results["correctness"].passed is False
    assert results["deterministic_refusal"].passed is True
    assert results["abstention"].passed is None
