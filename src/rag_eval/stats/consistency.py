"""How consistent repeated runs of the same question are."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Consistency:
    items: int
    all_pass: float  # share of items that pass on every run
    all_fail: float  # share of items that fail on every run
    mixed: float  # share of items whose runs disagree


def consistency(outcomes: Mapping[str, Sequence[bool]]) -> Consistency:
    """Per item, whether its repeated runs all pass, all fail, or disagree."""
    rows = [list(runs) for runs in outcomes.values() if runs]
    if not rows:
        raise ValueError("no outcomes")
    n = len(rows)
    passing = sum(all(r) for r in rows)
    failing = sum(not any(r) for r in rows)
    return Consistency(n, passing / n, failing / n, (n - passing - failing) / n)
