"""Minimum detectable effect (MDE): the smallest true change a test of this size would
detect with the given power. A change smaller than the MDE can go undetected, so "no
significant difference" is reported as "no difference larger than the MDE was found",
not as "no difference".

Normal approximations, used for planning only (tests themselves are exact):

- paired pass/fail (McNemar), with a share ``d`` of items expected to change outcome
  between the configs: MDE = (z_alpha + z_power) * sqrt(d / n);
- two independent pass rates around ``p``: MDE = (z_alpha + z_power) * sqrt(2p(1-p) / n).
"""

from __future__ import annotations

import math
from statistics import NormalDist


def _z_sum(alpha: float, power: float) -> float:
    normal = NormalDist()
    return normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)


def mde_paired(n: int, discordant_share: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """MDE in pass rate for two configs on the same n items (McNemar)."""
    if n <= 0 or not 0 < discordant_share <= 1:
        raise ValueError("need n > 0 and 0 < discordant_share <= 1")
    return _z_sum(alpha, power) * math.sqrt(discordant_share / n)


def mde_two_rates(n: int, p: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """MDE between two independent pass rates near p, each from n items."""
    if n <= 0 or not 0 < p < 1:
        raise ValueError("need n > 0 and 0 < p < 1")
    return _z_sum(alpha, power) * math.sqrt(2 * p * (1 - p) / n)
