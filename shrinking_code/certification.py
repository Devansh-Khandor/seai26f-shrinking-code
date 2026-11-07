"""Numeric comparisons following course ezr.lua's same rule; see README provenance.

True means all three screens are within thresholds, not proven equivalence.
False means at least one screen fails, not necessarily statistical significance.
None means fewer than two observed values in either group.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import variance
from .stats import NumericSummary


@dataclass(frozen=True)
class Comparison:
    left: NumericSummary
    right: NumericSummary
    cohen: object
    ks_d: object
    ks: object
    cliffs: object
    same: object
    exceeded: tuple


def compare(xs, ys, Cohen=0.35, Ks=1.36, Cliffs=0.195):
    """Sort finite numeric observations, omit None, compute all diagnostics.

    KS is D / sqrt((n+m)/(n*m)); Cliff's delta is absolute, as in Lua.
    Minimum n=2 per group is an explicit safety guard absent in the source.
    """
    for threshold in (Cohen, Ks, Cliffs):
        if not isfinite(threshold) or threshold < 0:
            raise ValueError('Thresholds must be finite and nonnegative')
    xs, ys = ([v for v in values if v is not None] for values in (xs, ys))
    if any(not isfinite(v) for v in xs + ys):
        raise ValueError('Samples must contain finite numbers')
    xs, ys = sorted(xs), sorted(ys)
    a, b = NumericSummary.from_values(xs), NumericSummary.from_values(ys)
    n, m = a.count, b.count
    if min(n, m) < 2:
        return Comparison(a, b, None, None, None, None, None, ())
    pooled = sqrt(((n - 1) * variance(xs) + (m - 1) * variance(ys)) / (n + m - 2))
    cohen = abs(a.mean - b.mean) / (pooled + 1e-32)
    # Evaluate both empirical CDFs after each tied value; no random tie breaking.
    d = max(abs(bisect_right(xs, v) / n - bisect_right(ys, v) / m)
            for v in set(xs + ys))
    ks = d / sqrt((n + m) / (n * m))
    imbalance = sum(bisect_left(ys, x) - (m - bisect_right(ys, x)) for x in xs)
    cliffs = abs(imbalance) / (n * m)
    exceeded = tuple(name for name, value, limit in
                     [('cohen', cohen, Cohen), ('cliffs', cliffs, Cliffs), ('ks', ks, Ks)]
                     if value > limit)
    return Comparison(a, b, cohen, d, ks, cliffs, not exceeded, exceeded)


def same(xs, ys, Cohen=0.35, Ks=1.36, Cliffs=0.195):
    """Source-compatible boolean for sufficient samples, else None."""
    return compare(xs, ys, Cohen, Ks, Cliffs).same
