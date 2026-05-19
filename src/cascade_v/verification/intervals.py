"""
intervals.py — Interval arithmetic for tracked uncertainty propagation.

Implements the patent's "confidence-bounded interval arithmetic" using
mpmath for arbitrary-precision IEEE-754-aware computation. Every numerical
quantity in the verification path can carry [lower, upper] bounds that
propagate correctly through arithmetic.

For our purposes, this is used to:
1. Bound Monte Carlo Shapley estimates with Hoeffding intervals
2. Propagate those bounds through the cluster -> instance weight composition
3. Report final per-creator weights as (point, [lo, hi]) tuples

We use mpmath's `mpi` interval type for exact rounding-aware arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath
import numpy as np

from cascade_v.config import INTERVAL_PRECISION_BITS

# Set mpmath precision once
mpmath.mp.prec = INTERVAL_PRECISION_BITS


# ---------------------------------------------------------------------------
# Interval type
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """[lower, upper] bound on a real value, with point estimate."""
    point: float
    lower: float
    upper: float

    def __post_init__(self):
        # Sanity: ensure point is within [lower, upper]
        if not (self.lower - 1e-9 <= self.point <= self.upper + 1e-9):
            # Snap point into interval
            self.point = max(self.lower, min(self.upper, self.point))

    @classmethod
    def point_value(cls, x: float) -> "Interval":
        """A degenerate interval at a single point (no uncertainty)."""
        return cls(point=float(x), lower=float(x), upper=float(x))

    @classmethod
    def from_bound(cls, point: float, half_width: float) -> "Interval":
        """Symmetric interval around a point."""
        return cls(point=float(point), lower=float(point - half_width), upper=float(point + half_width))

    def width(self) -> float:
        return float(self.upper - self.lower)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.point, self.lower, self.upper)

    def to_dict(self) -> dict:
        return {"point": self.point, "lower": self.lower, "upper": self.upper}

    # Arithmetic
    def __add__(self, other: "Interval") -> "Interval":
        a = mpmath.mpi(self.lower, self.upper)
        b = mpmath.mpi(other.lower, other.upper)
        c = a + b
        return Interval(
            point=float(self.point + other.point),
            lower=float(c.a), upper=float(c.b),
        )

    def __mul__(self, other: "Interval") -> "Interval":
        a = mpmath.mpi(self.lower, self.upper)
        b = mpmath.mpi(other.lower, other.upper)
        c = a * b
        return Interval(
            point=float(self.point * other.point),
            lower=float(c.a), upper=float(c.b),
        )

    def __truediv__(self, other: "Interval") -> "Interval":
        if other.lower <= 0 <= other.upper:
            raise ZeroDivisionError("interval division: divisor contains zero")
        a = mpmath.mpi(self.lower, self.upper)
        b = mpmath.mpi(other.lower, other.upper)
        c = a / b
        return Interval(
            point=float(self.point / other.point),
            lower=float(c.a), upper=float(c.b),
        )

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> "Interval":
        return Interval(
            point=float(np.clip(self.point, lo, hi)),
            lower=float(np.clip(self.lower, lo, hi)),
            upper=float(np.clip(self.upper, lo, hi)),
        )


# ---------------------------------------------------------------------------
# Compose interval weights through cluster -> instance pipeline
# ---------------------------------------------------------------------------

def compose_cluster_instance_intervals(
    cluster_weight: Interval,
    instance_weights_in_cluster: list[Interval],
) -> list[Interval]:
    """
    Multiply cluster weight by each instance weight to get final intervals.

    This is the key uncertainty-propagation step: a candidate's final weight
    is cluster_weight * instance_weight_within_cluster, and uncertainty in
    both compounds via interval multiplication.
    """
    return [cluster_weight * w for w in instance_weights_in_cluster]


def normalize_intervals(intervals: list[Interval], target_sum: float = 1.0) -> list[Interval]:
    """
    Normalize a list of intervals so their point estimates sum to target_sum,
    while preserving relative bounds.
    """
    point_sum = sum(iv.point for iv in intervals)
    if point_sum < 1e-12:
        n = len(intervals)
        uniform = target_sum / n
        return [Interval.point_value(uniform) for _ in range(n)]
    scale = target_sum / point_sum
    return [
        Interval(
            point=iv.point * scale,
            lower=iv.lower * scale,
            upper=iv.upper * scale,
        )
        for iv in intervals
    ]
