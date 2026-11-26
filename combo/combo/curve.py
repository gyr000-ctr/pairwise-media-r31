"""Piecewise multiplier curve helpers."""

from .core import (  # noqa: F401
    MultiplierCurve,
    PiecewiseCurve,
    analytic_integral,
    combo_multiplier,
    multiplier_at,
    multiplier_for_count,
    multiplier_integral,
    numerical_integral,
    piecewise_integral,
    piecewise_multiplier,
    tier,
    tier_for_count,
)

integrate = multiplier_integral
integral = multiplier_integral
curve_integral = multiplier_integral
multiplier = multiplier_for_count

__all__ = [
    "PiecewiseCurve", "MultiplierCurve", "tier_for_count", "tier",
    "multiplier_for_count", "combo_multiplier", "piecewise_multiplier",
    "multiplier_at", "multiplier", "multiplier_integral",
    "piecewise_integral", "analytic_integral", "curve_integral",
    "numerical_integral", "integrate", "integral",
]
