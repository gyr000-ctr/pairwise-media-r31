"""Double-exponential decay and analytic primitives."""

from .core import (  # noqa: F401
    analytic_decay_integral,
    decay_curve,
    decay_integral,
    double_exp,
    double_exp_decay,
    double_exp_integral,
    double_exponential,
    double_exponential_decay,
    double_exponential_integral,
)


def solve_target(target, a=0.5, b=0.5, lambda1=0.5, lambda2=0.125, **kwargs):
    """Numerically solve ``a exp(-l1 t)+b exp(-l2 t)=target`` for t >= 0.

    The generic two-exponential equation has no useful elementary closed form
    for arbitrary rates; a bounded bisection gives a robust analytic-quality
    root when a solution exists.
    """
    if kwargs:
        from .core import _decay_parameters
        weights, rates = _decay_parameters(kwargs)
        a, b = weights
        lambda1, lambda2 = rates

    f0 = a + b
    if not (0.0 < target <= f0):
        raise ValueError("target must lie in (0, a+b]")
    if target == f0:
        return 0.0

    lo, hi = 0.0, 1.0
    while double_exponential(hi, a, b, lambda1, lambda2) > target:
        hi *= 2.0
        if hi > 1e308:
            raise ValueError("decay rates are too small to reach target")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if double_exponential(mid, a, b, lambda1, lambda2) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


time_to_value = solve_target

__all__ = [
    "double_exponential", "double_exp", "double_exponential_decay",
    "double_exp_decay", "decay_curve", "double_exponential_integral",
    "double_exp_integral", "analytic_decay_integral", "decay_integral",
    "solve_target", "time_to_value",
]
