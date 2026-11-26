from . import core
from .core import *  # noqa: F401,F403
from .core import (  # noqa: F401
    Combo,
    ComboError,
    MultiplierCurve,
    PiecewiseCurve,
    double_exponential,
    multiplier_for_count,
    multiplier_integral,
)


def __getattr__(name):
    # Use normal attribute lookup first; this finds dynamically created error
    # aliases before falling through to core's PEP 562 phantom generator.
    try:
        return getattr(core, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(dir(core)))
