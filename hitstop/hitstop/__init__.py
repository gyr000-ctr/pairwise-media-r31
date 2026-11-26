from . import core
from .core import *  # noqa: F401,F403
from .core import HitStop, HitStopError, Hitstop, TickResult  # noqa: F401


def __getattr__(name):
    # 未在 core 中预定义的异常名，交给 core 的 PEP 562 兜底分类生成
    return getattr(core, name)


def __dir__():
    return sorted(set(globals()) | set(dir(core)))
