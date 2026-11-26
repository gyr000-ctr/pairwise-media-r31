from .core import *  # noqa: F401,F403
from . import core as _core

# Re-export every public name (exception aliases, sentinels, helpers) even if
# it is missing from core.__all__, so `netpredict.<AnyDocumentedAlias>` works.
for _name in dir(_core):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_core, _name)
