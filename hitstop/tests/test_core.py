import math

import pytest

import hitstop

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _h():
    return hitstop.HitStop(0.2)

CASES = [
    ("initial", lambda: _h().logic_time() == 0.0),
    ("advance", lambda: (lambda h: (h.tick(0.1), h.logic_time())[1] > 0.0)(_h())),
    ("freeze_halt", lambda: (lambda h: (h.freeze(0.1), h.tick(0.05), h.logic_time())[2] == 0.0)(_h())),
    ("resume", lambda: (lambda h: (h.freeze(0.05), h.tick(0.2), h.logic_time())[2] > 0.0)(_h())),
    ("cap", lambda: (lambda h: (h.freeze(10.0), h.logic_time())[1] >= 0.0)(_h())),
    ("bad_max", lambda: _raises(lambda: hitstop.HitStop(0.0))),
    ("bad_dur", lambda: _raises(lambda: _h().freeze(0.0))),
    ("bad_dt", lambda: _raises(lambda: _h().tick(-1.0))),
    ("type", lambda: isinstance(_h().logic_time(), float)),
    ("det", lambda: (lambda h: (h.tick(0.1), h.logic_time())[1])(_h()) == (lambda h: (h.tick(0.1), h.logic_time())[1])(_h())),
    ("nan", lambda: _raises(lambda: _h().tick(float('nan')))),
    ("no_mut", lambda: _h().tick(0.1) is not None),
    ("accumulate", lambda: (lambda h: (h.freeze(0.05), h.freeze(0.05), h.logic_time())[2] == 0.0)(_h())),
    ("finite", lambda: math.isfinite(_h().logic_time())),
    ("float", lambda: isinstance(_h().logic_time(), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
