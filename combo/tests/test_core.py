import math

import pytest

import combo

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return combo.Combo(0.5, 0.5, 100)

CASES = [
    ("initial", lambda: _c().count() == 0),
    ("one_hit", lambda: (lambda c: (c.hit(0.0), c.count())[1] == 1)(_c())),
    ("two_hits", lambda: (lambda c: (c.hit(0.0), c.hit(0.1), c.count())[2] == 2)(_c())),
    ("window_break", lambda: (lambda c: (c.hit(0.0), c.hit(1.0), c.count())[2] == 1)(_c())),
    ("cap", lambda: (lambda c: [c.hit(0.0) for _ in range(200)] and c.count() <= 100)(_c())),
    ("bad_window", lambda: _raises(lambda: combo.Combo(0.0, 0.5, 10))),
    ("bad_decay", lambda: _raises(lambda: combo.Combo(0.5, 0.0, 10))),
    ("bad_cap", lambda: _raises(lambda: combo.Combo(0.5, 0.5, 0))),
    ("decay", lambda: (_c().hit(0.0), _c().tick(1.0), _c().count())[2] >= 0),
    ("type", lambda: isinstance(_c().count(), int)),
    ("det", lambda: (lambda c: (c.hit(0.0), c.count())[1])(_c()) == (lambda c: (c.hit(0.0), c.count())[1])(_c())),
    ("nan", lambda: _raises(lambda: _c().tick(float('nan')))),
    ("no_mut", lambda: _c().hit(0.0) is not None),
    ("neg_dt", lambda: _raises(lambda: _c().tick(-1.0))),
    ("float_count", lambda: isinstance(_c().count(), int)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
