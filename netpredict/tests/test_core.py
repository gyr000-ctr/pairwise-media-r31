import math

import pytest

import netpredict

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _p():
    return netpredict.Predict(0.2)

CASES = [
    ("predict_type", lambda: isinstance(_p().predict(), tuple)),
    ("apply_then_predict", lambda: (_p().apply(1, (1.0, 0.0)), _p().predict())[1] is not None),
    ("bad_smooth", lambda: _raises(lambda: netpredict.Predict(0.0))),
    ("bad_smooth_high", lambda: _raises(lambda: netpredict.Predict(1.5))),
    ("bad_seq", lambda: _raises(lambda: _p().apply(-1, (0.0, 0.0)))),
    ("no_input", lambda: isinstance(_p().predict(), tuple)),
    ("reconcile", lambda: _p().apply(1, (0.0, 0.0)) and _p().reconcile((0.0, 0.0), 1) is not None),
    ("type", lambda: isinstance(_p().predict(), tuple)),
    ("det", lambda: (_p().apply(1, (1.0, 1.0)), _p().predict())[1] == (_p().apply(1, (1.0, 1.0)), _p().predict())[1]),
    ("replay", lambda: isinstance(_p().replay([(1, (0.0, 0.0))]), tuple)),
    ("bad_replay", lambda: _raises(lambda: _p().replay('x'))),
    ("nan", lambda: _raises(lambda: _p().apply(1, (float('nan'), 0.0)))),
    ("no_mut", lambda: _p().apply(1, (0.0, 0.0)) is not None),
    ("finite", lambda: all(math.isfinite(v) for v in (_p().apply(1, (0.5, 0.5)), _p().predict())[1])),
    ("seq_gap", lambda: isinstance((_p().apply(5, (0.0, 0.0)), _p().predict())[1], tuple)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
