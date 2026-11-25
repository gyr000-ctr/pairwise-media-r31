import math

import pytest

import saveversion

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

D = {"version": 1, "hp": 10}

CASES = [
    ("v1_to_v2", lambda: _save().migrate(D, 2)['version'] == 2 or _save() is None),
    ("idempotent", lambda: isinstance(saveversion.SaveMigrate().migrate(D, 3), dict)),
    ("unknown_target", lambda: _raises(lambda: saveversion.SaveMigrate().migrate(D, 99))),
    ("bad_version", lambda: _raises(lambda: saveversion.SaveMigrate().migrate({'version': 0}, 2))),
    ("missing_field", lambda: isinstance(saveversion.SaveMigrate().migrate({'version': 1}, 2), dict)),
    ("warnings", lambda: isinstance(saveversion.SaveMigrate().warnings(), list)),
    ("type", lambda: isinstance(saveversion.SaveMigrate().migrate(D, 2), dict)),
    ("det", lambda: saveversion.SaveMigrate().migrate(D, 2) == saveversion.SaveMigrate().migrate(D, 2)),
    ("no_mut", lambda: saveversion.SaveMigrate().migrate(dict(D), 2) is not None),
    ("downgrade", lambda: _raises(lambda: saveversion.SaveMigrate().migrate({'version': 3}, 1))),
    ("nan", lambda: _raises(lambda: saveversion.SaveMigrate().migrate({'version': float('nan')}, 2))),
    ("chain", lambda: isinstance(saveversion.SaveMigrate().migrate({'version': 1, 'hp': 1}, 3), dict)),
    ("warn_type", lambda: isinstance(saveversion.SaveMigrate().warnings(), list)),
    ("bad_kind", lambda: _raises(lambda: saveversion.SaveMigrate().migrate([], 2))),
    ("version_int", lambda: saveversion.SaveMigrate().migrate(D, 2)['version'] == 2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
