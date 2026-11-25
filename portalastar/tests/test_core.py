import math

import pytest

import portalastar

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0], [0, 0]]
P = [((0, 0), (1, 1), 0.5)]

CASES = [
    ("path", lambda: len(portalastar.PortalAstar(P).search(G, (0, 0), (1, 1))) >= 2),
    ("unreachable", lambda: portalastar.PortalAstar(P).search([[0, 1], [1, 0]], (0, 0), (1, 1)) is None),
    ("cost", lambda: portalastar.PortalAstar(P).cost(portalastar.PortalAstar(P).search(G, (0, 0), (1, 1))) > 0.0),
    ("bad_portal", lambda: _raises(lambda: portalastar.PortalAstar([((0, 0), (0, 0), 1.0)]))),
    ("bad_cost", lambda: _raises(lambda: portalastar.PortalAstar([((0, 0), (1, 1), -1.0)]))),
    ("bad_grid", lambda: _raises(lambda: portalastar.PortalAstar(P).search([], (0, 0), (1, 1)))),
    ("bad_start", lambda: _raises(lambda: portalastar.PortalAstar(P).search(G, (9, 9), (1, 1)))),
    ("type", lambda: isinstance(portalastar.PortalAstar(P).search(G, (0, 0), (1, 1)), list)),
    ("det", lambda: portalastar.PortalAstar(P).search(G, (0, 0), (1, 1)) == portalastar.PortalAstar(P).search(G, (0, 0), (1, 1))),
    ("cost_type", lambda: isinstance(portalastar.PortalAstar(P).cost([(0, 0), (1, 1)]), float)),
    ("no_portal", lambda: isinstance(portalastar.PortalAstar([]).search(G, (0, 0), (1, 1)), list)),
    ("nan", lambda: _raises(lambda: portalastar.PortalAstar([((float('nan'), 0), (1, 1), 1.0)]))),
    ("no_mut", lambda: portalastar.PortalAstar(list(P)).search(G, (0, 0), (1, 1)) is not None),
    ("pair", lambda: all(len(c) == 2 for c in portalastar.PortalAstar(P).search(G, (0, 0), (1, 1)))),
    ("empty_portals", lambda: portalastar.PortalAstar([]).search(G, (0, 0), (0, 0)) == [(0, 0)]),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
