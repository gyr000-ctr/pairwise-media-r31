"""Portal-aware A* on a 2-D grid.

Public API
----------
``PortalAstar(portals)``
    ``portals`` is an iterable of ``(a, b, cost)`` triples, where ``a`` and
    ``b`` are ``(row, col)`` coordinate pairs and ``cost`` is the non-negative
    cost of travelling through the portal.  A portal is undirected: it can be
    entered from ``a`` and exited at ``b`` or the other way round, for the same
    cost.  ``portals`` may be any iterable (it is copied, never mutated).

``PortalAstar.search(grid, start, goal)``
    ``grid`` is a rectangular sequence of rows; ``0`` (or ``False``) marks a
    free cell and any other value marks a wall.  Movement is 4-connected
    (up/down/left/right) at a cost of ``1.0`` per step; portals may be used
    instead of walking.  Returns a list of ``(row, col)`` tuples describing a
    minimum-cost path from ``start`` to ``goal`` (``[start]`` when they are
    equal) or ``None`` when no path exists.

``PortalAstar.cost(path)``
    Returns the total cost of a path as a ``float``.

Cost model
----------
The graph's vertices are grid cells.  Its edges are the unit step between
orthogonally adjacent free cells and the undirected portal links (each portal
is added in both directions with the same cost).  ``search`` runs A* on that
graph, so the returned path's cost is the Dijkstra optimum.

Portals **shortcut** travel; they do not create passages.  A goal that cannot
be reached by walking over free cells is unreachable even if a portal happens
to link the start's region to the goal's region (this keeps ``None`` meaning
"no route through the level" rather than "no portal chain exists").  Once the
goal is walk-reachable, the search is an ordinary shortest-path search over
grid moves plus portal links, so portal costs are fully part of the
adjacency.

Repeated portal use can never improve a shortest path with non-negative edge
weights, so the returned path is simple: no cell - and hence no portal
endpoint - appears twice.  When every portal cost is at least the Manhattan
distance between its endpoints, the Manhattan distance is a consistent
heuristic and plain A* is used; otherwise the search falls back to a
Dijkstra-style zero heuristic to remain exact.

Everything used during a search is local to that call, so a single
``PortalAstar`` instance may be shared freely between threads.
"""

from __future__ import annotations

import heapq
import math
import numbers


__all__ = [
    "PortalAstar",
    "PortalAstarError",
    "InputError",
    "NonFiniteInputError",
    "PortalError",
    "PortalFormatError",
    "PortalCoordinateError",
    "PortalCostError",
    "PortalNonFiniteError",
    "PortalCoordinateNonFiniteError",
    "PortalCostNonFiniteError",
    "GridError",
    "GridNonFiniteError",
    "SearchInputError",
    "SearchNonFiniteError",
]


# ---------------------------------------------------------------------------
# Errors
#
# All of them derive from ``ValueError`` so callers can catch either the
# narrow class (to tell the failure modes apart) or ``ValueError`` generally.
# ---------------------------------------------------------------------------


class PortalAstarError(ValueError):
    """Base class for every error raised by this module."""


class InputError(PortalAstarError):
    """Base class for invalid arguments."""


class NonFiniteInputError(InputError):
    """Raised for ``NaN``/infinite values anywhere in the input.

    Attributes ``field`` (the offending argument), ``value`` (the offending
    value) and ``kind`` (``"nan"`` or ``"inf"``) allow programmatic handling.
    """

    def __init__(self, field: str, value: object) -> None:
        self.field = field
        self.value = value
        self.kind = "nan" if isinstance(value, float) and math.isnan(value) else "inf"
        super().__init__(f"{field} must be finite, got {value!r}")


class PortalError(InputError):
    """Base class for invalid portal specifications."""


class PortalFormatError(PortalError):
    """A portal entry is not a ``(a, b, cost)`` triple."""


class PortalCoordinateError(PortalError):
    """A portal endpoint is malformed or the two endpoints are identical."""


class PortalCostError(PortalError):
    """A portal cost is malformed or negative."""


class PortalNonFiniteError(PortalError, NonFiniteInputError):
    """A portal endpoint or cost is ``NaN``/infinite."""


class PortalCoordinateNonFiniteError(PortalCoordinateError, PortalNonFiniteError):
    """A portal endpoint is ``NaN``/infinite (also a coordinate error)."""


class PortalCostNonFiniteError(PortalCostError, PortalNonFiniteError):
    """A portal cost is ``NaN``/infinite (also a cost error)."""


class GridError(InputError):
    """The grid is empty, ragged, or contains a non-numeric cell."""


class GridNonFiniteError(GridError, NonFiniteInputError):
    """The grid contains a ``NaN``/infinite cell."""


class SearchInputError(InputError):
    """``start``/``goal`` (or a path passed to :meth:`cost`) is invalid."""


class SearchNonFiniteError(SearchInputError, NonFiniteInputError):
    """``start``/``goal`` (or a path coordinate) is ``NaN``/infinite."""


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

# Maps any byte to 1 (wall) unless it is 0 (free).
_WALL_TABLE = bytes([0] + [1] * 255)


def _is_sequence(value) -> bool:
    """True for indexable, sized containers such as list/tuple/ndarray."""
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _as_cell(value, field, non_finite_error, error, message):
    """Normalise a ``(row, col)`` pair into a tuple of two ints."""
    if not _is_sequence(value) or len(value) != 2:
        raise error(message)
    row, col = value[0], value[1]
    for component in (row, col):
        if isinstance(component, bool) or not isinstance(component, numbers.Real):
            raise error(message)
        if not math.isfinite(component):
            raise non_finite_error(field, component)
        if int(component) != component:
            raise error(message)
    return (int(row), int(col))


def _normalise_portals(portals):
    """Validate and normalise the portal specification.

    Returns a tuple of ``((row, col), (row, col), float)`` triples.
    """
    if portals is None:
        return ()
    if isinstance(portals, (str, bytes, bytearray)) or not hasattr(portals, "__iter__"):
        raise PortalFormatError(
            "portals must be an iterable of (a, b, cost) triples"
        )

    normalised = []
    for entry in portals:
        if not _is_sequence(entry) or len(entry) != 3:
            raise PortalFormatError(
                f"each portal must be an (a, b, cost) triple, got {entry!r}"
            )
        a_raw, b_raw, cost_raw = entry[0], entry[1], entry[2]
        a = _as_cell(
            a_raw,
            "portal endpoint",
            PortalCoordinateNonFiniteError,
            PortalCoordinateError,
            f"portal endpoints must be (row, col) pairs, got {a_raw!r}",
        )
        b = _as_cell(
            b_raw,
            "portal endpoint",
            PortalCoordinateNonFiniteError,
            PortalCoordinateError,
            f"portal endpoints must be (row, col) pairs, got {b_raw!r}",
        )
        if a == b:
            raise PortalCoordinateError(f"portal endpoints must differ, both are {a!r}")
        if isinstance(cost_raw, bool) or not isinstance(cost_raw, numbers.Real):
            raise PortalCostError(
                f"portal cost must be a finite number, got {cost_raw!r}"
            )
        if not math.isfinite(cost_raw):
            raise PortalCostNonFiniteError("portal cost", cost_raw)
        cost = float(cost_raw)
        if cost < 0.0:
            raise PortalCostError(
                f"portal cost must be non-negative, got {cost_raw!r}"
            )
        normalised.append((a, b, cost))
    return tuple(normalised)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _prepare_grid(grid):
    """Validate *grid* and flatten it into a padded wall bitmap.

    Returns ``(rows, cols, walls, width)``.  ``walls`` is a ``bytearray`` of
    size ``(rows + 2) * (cols + 2)``; ``1`` means wall, ``0`` means free.  The
    one-cell border is all wall, so neighbour lookups need no bounds checks.
    """
    if not _is_sequence(grid):
        raise GridError("grid must be a sequence of rows")
    n_rows = len(grid)
    if n_rows == 0:
        raise GridError("grid must not be empty")

    first = grid[0]
    if not _is_sequence(first):
        raise GridError("each grid row must be a sequence of cell values")
    n_cols = len(first)
    if n_cols == 0:
        raise GridError("grid rows must not be empty")

    width = n_cols + 2
    walls = bytearray(b"\x01" * ((n_rows + 2) * width))

    for r in range(n_rows):
        row = grid[r]
        if not _is_sequence(row):
            raise GridError("each grid row must be a sequence of cell values")
        if len(row) != n_cols:
            raise GridError(
                f"grid must be rectangular; row {r} has length {len(row)}, "
                f"expected {n_cols}"
            )
        base = (r + 1) * width + 1
        # Fast path: plain 0..255 int rows convert at C speed.
        try:
            values = bytes(row)
        except (TypeError, ValueError):
            values = None
        if values is not None:
            walls[base : base + n_cols] = values.translate(_WALL_TABLE)
            continue
        # Slow path: validate every cell individually.
        for c in range(n_cols):
            value = row[c]
            if isinstance(value, bool):
                passable = not value
            elif isinstance(value, numbers.Real):
                if not math.isfinite(value):
                    raise GridNonFiniteError(f"grid cell ({r}, {c})", value)
                passable = value == 0
            else:
                raise GridError(
                    f"grid cell ({r}, {c}) must be a number, got {value!r}"
                )
            walls[base + c] = 0 if passable else 1

    return n_rows, n_cols, walls, width


def _check_cell_in_grid(cell, n_rows, n_cols, field):
    row, col = cell
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise SearchInputError(
            f"{field} {cell!r} is outside the {n_rows}x{n_cols} grid"
        )
    return cell


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class PortalAstar:
    """A* shortest-path search over a grid augmented with symmetric portals."""

    __slots__ = ("_portals", "_portal_edges")

    def __init__(self, portals=None):
        self._portals = _normalise_portals(portals)

        # Directed adjacency used by cost(): both directions of every portal,
        # keeping the cheapest link when duplicates are supplied.  The search
        # builds the same mapping from this dict, so search() and cost() can
        # never disagree about an edge's cost.
        edges = {}
        for a, b, cost in self._portals:
            for src, dst in ((a, b), (b, a)):
                key = (src, dst)
                previous = edges.get(key)
                if previous is None or cost < previous:
                    edges[key] = cost
        self._portal_edges = edges

    # -- public API ---------------------------------------------------------

    def search(self, grid, start, goal):
        """Return a minimum-cost path from *start* to *goal*, or ``None``.

        Raises ``GridError``/``GridNonFiniteError`` for an invalid grid and
        ``SearchInputError``/``SearchNonFiniteError`` for an invalid start or
        goal.  A path is a list of ``(row, col)`` tuples; unreachable goals
        yield ``None``.
        """
        n_rows, n_cols, walls, width = _prepare_grid(grid)
        start = _as_cell(
            start,
            "start",
            SearchNonFiniteError,
            SearchInputError,
            f"start must be a (row, col) pair, got {start!r}",
        )
        goal = _as_cell(
            goal,
            "goal",
            SearchNonFiniteError,
            SearchInputError,
            f"goal must be a (row, col) pair, got {goal!r}",
        )
        _check_cell_in_grid(start, n_rows, n_cols, "start")
        _check_cell_in_grid(goal, n_rows, n_cols, "goal")

        start_idx = (start[0] + 1) * width + start[1] + 1
        goal_idx = (goal[0] + 1) * width + goal[1] + 1
        if walls[start_idx] or walls[goal_idx]:
            # A wall cell cannot be occupied, so the goal is unreachable.
            return None
        if start == goal:
            return [start]

        portal_adj, portal_safe = self._build_portal_adjacency(
            walls, n_rows, n_cols, width
        )

        if not portal_adj:
            # No usable portal: the grid-only search *is* the answer.
            return self._search(walls, width, goal, start_idx, goal_idx, None, True)

        # Reachability is decided by the walkable grid alone: portals are
        # shortcuts, not passages.  A grid-only A* doubles as the
        # reachability probe and stays cheap when the goal is reachable (the
        # Manhattan heuristic drives it straight there); when the goal is not
        # reachable it exhausts the start's component, which is what any
        # reachability test must do anyway.
        if self._search(walls, width, goal, start_idx, goal_idx, None, True) is None:
            return None

        # Portals are part of the optimum whenever they can be used.
        return self._search(
            walls, width, goal, start_idx, goal_idx, portal_adj, portal_safe
        )

    def cost(self, path):
        """Return the total cost of *path* as a ``float``.

        Consecutive coordinates must be either orthogonally adjacent (cost
        ``1.0``) or the two ends of a portal (the cheaper of the portal cost
        and a walk, when both apply).  Anything else raises
        ``SearchInputError``.
        """
        if path is None or not _is_sequence(path):
            raise SearchInputError("path must be a sequence of (row, col) pairs")

        total = 0.0
        previous = None
        for index in range(len(path)):
            item = path[index]
            cell = _as_cell(
                item,
                "path coordinate",
                SearchNonFiniteError,
                SearchInputError,
                f"path coordinates must be (row, col) pairs, got {item!r}",
            )
            if previous is not None:
                total += self._transition_cost(previous, cell)
            previous = cell
        return float(total)

    # -- internals ----------------------------------------------------------

    def _search(self, walls, width, goal, start_idx, goal_idx, portal_adj, use_heuristic):
        """A*/Dijkstra core shared by both phases.

        ``portal_adj`` is ``None`` for a grid-only search (portals ignored).
        The heap is ordered by ``(f, h, node)``: preferring a smaller
        ``h`` among equal ``f`` makes A* drive straight at the goal instead
        of expanding every node that ties on ``f`` (a whole open grid ties
        when the heuristic is exactly consistent).
        """
        goal_prow = goal[0] + 1
        goal_pcol = goal[1] + 1

        inf = math.inf
        dist = [inf] * len(walls)
        parent = [-1] * len(walls)
        closed = bytearray(len(walls))
        dist[start_idx] = 0.0

        heappush = heapq.heappush
        heappop = heapq.heappop

        if use_heuristic:
            prow, pcol = divmod(start_idx, width)
            h0 = float(abs(prow - goal_prow) + abs(pcol - goal_pcol))
        else:
            h0 = 0.0
        heap = [(h0, h0, start_idx)]

        while heap:
            f, h, u = heappop(heap)
            du = dist[u]
            # Lazy deletion: skip entries superseded by a shorter path.
            if du + h != f or closed[u]:
                continue
            closed[u] = 1
            if u == goal_idx:
                return self._reconstruct(parent, u, width)

            nd = du + 1.0

            if use_heuristic:
                prow, pcol = divmod(u, width)
                # Each neighbour's heuristic differs from the current h by 1.
                h_up = h - 1 if prow > goal_prow else h + 1
                h_down = h - 1 if prow < goal_prow else h + 1
                h_left = h - 1 if pcol > goal_pcol else h + 1
                h_right = h - 1 if pcol < goal_pcol else h + 1

            v = u - width
            if not walls[v] and not closed[v] and nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                if use_heuristic:
                    heappush(heap, (nd + h_up, h_up, v))
                else:
                    heappush(heap, (nd, 0.0, v))
            v = u + width
            if not walls[v] and not closed[v] and nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                if use_heuristic:
                    heappush(heap, (nd + h_down, h_down, v))
                else:
                    heappush(heap, (nd, 0.0, v))
            v = u - 1
            if not walls[v] and not closed[v] and nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                if use_heuristic:
                    heappush(heap, (nd + h_left, h_left, v))
                else:
                    heappush(heap, (nd, 0.0, v))
            v = u + 1
            if not walls[v] and not closed[v] and nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                if use_heuristic:
                    heappush(heap, (nd + h_right, h_right, v))
                else:
                    heappush(heap, (nd, 0.0, v))

            if portal_adj is not None:
                for v, step_cost in portal_adj.get(u, ()):
                    if closed[v]:
                        continue
                    nd_p = du + step_cost
                    if nd_p < dist[v]:
                        dist[v] = nd_p
                        parent[v] = u
                        if use_heuristic:
                            prow_v, pcol_v = divmod(v, width)
                            hv = float(
                                abs(prow_v - goal_prow) + abs(pcol_v - goal_pcol)
                            )
                            heappush(heap, (nd_p + hv, hv, v))
                        else:
                            heappush(heap, (nd_p, 0.0, v))

        return None

    def _build_portal_adjacency(self, walls, n_rows, n_cols, width):
        """Return ``(adjacency, heuristic_safe)`` for the usable portals.

        ``adjacency`` maps a padded flat index to a tuple of
        ``(padded flat index, cost)`` links.  Portals whose endpoints lie
        outside this grid or on a wall cannot be entered and are omitted.
        ``adjacency`` is empty when nothing is usable.  ``heuristic_safe``
        says whether the Manhattan distance stays a consistent heuristic for
        the retained links.
        """
        adjacency = {}
        heuristic_safe = True
        for (a, b), cost in self._portal_edges.items():
            ar, ac = a
            br, bc = b
            if not (0 <= ar < n_rows and 0 <= ac < n_cols):
                continue
            if not (0 <= br < n_rows and 0 <= bc < n_cols):
                continue
            a_idx = (ar + 1) * width + ac + 1
            b_idx = (br + 1) * width + bc + 1
            if walls[a_idx] or walls[b_idx]:
                continue
            adjacency.setdefault(a_idx, []).append((b_idx, cost))
            if cost < _manhattan(a, b):
                heuristic_safe = False
        return (
            {src: tuple(edges) for src, edges in adjacency.items()},
            heuristic_safe,
        )

    def _transition_cost(self, previous, cell):
        portal = self._portal_edges.get((previous, cell))
        if abs(previous[0] - cell[0]) + abs(previous[1] - cell[1]) == 1:
            return 1.0 if portal is None else min(portal, 1.0)
        if portal is not None:
            return portal
        raise SearchInputError(
            f"path contains a non-adjacent, non-portal step "
            f"{previous!r} -> {cell!r}"
        )

    @staticmethod
    def _reconstruct(parent, goal_idx, width):
        indices = []
        node = goal_idx
        while node != -1:
            indices.append(node)
            node = parent[node]
        indices.reverse()
        return [((idx // width) - 1, (idx % width) - 1) for idx in indices]
