"""Check optimality, footprint clearance and map coordinates of the planner."""

import heapq
import math

from drone_vision_control.astar_navigation import (
    astar, cell_to_world, compress_path, inflated_grid, world_to_cell)
import numpy as np
import pytest


def cost(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def dijkstra(grid, start, goal):
    """Independent zero-heuristic reference for small maps."""
    q, seen = [(0., start)], set()
    while q:
        dist, (x, y) = heapq.heappop(q)
        if (x, y) in seen:
            continue
        if (x, y) == goal:
            return dist
        seen.add((x, y))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                a, b = x + dx, y + dy
                if not (dx or dy) or not (0 <= a < grid.shape[1] and 0 <= b < grid.shape[0]):
                    continue
                if grid[b, a] or (dx and dy and (grid[y, a] or grid[b, x])):
                    continue
                heapq.heappush(q, (dist + math.hypot(dx, dy), (a, b)))
    return math.inf


def test_open_grid_shortest_diagonal():
    path = astar(np.zeros((10, 10), dtype=bool), (0, 0), (6, 4))
    assert cost(path) == pytest.approx(4 * math.sqrt(2) + 2)


def test_no_diagonal_corner_cut():
    blocked = np.array([[False, True], [True, False]])
    assert astar(blocked, (0, 0), (1, 1)) == []


def test_blocked_or_outside_goals_have_no_route():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[3, 3] = True
    assert astar(blocked, (0, 0), (3, 3)) == []
    assert astar(blocked, (0, 0), (-1, 0)) == []


def test_unknown_cells_are_not_traversable():
    data = np.zeros((7, 7), dtype=int)
    data[:, 3] = -1
    blocked = inflated_grid(data, 0.1, 0.)
    assert astar(blocked, (1, 3), (5, 3)) == []


def test_inflation_rejects_narrow_passage():
    data = np.zeros((30, 30), dtype=int)
    data[:, 10] = data[:, 14] = 100
    blocked = inflated_grid(data, .1, .3)
    assert blocked[15, 12]


@pytest.mark.parametrize('seed', range(10))
def test_astar_matches_reference_optimal_cost(seed):
    rng = np.random.default_rng(seed)
    grid = rng.random((12, 12)) < .22
    grid[1, 1] = grid[10, 10] = False
    path = astar(grid, (1, 1), (10, 10))
    reference = dijkstra(grid, (1, 1), (10, 10))
    assert (cost(path) if path else math.inf) == pytest.approx(reference)


def test_rotated_grid_origin_round_trip():
    origin = (-3., 4., math.pi / 2)
    for cell in [(0, 0), (3, 5), (12, 9)]:
        point = cell_to_world(cell, origin, .05)
        assert world_to_cell(*point, origin, .05) == cell


def test_compression_preserves_turns_and_length():
    cells = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (4, 3)]
    simple = compress_path(cells)
    assert simple == [(0, 0), (2, 0), (4, 2), (4, 3)]
    assert cost(simple) == pytest.approx(cost(cells))
