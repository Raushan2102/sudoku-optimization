"""The four methods compared in the project. All share one signature:

    method(puzzle, max_iter=..., **params) -> Result

`puzzle` is a 9x9 int array (0 = empty). Every method keeps clue cells fixed,
returns its final weights and a history of constraint_violation() per iteration,
and stops early once round_grid(x) is a valid solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    x: np.ndarray                                 # final weights, shape (9, 9, 9)
    grid: np.ndarray                              # round_grid(x)
    solved: bool                                  # is_solved(grid, puzzle)
    iterations: int                               # iterations actually run
    history: list = field(default_factory=list)   # constraint_violation per iteration


def lam_schedule(t: int, max_iter: int, lam_max: float = 0.6) -> float:
    """Weight of the 0/1 term at iteration t: starts at 0 (convex problem) and grows
    linearly to lam_max over the first 60% of the run. Worth experimenting with."""
    return lam_max * min(1.0, t / (0.6 * max_iter))


def projected_gradient(puzzle: np.ndarray, max_iter: int = 600, **params) -> Result:
    """TODO. x <- project_simplex(x - alpha * grad f(x)), alpha by Armijo backtracking
    (course notes ch. 6 and 10). Keep clue cells fixed after every step."""
    raise NotImplementedError


def accelerated_projected_gradient(puzzle: np.ndarray, max_iter: int = 600, **params) -> Result:
    """TODO. Nesterov extrapolation y = x + beta * (x - x_prev) before the projected step
    (course notes ch. 8). Try it with and without restarting when f goes up."""
    raise NotImplementedError


def frank_wolfe(puzzle: np.ndarray, max_iter: int = 600, **params) -> Result:
    """TODO. Linear subproblem on each cell's simplex: put all weight on the digit with the
    smallest partial derivative (s). Then x <- x + gamma * (s - x), gamma by Armijo or
    2 / (t + 2) (course notes ch. 10)."""
    raise NotImplementedError


# ---------- Douglas-Rachford (outside the course, for comparison) ----------
# Reference: F. J. Aragón Artacho, J. M. Borwein, M. K. Tam (2014), Douglas-Rachford
# methods applied to combinatorial problems including Sudoku (see references/papers).
#
# All helpers below work for any side length N = n * n (9, 16, 25, ...).

KINDS = ("cell", "row", "col", "box")


def to_groups(x: np.ndarray, kind: str) -> np.ndarray:
    """(N, N, N) -> (N*N, N): every row of the result is one group of the rule `kind`.
    x[r, c, k] is the weight of digit k+1 in cell (r, c).
      cell: the N digit weights of one cell
      row:  one digit k in one row r, over all columns
      col:  one digit k in one column c, over all rows
      box:  one digit k in one n x n box, over its N cells"""
    N = x.shape[0]
    if kind == "cell":
        return x.reshape(N * N, N)
    if kind == "row":                       # axes (r, k, c)
        return x.transpose(0, 2, 1).reshape(N * N, N)
    if kind == "col":                       # axes (c, k, r)
        return x.transpose(1, 2, 0).reshape(N * N, N)
    if kind == "box":                       # r = br*n + i, c = bc*n + j -> axes (br, bc, k, i, j)
        n = int(round(N ** 0.5))
        return x.reshape(n, n, n, n, N).transpose(0, 2, 4, 1, 3).reshape(N * N, N)
    raise ValueError(f"unknown kind {kind!r}")


def from_groups(G: np.ndarray, kind: str, N: int) -> np.ndarray:
    """Inverse of to_groups: (N*N, N) -> (N, N, N)."""
    if kind == "cell":
        return G.reshape(N, N, N)
    if kind == "row":
        return G.reshape(N, N, N).transpose(0, 2, 1)
    if kind == "col":
        return G.reshape(N, N, N).transpose(2, 0, 1)
    if kind == "box":
        n = int(round(N ** 0.5))
        return G.reshape(n, n, N, n, n).transpose(0, 3, 1, 4, 2).reshape(N, N, N)
    raise ValueError(f"unknown kind {kind!r}")


def _clamp_clues(x: np.ndarray, puzzle: np.ndarray) -> np.ndarray:
    x = x.copy()
    rows, cols = np.nonzero(puzzle)
    x[rows, cols, :] = 0.0
    x[rows, cols, puzzle[rows, cols] - 1] = 1.0
    return x


def project_onehot(x: np.ndarray, kind: str, puzzle: np.ndarray) -> np.ndarray:
    """Projection onto the exact rule 'each group of `kind` holds exactly one 1, the rest 0':
    the largest entry of every group becomes 1. This set is NOT convex. For kind='cell'
    the clue cells are set to their given digit."""
    N = x.shape[0]
    G = to_groups(x, kind)
    H = np.zeros_like(G)
    H[np.arange(G.shape[0]), G.argmax(axis=1)] = 1.0
    y = from_groups(H, kind, N)
    return _clamp_clues(y, puzzle) if kind == "cell" else y


def group_violation(x: np.ndarray) -> float:
    """1/2 * sum of (group sum - 1)^2 over rows, columns and boxes; any N.
    Same quantity as objective.constraint_violation for 9x9."""
    return 0.5 * float(sum(((to_groups(x, k).sum(axis=1) - 1) ** 2).sum()
                           for k in ("row", "col", "box")))


def _solved_any_n(grid: np.ndarray, puzzle: np.ndarray) -> bool:
    """Valid solution check for any N: every row, column and box holds 1..N once,
    and the clues are kept."""
    N = grid.shape[0]
    onehot = np.zeros((N, N, N))
    r, c = np.indices((N, N))
    onehot[r, c, grid - 1] = 1.0
    rules_ok = all((to_groups(onehot, k).sum(axis=1) == 1).all() for k in ("row", "col", "box"))
    clues_ok = bool(np.all(grid[puzzle > 0] == puzzle[puzzle > 0]))
    return rules_ok and clues_ok


def douglas_rachford(puzzle: np.ndarray, max_iter: int = 3000, seed: int = 0,
                     noise: float = 0.1, **params) -> Result:
    """Douglas-Rachford in the product space. One copy z_i of the grid per rule family
    (cells, rows, columns, boxes).
        P_A: project each copy onto its own exact 0/1 rule  -> p_i
        P_B: average of the copies ("all copies equal")
        z_i <- z_i + P_B(2p - z) - p_i
    The candidate solution is the average of the p_i, rounded per cell. Stops as soon
    as the candidate is a valid solution. Works for any N = n*n."""
    N = puzzle.shape[0]
    rng = np.random.default_rng(seed)
    start = _clamp_clues(np.full((N, N, N), 1.0 / N), puzzle)
    # small random noise: with equal weights argmax would always pick digit 1
    z = np.stack([start + noise * rng.random(start.shape) for _ in KINDS])
    history = []
    cand = start
    for t in range(1, max_iter + 1):
        p = np.stack([project_onehot(z[i], kind, puzzle) for i, kind in enumerate(KINDS)])
        avg = (2 * p - z).mean(axis=0)          # P_B of the reflections
        z = z + avg[None] - p
        cand = p.mean(axis=0)
        history.append(group_violation(cand))
        grid = cand.argmax(axis=2) + 1
        if _solved_any_n(grid, puzzle):
            return Result(cand, grid, True, t, history)
    grid = cand.argmax(axis=2) + 1
    return Result(cand, grid, _solved_any_n(grid, puzzle), max_iter, history)


METHODS = {
    "pg": projected_gradient,
    "apg": accelerated_projected_gradient,
    "fw": frank_wolfe,
    "dr": douglas_rachford,
}
