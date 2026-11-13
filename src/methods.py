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
# Reference: F. J. Aragón Artacho, J. M. Borwein, M. K. Tam. Recent Results on Douglas-Rachford
# Methods for Combinatorial Optimization Problems. J. Optim. Theory Appl., 2014.
# arXiv:1305.2657, Section 6.2 (binary Sudoku formulation).
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


def project_onehot(x: np.ndarray, kind: str, puzzle: np.ndarray,
                   clamp_clues: bool = True) -> np.ndarray:
    """Projection onto the exact rule 'each group of `kind` holds exactly one 1, the rest 0':
    the largest entry of every group becomes 1 (Remark 5.1 of the paper; ties are broken by
    taking the first maximum). This set is NOT convex. For kind='cell' with clamp_clues=True
    the clue cells are set to their given digit (the 4-set formulation)."""
    N = x.shape[0]
    G = to_groups(x, kind)
    H = np.zeros_like(G)
    H[np.arange(G.shape[0]), G.argmax(axis=1)] = 1.0
    y = from_groups(H, kind, N)
    return _clamp_clues(y, puzzle) if (kind == "cell" and clamp_clues) else y


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


def project_givens(x: np.ndarray, puzzle: np.ndarray) -> np.ndarray:
    """Projection onto the givens set of the paper (C4): the entry of every given digit is
    set to 1, all other entries stay as they are."""
    y = x.copy()
    rows, cols = np.nonzero(puzzle)
    y[rows, cols, puzzle[rows, cols] - 1] = 1.0
    return y


def douglas_rachford(puzzle: np.ndarray, max_iter: int = 10000, seed: int = 0,
                     n_sets: int = 5, order: str = "paper", start: str = "paper",
                     noise: float = 0.1, **params) -> Result:
    """Douglas-Rachford in the product space (Aragón Artacho, Borwein & Tam 2014, Sec. 6.2).

    One copy of the grid per constraint set. With n_sets=5 (as in the paper) the sets are
    rows, columns, boxes, givens and cells; with n_sets=4 the givens are merged into the
    cell set (clue cells are fixed inside the cell projection).

    P_C: project every copy onto its own set; P_D: average the copies (the diagonal).
    order="paper":       x <- x + P_C(2 P_D x - x) - P_D x   (reflect w.r.t. D first),
                         candidate = P_D x
    order="rules_first": z <- z + P_D(2 P_C z - z) - P_C z,  candidate = P_D(P_C z)

    start="paper":   x0 = (y, ..., y) with y uniform in [0, 1]^(N x N x N), a point of D
    start="uniform": weights 1/N plus `noise` * uniform noise, separately in every copy

    The candidate is rounded per cell (largest weight) and the run stops as soon as it is
    a valid solution; at a solution the paper's rounding to the nearest integer gives the
    same grid. The paper uses at most 10,000 iterations. Works for any N = n*n."""
    N = puzzle.shape[0]
    rng = np.random.default_rng(seed)
    if n_sets == 5:
        sets = [("row", False), ("col", False), ("box", False), ("givens", None), ("cell", False)]
    elif n_sets == 4:
        sets = [("row", False), ("col", False), ("box", False), ("cell", True)]
    else:
        raise ValueError("n_sets must be 4 or 5")

    def proj(xi, kind, clamp):
        if kind == "givens":
            return project_givens(xi, puzzle)
        return project_onehot(xi, kind, puzzle, clamp_clues=clamp)

    m = len(sets)
    if start == "paper":
        y = rng.random((N, N, N))
        x = np.stack([y] * m)
    elif start == "uniform":
        base = _clamp_clues(np.full((N, N, N), 1.0 / N), puzzle)
        x = np.stack([base + noise * rng.random(base.shape) for _ in range(m)])
    else:
        raise ValueError("start must be 'paper' or 'uniform'")

    history = []
    cand = x.mean(axis=0)
    for t in range(1, max_iter + 1):
        if order == "paper":
            d = x.mean(axis=0)                                      # P_D x
            r = 2 * d[None] - x                                     # R_D x
            pc = np.stack([proj(r[i], k, c) for i, (k, c) in enumerate(sets)])
            x = x + pc - d[None]
            cand = x.mean(axis=0)                                   # P_D of the new iterate
        elif order == "rules_first":
            p = np.stack([proj(x[i], k, c) for i, (k, c) in enumerate(sets)])
            x = x + (2 * p - x).mean(axis=0)[None] - p
            cand = p.mean(axis=0)
        else:
            raise ValueError("order must be 'paper' or 'rules_first'")
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
