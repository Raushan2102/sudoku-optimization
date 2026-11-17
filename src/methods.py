"""The four methods compared in the project (9x9 only). All share one signature:

    method(puzzle, max_iter=..., **params) -> Result

`puzzle` is a 9x9 int array (0 = empty). Every method keeps clue cells fixed,
returns its final weights and a history of constraint_violation() per iteration,
and stops early once round_grid(x) is a valid solution. An optional callback(t, x) is called
after every iteration with the current weights (for Douglas-Rachford: the averaged candidate);
experiments/animate.py uses it to draw the grid as it fills in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    x: np.ndarray                                 # final weights, shape (N, N, N)
    grid: np.ndarray                              # round_grid(x)
    solved: bool                                  # is_solved(grid, puzzle)
    iterations: int                               # iterations actually run
    history: list = field(default_factory=list)   # constraint_violation per iteration


def lam_schedule(t: int, max_iter: int, lam_max: float = 0.6) -> float:
    """Weight of the 0/1 term at iteration t: starts at 0 (convex problem) and grows
    linearly to lam_max over the first 60% of the run. Worth experimenting with."""
    return lam_max * min(1.0, t / (0.6 * max_iter))


# ---------- shared pieces of the three gradient-based methods ----------

def _setup(puzzle: np.ndarray, seed=None, noise: float = 0.0):
    """Start point (uniform weights, clue cells one-hot, optional noise for restarts),
    the clue mask and a copy of the start used to keep clues fixed."""
    from src.objective import init_weights, project_simplex
    x, mask = init_weights(puzzle, noise=noise, seed=seed)
    if noise > 0:
        x0, _ = init_weights(puzzle)
        x = project_simplex(x)
        x[mask] = x0[mask]
    return x, mask, x.copy()


def _done(x: np.ndarray, puzzle: np.ndarray):
    from src.objective import round_grid
    from src.sudoku import is_solved
    grid = round_grid(x)
    return grid, is_solved(grid, puzzle)


def projected_gradient(puzzle: np.ndarray, max_iter: int = 1000, lam_max: float = 0.6,
                       sigma: float = 0.25, seed=None, noise: float = 0.0, callback=None,
                       **params) -> Result:
    """Projected gradient with Armijo backtracking along the projection arc
    (course notes ch. 6 and 10):
        x(a) = P_simplex(x - a * grad f(x)),  accept a when
        f(x(a)) <= f(x) + sigma * <grad f(x), x(a) - x>.
    The step found at one iteration is the starting guess (doubled) for the next.
    lam grows from 0 to lam_max (lam_schedule); clue cells never move."""
    from src.objective import constraint_violation, objective_and_grad, project_simplex
    x, mask, x0 = _setup(puzzle, seed, noise)
    step, history = 1.0, []
    for t in range(1, max_iter + 1):
        lam = lam_schedule(t, max_iter, lam_max)
        f, g = objective_and_grad(x, lam)
        g[mask] = 0.0
        a = 2 * step
        while True:
            xn = project_simplex(x - a * g)
            xn[mask] = x0[mask]
            fn, _ = objective_and_grad(xn, lam)
            if fn <= f + sigma * float((g * (xn - x)).sum()) or a < 1e-10:
                break
            a *= 0.5
        step, x = a, xn
        history.append(constraint_violation(x))
        if callback is not None:
            callback(t, x)
        grid, ok = _done(x, puzzle)
        if ok:
            return Result(x, grid, True, t, history)
    grid, ok = _done(x, puzzle)
    return Result(x, grid, ok, max_iter, history)


def accelerated_projected_gradient(puzzle: np.ndarray, max_iter: int = 1000, lam_max: float = 0.6,
                                   sigma: float = 0.25, restart: bool = True, seed=None,
                                   noise: float = 0.0, callback=None, **params) -> Result:
    """Nesterov's accelerated projected gradient (course notes ch. 8), FISTA form:
        y = x_k + ((t_k - 1) / t_{k+1}) (x_k - x_{k-1}),   x_{k+1} = projected Armijo step from y.
    restart=True resets the momentum whenever f goes up (the objective is nonconvex and
    changes with lam, so plain momentum can overshoot). If the extrapolated point is worse
    than x_k, the step is taken from x_k instead."""
    from src.objective import constraint_violation, objective_and_grad, project_simplex
    x, mask, x0 = _setup(puzzle, seed, noise)
    x_prev, tk, step, f_prev, history = x.copy(), 1.0, 1.0, np.inf, []
    for t in range(1, max_iter + 1):
        lam = lam_schedule(t, max_iter, lam_max)
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        y = project_simplex(x + ((tk - 1) / tn) * (x - x_prev))
        y[mask] = x0[mask]
        fy, gy = objective_and_grad(y, lam)
        fx, gx = objective_and_grad(x, lam)
        if fy > fx:                               # bad extrapolation: step from x instead
            y, fy, gy = x, fx, gx
        gy = gy.copy()
        gy[mask] = 0.0
        a = 2 * step
        while True:
            xn = project_simplex(y - a * gy)
            xn[mask] = x0[mask]
            fn, _ = objective_and_grad(xn, lam)
            if fn <= fy + sigma * float((gy * (xn - y)).sum()) or a < 1e-10:
                break
            a *= 0.5
        step = a
        if restart and fn > f_prev:
            tn = 1.0                              # momentum reset
        x_prev, x, tk, f_prev = x, xn, tn, fn
        history.append(constraint_violation(x))
        if callback is not None:
            callback(t, x)
        grid, ok = _done(x, puzzle)
        if ok:
            return Result(x, grid, True, t, history)
    grid, ok = _done(x, puzzle)
    return Result(x, grid, ok, max_iter, history)


def frank_wolfe(puzzle: np.ndarray, max_iter: int = 1000, lam_max: float = 0.6,
                step_rule: str = "armijo", sigma: float = 0.25, seed=None,
                noise: float = 0.0, callback=None, **params) -> Result:
    """Frank-Wolfe / conditional gradient (course notes ch. 10). The linear subproblem
    min_s <grad f(x), s> over a product of simplices has a closed form: in every cell, all
    weight on the digit with the smallest partial derivative. Then x <- x + gamma (s - x),
        step_rule="armijo":  gamma by backtracking from 1 on f(x + gamma d) <= f + sigma gamma <g, d>
        step_rule="classic": gamma = 2 / (t + 2).
    No projection is needed: every iterate is a convex combination of feasible points."""
    from src.objective import constraint_violation, objective_and_grad
    x, mask, x0 = _setup(puzzle, seed, noise)
    r, c = np.indices((9, 9))
    history = []
    for t in range(1, max_iter + 1):
        lam = lam_schedule(t, max_iter, lam_max)
        f, g = objective_and_grad(x, lam)
        s_ = np.zeros_like(x)
        s_[r, c, g.argmin(axis=2)] = 1.0
        s_[mask] = x0[mask]
        d = s_ - x
        if step_rule == "classic":
            gamma = 2.0 / (t + 2)
        elif step_rule == "armijo":
            slope = float((g * d).sum())
            gamma = 1.0
            while gamma > 1e-6:
                fn, _ = objective_and_grad(x + gamma * d, lam)
                if fn <= f + sigma * gamma * slope:
                    break
                gamma *= 0.5
        else:
            raise ValueError("step_rule must be 'armijo' or 'classic'")
        x = x + gamma * d
        history.append(constraint_violation(x))
        if callback is not None:
            callback(t, x)
        grid, ok = _done(x, puzzle)
        if ok:
            return Result(x, grid, True, t, history)
    grid, ok = _done(x, puzzle)
    return Result(x, grid, ok, max_iter, history)


# ---------- Douglas-Rachford (outside the course, for comparison) ----------
# Reference: F. J. Aragón Artacho, J. M. Borwein, M. K. Tam. Recent Results on Douglas-Rachford
# Methods for Combinatorial Optimization Problems. J. Optim. Theory Appl., 2014.
# arXiv:1305.2657, Section 6.2 (binary Sudoku formulation).
#
# All helpers below are for the fixed 9x9, 3x3-box grid.

KINDS = ("cell", "row", "col", "box")


def to_groups(x: np.ndarray, kind: str) -> np.ndarray:
    """(9, 9, 9) -> (81, 9): every row of the result is one group of the rule `kind`.
    x[r, c, k] is the weight of digit k+1 in cell (r, c).
      cell: the 9 digit weights of one cell
      row:  one digit k in one row r, over all columns
      col:  one digit k in one column c, over all rows
      box:  one digit k in one 3x3 box, over its 9 cells"""
    if kind == "cell":
        return x.reshape(81, 9)
    if kind == "row":                       # axes (r, k, c)
        return x.transpose(0, 2, 1).reshape(81, 9)
    if kind == "col":                       # axes (c, k, r)
        return x.transpose(1, 2, 0).reshape(81, 9)
    if kind == "box":                       # r = br*3 + i, c = bc*3 + j -> axes (br, bc, k, i, j)
        return x.reshape(3, 3, 3, 3, 9).transpose(0, 2, 4, 1, 3).reshape(81, 9)
    raise ValueError(f"unknown kind {kind!r}")


def from_groups(G: np.ndarray, kind: str, N: int = 9) -> np.ndarray:
    """Inverse of to_groups: (81, 9) -> (9, 9, 9)."""
    if kind == "cell":
        return G.reshape(9, 9, 9)
    if kind == "row":
        return G.reshape(9, 9, 9).transpose(0, 2, 1)
    if kind == "col":
        return G.reshape(9, 9, 9).transpose(2, 0, 1)
    if kind == "box":
        return G.reshape(3, 3, 9, 3, 3).transpose(0, 3, 1, 4, 2).reshape(9, 9, 9)
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
    G = to_groups(x, kind)
    H = np.zeros_like(G)
    H[np.arange(G.shape[0]), G.argmax(axis=1)] = 1.0
    y = from_groups(H, kind)
    return _clamp_clues(y, puzzle) if (kind == "cell" and clamp_clues) else y


def group_violation(x: np.ndarray) -> float:
    """1/2 * sum of (group sum - 1)^2 over rows, columns and boxes.
    Same quantity as objective.constraint_violation."""
    return 0.5 * float(sum(((to_groups(x, k).sum(axis=1) - 1) ** 2).sum()
                           for k in ("row", "col", "box")))


def _is_solved_grid(grid: np.ndarray, puzzle: np.ndarray) -> bool:
    """Valid solution check: every row, column and box holds 1..9 once, and the clues are kept."""
    onehot = np.zeros((9, 9, 9))
    r, c = np.indices((9, 9))
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
                     noise: float = 0.1, callback=None, **params) -> Result:
    """Douglas-Rachford in the product space (Aragón Artacho, Borwein & Tam 2014, Sec. 6.2).

    One copy of the grid per constraint set. With n_sets=5 (as in the paper) the sets are
    rows, columns, boxes, givens and cells; with n_sets=4 the givens are merged into the
    cell set (clue cells are fixed inside the cell projection).

    P_C: project every copy onto its own set; P_D: average the copies (the diagonal).
    order="paper":       x <- x + P_C(2 P_D x - x) - P_D x   (reflect w.r.t. D first),
                         candidate = P_D x
    order="rules_first": z <- z + P_D(2 P_C z - z) - P_C z,  candidate = P_D(P_C z)

    start="paper":   x0 = (y, ..., y) with y uniform in [0, 1]^(9x9x9), a point of D
    start="uniform": weights 1/9 plus `noise` * uniform noise, separately in every copy

    The candidate is rounded per cell (largest weight) and the run stops as soon as it is
    a valid solution; at a solution the paper's rounding to the nearest integer gives the
    same grid. The paper uses at most 10,000 iterations."""
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
        y = rng.random((9, 9, 9))
        x = np.stack([y] * m)
    elif start == "uniform":
        base = _clamp_clues(np.full((9, 9, 9), 1.0 / 9), puzzle)
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
        if callback is not None:
            callback(t, cand)
        grid = cand.argmax(axis=2) + 1
        if _is_solved_grid(grid, puzzle):
            return Result(cand, grid, True, t, history)
    grid = cand.argmax(axis=2) + 1
    return Result(cand, grid, _is_solved_grid(grid, puzzle), max_iter, history)


METHODS = {
    "pg": projected_gradient,
    "apg": accelerated_projected_gradient,
    "fw": frank_wolfe,
    "dr": douglas_rachford,
}
