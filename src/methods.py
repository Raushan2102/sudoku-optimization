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


def douglas_rachford(puzzle: np.ndarray, max_iter: int = 3000, **params) -> Result:
    """TODO (method outside the course, for comparison). Keep one copy of the grid per rule
    family (cells, rows, columns, boxes). Project each copy onto its rule in exact 0/1 form
    (keep the largest entry of each group, zero the rest), then average and reflect:
        z <- z + P_avg(2 * P_rules(z) - z) - P_rules(z)
    Read the candidate solution from the average of P_rules(z)."""
    raise NotImplementedError


METHODS = {
    "pg": projected_gradient,
    "apg": accelerated_projected_gradient,
    "fw": frank_wolfe,
    "dr": douglas_rachford,
}
