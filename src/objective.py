"""The continuous relaxation of Sudoku.

Variables: x has shape (N, N, N) for an N x N grid (N = 9, 16, 25, ...); x[r, c, k] is the weight
of digit k+1 in cell (r, c).
Each cell's weights x[r, c, :] lie on the probability simplex (nonnegative, sum to 1).

Objective (to be written by the team):
    f(x) = 1/2 * sum over rows, columns, boxes and digits of (group sum - 1)^2
           + lam * sum of x * (1 - x)
The first term is zero exactly when every row, column and box contains each digit once
(in the relaxed sense); the second term pushes weights toward 0 or 1.

Functions marked TODO are the ones the project is about. The rest is infrastructure.
"""
from __future__ import annotations

import numpy as np


# ---------- infrastructure (done) ----------

def init_weights(puzzle: np.ndarray, noise: float = 0.0, seed: int | None = None):
    """Uniform weights 1/N in empty cells, one-hot weights in clue cells (any N = n*n).
    Returns (x, mask) where mask[r, c] is True for clue cells (keep them fixed).
    `noise` > 0 adds a small random perturbation (for random restarts); call
    project_simplex afterwards to make the weights valid again."""
    N = puzzle.shape[0]
    x = np.full((N, N, N), 1.0 / N)
    if noise > 0:
        x = x + noise * np.random.default_rng(seed).random(x.shape)
    mask = puzzle > 0
    for r, c in np.argwhere(mask):
        x[r, c] = 0.0
        x[r, c, puzzle[r, c] - 1] = 1.0
    return x, mask


def fix_clues(x: np.ndarray, x_clues: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy the clue cells from x_clues into x (clues never move)."""
    x = x.copy()
    x[mask] = x_clues[mask]
    return x


def round_grid(x: np.ndarray) -> np.ndarray:
    """Each cell takes the digit with the largest weight -> N x N int grid."""
    return x.argmax(axis=2) + 1


def box_sums(x: np.ndarray) -> np.ndarray:
    """Sum of weights over each n x n box, per digit -> shape (n, n, N), with N = n*n."""
    N = x.shape[0]
    n = int(round(N ** 0.5))
    return x.reshape(n, n, n, n, N).sum(axis=(1, 3))


def constraint_violation(x: np.ndarray) -> float:
    """1/2 * sum of squared (group sum - 1) over rows, columns and boxes.
    Zero for a solved grid; useful for plots and stopping rules."""
    rows = x.sum(axis=1) - 1
    cols = x.sum(axis=0) - 1
    boxes = box_sums(x) - 1
    return 0.5 * float((rows ** 2).sum() + (cols ** 2).sum() + (boxes ** 2).sum())


def check_gradient(f_and_grad, x: np.ndarray, n_checks: int = 5, eps: float = 1e-6,
                   seed: int = 0) -> float:
    """Compare the analytic gradient with central finite differences at a few random
    coordinates. Returns the largest relative error (should be around 1e-6 or smaller)."""
    rng = np.random.default_rng(seed)
    _, g = f_and_grad(x)
    worst = 0.0
    for _ in range(n_checks):
        idx = tuple(rng.integers(0, x.shape[0], size=3))
        e = np.zeros_like(x); e[idx] = eps
        fd = (f_and_grad(x + e)[0] - f_and_grad(x - e)[0]) / (2 * eps)
        worst = max(worst, abs(fd - g[idx]) / max(1.0, abs(fd), abs(g[idx])))
    return worst


# ---------- TODO: the team writes these ----------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """TODO. Euclidean projection of every vector along the LAST axis onto the
    probability simplex {w : w >= 0, sum(w) = 1}.

    Works for any shape (..., 9), e.g. (9, 9, 9) for a whole grid.
    Hint: the sorting-based algorithm of Duchi, Shalev-Shwartz, Singer, Chandra (ICML 2008);
    the course notes cite the same paper for the l1-ball projection (ch. 11).
    """
    raise NotImplementedError


def objective_and_grad(x: np.ndarray, lam: float):
    """TODO. Return (f, grad) for the objective in the module docstring.
    grad has the same shape as x. Check it with check_gradient()."""
    raise NotImplementedError
