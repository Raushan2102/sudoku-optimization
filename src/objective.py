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


# ---------- core math ----------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of every vector along the LAST axis onto the probability simplex
    {w : w >= 0, sum(w) = 1}. Works for any shape (..., N), e.g. (N, N, N) for a whole grid.

    From the KKT conditions (course notes ch. 12) the projection has the form
        w_i = max(v_i - theta, 0)
    for one threshold theta per vector. Sort v in decreasing order (u), and let rho be the
    largest k with u_k - (u_1 + ... + u_k - 1) / k > 0; then theta = (u_1 + ... + u_rho - 1) / rho.
    Duchi, Shalev-Shwartz, Singer, Chandra (ICML 2008); O(N log N) per vector.

    Example: v = (0.5, 0.8, -0.1) -> u = (0.8, 0.5, -0.1), rho = 2, theta = 0.15,
             w = (0.35, 0.65, 0).
    """
    v = np.asarray(v, dtype=float)
    shape = v.shape
    V = v.reshape(-1, shape[-1])
    U = -np.sort(-V, axis=1)                        # decreasing
    css = np.cumsum(U, axis=1) - 1.0                # (u_1 + ... + u_k) - 1
    k = np.arange(1, shape[-1] + 1)
    positive = U - css / k > 0                      # True for k = 1..rho, then False
    rho = shape[-1] - np.argmax(positive[:, ::-1], axis=1)   # index of the last True, 1-based
    theta = css[np.arange(V.shape[0]), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0).reshape(shape)


def objective_and_grad(x: np.ndarray, lam: float):
    """Return (f, grad) for the objective in the module docstring, for any N = n*n:

        f(x) = 1/2 sum (row sums - 1)^2 + 1/2 sum (column sums - 1)^2
             + 1/2 sum (box sums - 1)^2 + lam * sum x (1 - x)

    where a row sum is sum_c x[r, c, k] for one row r and digit k (and likewise for columns
    and boxes). Every x[r, c, k] lies in exactly one row group, one column group and one box
    group, so
        df/dx[r, c, k] = R[r, k] + C[c, k] + B[box(r, c), k] + lam * (1 - 2 x[r, c, k]).
    grad has the same shape as x; check it with check_gradient()."""
    N = x.shape[0]
    n = int(round(N ** 0.5))
    R = x.sum(axis=1) - 1.0                         # (N rows, N digits)
    C = x.sum(axis=0) - 1.0                         # (N cols, N digits)
    B = box_sums(x) - 1.0                           # (n, n, N)
    f = 0.5 * ((R ** 2).sum() + (C ** 2).sum() + (B ** 2).sum()) + lam * (x * (1.0 - x)).sum()
    B_full = np.repeat(np.repeat(B, n, axis=0), n, axis=1)   # (N, N, N): box value per cell
    grad = R[:, None, :] + C[None, :, :] + B_full + lam * (1.0 - 2.0 * x)
    return float(f), grad
