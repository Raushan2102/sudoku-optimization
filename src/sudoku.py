"""Loading puzzles from the Kaggle dataset and checking solutions.

Dataset: "3 million Sudoku puzzles with ratings" (Kaggle, CC0).
Columns: id, puzzle, solution, clues, difficulty.
`puzzle` is an 81-character string with '.' for empty cells.

Grids of any size N = n*n (9, 16, 25, ...) are supported. A 9x9 grid can be written in the
Kaggle format (81 characters); any grid can be written as N*N space-separated integers with
0 for an empty cell, which is the format the N x N generator uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- parsing ----------

def parse_grid(s: str) -> np.ndarray:
    """String -> N x N int array, 0 for an empty cell. Accepts the Kaggle 9x9 format
    (81 characters, '.' or '0' for empty) or N*N space-separated integers."""
    s = s.strip()
    if " " in s:
        vals = [int(v) for v in s.split()]
    else:
        s = s.replace(".", "0")
        if not s.isdigit():
            raise ValueError("expected digits and dots, or space-separated integers")
        vals = [int(ch) for ch in s]
    N = int(round(len(vals) ** 0.5))
    n = int(round(N ** 0.5))
    if N * N != len(vals) or n * n != N:
        raise ValueError(f"{len(vals)} values do not form an N x N grid with N = n*n")
    return np.array(vals, dtype=int).reshape(N, N)


def grid_to_string(grid: np.ndarray) -> str:
    """9x9 -> Kaggle format (81 characters, '.' for empty); larger grids -> space-separated."""
    g = np.asarray(grid)
    if g.shape == (9, 9):
        return "".join(str(int(v)) if v else "." for v in g.ravel())
    return " ".join(str(int(v)) for v in g.ravel())


def load_puzzles(path: str, n: int | None = None) -> pd.DataFrame:
    """Read the dataset CSV (or a sample of it). `n` limits the number of rows read."""
    df = pd.read_csv(path, nrows=n, dtype={"puzzle": str, "solution": str})
    missing = {"puzzle", "solution", "difficulty"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df


# ---------- checking ----------

def is_valid_solution(grid: np.ndarray) -> bool:
    """True if every row, column and n x n box contains 1..N exactly once (N = n*n)."""
    g = np.asarray(grid)
    N = g.shape[0]
    n = int(round(N ** 0.5))
    if g.ndim != 2 or g.shape != (N, N) or n * n != N:
        return False
    full = set(range(1, N + 1))
    for i in range(N):
        if set(g[i]) != full or set(g[:, i]) != full:
            return False
    for br in range(0, N, n):
        for bc in range(0, N, n):
            if set(g[br:br + n, bc:bc + n].ravel()) != full:
                return False
    return True


def matches_clues(grid: np.ndarray, puzzle: np.ndarray) -> bool:
    """True if the grid keeps every given clue of the puzzle."""
    mask = puzzle > 0
    return bool(np.all(grid[mask] == puzzle[mask]))


def is_solved(grid: np.ndarray, puzzle: np.ndarray) -> bool:
    return is_valid_solution(grid) and matches_clues(grid, puzzle)


# ---------- sampling ----------

DEFAULT_BANDS = [(0.0, 0.0), (0.1, 1.9), (2.0, 2.9), (3.0, 3.9), (4.0, 4.9), (5.0, 8.5)]


def band_label(lo: float, hi: float) -> str:
    return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"


def stratified_sample(df: pd.DataFrame, per_band: int, bands=DEFAULT_BANDS,
                      seed: int = 0) -> pd.DataFrame:
    """Take up to `per_band` puzzles from each difficulty band, so hard puzzles are not
    swamped by the ~43% of puzzles with difficulty 0. Adds a `band` column."""
    parts = []
    for lo, hi in bands:
        sel = df[(df["difficulty"] >= lo) & (df["difficulty"] <= hi)]
        take = sel.sample(n=min(per_band, len(sel)), random_state=seed)
        parts.append(take.assign(band=band_label(lo, hi)))
    return pd.concat(parts, ignore_index=True)


# ---------- N x N generator ----------

def random_solution(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random solved grid of side N = n*n: a pattern grid, then shuffled digits, rows within
    bands, bands, columns within stacks, stacks and possibly a transpose. All of these keep
    every rule satisfied."""
    N = n * n
    r = np.arange(N)[:, None]
    c = np.arange(N)[None, :]
    base = (n * (r % n) + r // n + c) % N                  # valid pattern, values 0..N-1
    digits = rng.permutation(N) + 1
    rows = np.concatenate([b * n + rng.permutation(n) for b in rng.permutation(n)])
    cols = np.concatenate([b * n + rng.permutation(n) for b in rng.permutation(n)])
    g = digits[base][rows][:, cols]
    return g.T.copy() if rng.random() < 0.5 else g


def make_puzzle(solution: np.ndarray, blank_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Blank out a random `blank_fraction` of the cells (0 = empty). The puzzle may have more
    than one solution; is_solved() accepts any grid that keeps the clues and the rules."""
    N = solution.shape[0]
    puzzle = solution.copy()
    k = int(round(blank_fraction * N * N))
    idx = rng.choice(N * N, size=k, replace=False)
    puzzle.ravel()[idx] = 0
    return puzzle
