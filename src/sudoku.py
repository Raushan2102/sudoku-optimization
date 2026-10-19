"""Loading puzzles from the Kaggle dataset and checking solutions.

Dataset: "3 million Sudoku puzzles with ratings" (Kaggle, CC0).
Columns: id, puzzle, solution, clues, difficulty.
`puzzle` is an 81-character string with '.' for empty cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- parsing ----------

def parse_grid(s: str) -> np.ndarray:
    """81-character string -> 9x9 int array, 0 for an empty cell."""
    s = s.strip().replace(".", "0")
    if len(s) != 81 or not s.isdigit():
        raise ValueError(f"expected 81 digits or dots, got {len(s)} characters")
    return np.array([int(ch) for ch in s], dtype=int).reshape(9, 9)


def grid_to_string(grid: np.ndarray) -> str:
    return "".join(str(int(v)) if v else "." for v in np.asarray(grid).ravel())


def load_puzzles(path: str, n: int | None = None) -> pd.DataFrame:
    """Read the dataset CSV (or a sample of it). `n` limits the number of rows read."""
    df = pd.read_csv(path, nrows=n, dtype={"puzzle": str, "solution": str})
    missing = {"puzzle", "solution", "difficulty"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df


# ---------- checking ----------

def is_valid_solution(grid: np.ndarray) -> bool:
    """True if every row, column and 3x3 box contains 1..9 exactly once."""
    g = np.asarray(grid)
    if g.shape != (9, 9):
        return False
    full = set(range(1, 10))
    for i in range(9):
        if set(g[i]) != full or set(g[:, i]) != full:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(g[br:br + 3, bc:bc + 3].ravel()) != full:
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
