"""Generate N x N Sudoku puzzles (N = n*n, e.g. 16 or 25) for the scaling study.

    python scripts/make_sample_nxn.py --n 4 --per-band 20 --out data/dev_sample_16.csv
    python scripts/make_sample_nxn.py --n 5 --per-band 20 --out data/dev_sample_25.csv

The Kaggle dataset has only 9x9 puzzles, so larger ones are generated: a random solved grid,
then a fraction of the cells blanked out. That fraction is the difficulty knob (`difficulty`
column, `band` column as a label). Puzzles are not checked for a unique solution; any grid
that keeps the clues and satisfies the rules counts as solved. Grids are stored as N*N
space-separated integers, 0 for an empty cell; parse_grid() reads them.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sudoku import grid_to_string, make_puzzle, random_solution  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, required=True, help="box size; the grid is N x N with N = n*n")
p.add_argument("--per-band", type=int, default=20)
p.add_argument("--blank", type=float, nargs="+", default=[0.4, 0.5, 0.6],
               help="fractions of empty cells, one band each")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
N = a.n * a.n
rows = []
for frac in a.blank:
    for _ in range(a.per_band):
        sol = random_solution(a.n, rng)
        puz = make_puzzle(sol, frac, rng)
        rows.append({"id": f"N{N}-{len(rows)}", "puzzle": grid_to_string(puz),
                     "solution": grid_to_string(sol), "clues": int((puz > 0).sum()),
                     "difficulty": frac, "band": f"{int(round(frac * 100))}% empty"})
df = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"wrote {len(df)} puzzles of size {N}x{N} to {a.out}")
print(df.groupby("band", sort=False).size().to_string())
