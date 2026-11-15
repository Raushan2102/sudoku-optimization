"""Replications on famous hard 9x9 puzzles (data/hard_9.csv), as in Table 3 of
Aragón Artacho, Borwein & Tam (2014): how many random starts out of R solve each puzzle.

    python experiments/run_hard.py --methods dr --reps 100 --out results/final/hard_dr.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods import METHODS  # noqa: E402
from src.sudoku import parse_grid  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--sample", default="data/hard_9.csv")
p.add_argument("--methods", nargs="+", default=list(METHODS))
p.add_argument("--reps", type=int, default=100)
p.add_argument("--max-iter", type=int, default=None)
p.add_argument("--out", required=True)
a = p.parse_args()

df = pd.read_csv(a.sample, dtype={"puzzle": str, "solution": str})
rows = []
for name in a.methods:
    method = METHODS[name]
    kw = {} if a.max_iter is None else {"max_iter": a.max_iter}
    for _, r in df.iterrows():
        puzzle = parse_grid(r["puzzle"])
        for k in range(a.reps):
            extra = {"seed": k} if name == "dr" else {"seed": k, "noise": 0.3}
            t0 = time.perf_counter()
            res = method(puzzle, **kw, **extra)
            rows.append({"method": name, "puzzle": r["band"], "rep": k, "solved": res.solved,
                         "iterations": res.iterations, "seconds": time.perf_counter() - t0})
        got = sum(x["solved"] for x in rows if x["method"] == name and x["puzzle"] == r["band"])
        print(f"{name:4s} {r['band']:24s} {got}/{a.reps}", flush=True)
out = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.out, index=False)
