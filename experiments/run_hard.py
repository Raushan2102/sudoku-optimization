"""Replications on famous hard 9x9 puzzles (data/hard_9.csv), as in Table 3 of
Aragón Artacho, Borwein & Tam (2014): how many random starts out of R solve each puzzle.

    python experiments/run_hard.py --methods dr --reps 100 --out results/final/hard_dr.csv

--jobs N runs N starts at once in separate processes (default: all CPU cores). Every start is
independent and seeded, so the results do not depend on N; only wall-clock time does. Per-run
`seconds` read higher when processes share the CPU: use --jobs 1 when the timings matter.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods import METHODS  # noqa: E402
from src.sudoku import parse_grid  # noqa: E402


def run_one(task):
    name, label, puzzle_str, k, max_iter = task
    kw = {} if max_iter is None else {"max_iter": max_iter}
    extra = {"seed": k} if name == "dr" else {"seed": k, "noise": 0.3}
    t0 = time.perf_counter()
    res = METHODS[name](parse_grid(puzzle_str), **kw, **extra)
    return {"method": name, "puzzle": label, "rep": k, "solved": res.solved,
            "iterations": res.iterations, "seconds": time.perf_counter() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="data/hard_9.csv")
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="starts run at once")
    a = p.parse_args()

    df = pd.read_csv(a.sample, dtype={"puzzle": str, "solution": str})
    rows = []
    with Pool(max(1, a.jobs)) as pool:
        for name in a.methods:
            for _, r in df.iterrows():
                tasks = [(name, r["band"], r["puzzle"], k, a.max_iter) for k in range(a.reps)]
                got_rows = pool.map(run_one, tasks, chunksize=1)   # keeps the start order
                rows += got_rows
                print(f"{name:4s} {r['band']:24s} {sum(x['solved'] for x in got_rows)}/{a.reps}", flush=True)
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
