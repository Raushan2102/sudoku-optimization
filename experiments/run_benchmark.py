"""Run every implemented method on a sample of puzzles and save one row per (method, puzzle).

    python experiments/run_benchmark.py --sample data/dev_sample.csv --methods pg fw --out results/dev_run.csv

Add --history-out results/dev_hist.jsonl to also save each run's constraint violation per
iteration (used by experiments/plots.py for convergence curves).

--restarts K gives every puzzle up to K starts (seed 0, 1, ...; the gradient methods add
random noise to the start from the second one on). Iterations and seconds are summed over
the starts that were run, and the `starts` column says how many were needed.

--jobs N runs N puzzles at once in separate processes (default: all CPU cores). Every run is
independent and seeded, so the results do not depend on N; only wall-clock time does. Processes
share the CPU, so the per-run `seconds` read higher (2-4x with 8 jobs on 4 cores): use --jobs 1
when the timings matter.

Methods that are not written yet (NotImplementedError) are skipped with a message.
"""
import argparse
import json
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
    """All starts of one method on one puzzle. Returns (row, history) or None if not written."""
    name, r, restarts, max_iter, label = task
    method = METHODS[name]
    kwargs = {} if max_iter is None else {"max_iter": max_iter}
    puzzle = parse_grid(r["puzzle"])
    t0 = time.perf_counter()
    total_it = 0
    try:
        for k in range(restarts):
            extra = {"seed": k}
            if name != "dr" and k > 0:
                extra["noise"] = 0.3
            res = method(puzzle, **kwargs, **extra)
            total_it += res.iterations
            if res.solved:
                break
    except NotImplementedError:
        return None
    row = {
        "method": name, "N": int(puzzle.shape[0]), "id": r.get("id"), "band": r.get("band"),
        "difficulty": r["difficulty"], "solved": res.solved,
        "iterations": total_it, "seconds": time.perf_counter() - t0, "starts": k + 1,
        "max_iter": kwargs.get("max_iter"), "setting": label,
        "final_violation": res.history[-1] if res.history else None,
    }
    hist = {"method": name, "id": str(r.get("id")), "N": int(puzzle.shape[0]), "band": r.get("band"),
            "history": [float(v) for v in res.history]}
    return row, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="data/dev_sample.csv")
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--max-iter", type=int, default=None, help="override each method's default")
    p.add_argument("--out", default="results/run.csv")
    p.add_argument("--history-out", default=None, help="optional .jsonl file for per-iteration histories")
    p.add_argument("--restarts", type=int, default=1, help="up to this many starts per puzzle")
    p.add_argument("--label", default=None, help="value for the `setting` column (default: method defaults)")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="puzzles run at once")
    a = p.parse_args()

    df = pd.read_csv(a.sample, dtype={"puzzle": str, "solution": str})
    records = df.to_dict("records")
    rows, hists = [], []
    with Pool(max(1, a.jobs)) as pool:
        for name in a.methods:
            tasks = [(name, r, a.restarts, a.max_iter, a.label) for r in records]
            results = pool.map(run_one, tasks, chunksize=1)      # keeps the puzzle order
            if any(x is None for x in results):
                print(f"skipping {name}: not implemented yet")
                continue
            rows += [row for row, _ in results]
            hists += [h for _, h in results]
            done = [x for x in rows if x["method"] == name]
            print(f"{name}: solved {sum(x['solved'] for x in done)}/{len(done)}", flush=True)

    if a.history_out:
        Path(a.history_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.history_out, "w") as f:
            for h in hists:
                f.write(json.dumps(h) + "\n")
    if rows:
        out = pd.DataFrame(rows)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"\nsaved {len(out)} rows to {a.out}")
        print(out.pivot_table(index="band", columns="method", values="solved", aggfunc="mean", sort=False).round(2))


if __name__ == "__main__":
    main()
