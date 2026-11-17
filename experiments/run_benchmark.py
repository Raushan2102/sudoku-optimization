"""Run every implemented method on a sample of puzzles and save one row per (method, puzzle).

    python experiments/run_benchmark.py --sample data/dev_sample.csv --methods pg fw --out results/dev_run.csv

Add --history-out results/dev_hist.jsonl to also save each run's constraint violation per
iteration (used by experiments/plots.py for convergence curves).

--restarts K gives every puzzle up to K starts (seed 0, 1, ...; the gradient methods add
random noise to the start from the second one on). Iterations and seconds are summed over
the starts that were run, and the `starts` column says how many were needed.

Methods that are not written yet (NotImplementedError) are skipped with a message.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods import METHODS  # noqa: E402
from src.sudoku import parse_grid  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--sample", default="data/dev_sample.csv")
p.add_argument("--methods", nargs="+", default=list(METHODS))
p.add_argument("--max-iter", type=int, default=None, help="override each method's default")
p.add_argument("--out", default="results/run.csv")
p.add_argument("--history-out", default=None, help="optional .jsonl file for per-iteration histories")
p.add_argument("--restarts", type=int, default=1, help="up to this many starts per puzzle")
p.add_argument("--label", default=None, help="value for the `setting` column (default: method defaults)")
a = p.parse_args()

df = pd.read_csv(a.sample, dtype={"puzzle": str, "solution": str})
rows = []
hist_file = None
if a.history_out:
    Path(a.history_out).parent.mkdir(parents=True, exist_ok=True)
    hist_file = open(a.history_out, "w")
for name in a.methods:
    method = METHODS[name]
    kwargs = {} if a.max_iter is None else {"max_iter": a.max_iter}
    try:
        for _, r in df.iterrows():
            puzzle = parse_grid(r["puzzle"])
            t0 = time.perf_counter()
            total_it = 0
            for k in range(a.restarts):
                extra = {"seed": k}
                if name != "dr" and k > 0:
                    extra["noise"] = 0.3
                res = method(puzzle, **kwargs, **extra)
                total_it += res.iterations
                if res.solved:
                    break
            if hist_file:
                hist_file.write(json.dumps({"method": name, "id": str(r.get("id")),
                                            "N": int(puzzle.shape[0]), "band": r.get("band"),
                                            "history": [float(v) for v in res.history]}) + "\n")
            rows.append({
                "method": name, "N": int(puzzle.shape[0]), "id": r.get("id"), "band": r.get("band"),
                "difficulty": r["difficulty"], "solved": res.solved,
                "iterations": total_it, "seconds": time.perf_counter() - t0, "starts": k + 1,
                "max_iter": kwargs.get("max_iter"), "setting": a.label,
                "final_violation": res.history[-1] if res.history else None,
            })
    except NotImplementedError:
        print(f"skipping {name}: not implemented yet")
        continue
    done = [x for x in rows if x["method"] == name]
    print(f"{name}: solved {sum(x['solved'] for x in done)}/{len(done)}")

if hist_file:
    hist_file.close()
if rows:
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nsaved {len(out)} rows to {a.out}")
    print(out.pivot_table(index="band", columns="method", values="solved", aggfunc="mean", sort=False).round(2))
