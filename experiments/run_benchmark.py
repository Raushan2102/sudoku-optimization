"""Run every implemented method on a sample of puzzles and save one row per (method, puzzle).

    python experiments/run_benchmark.py --sample data/dev_sample.csv --methods pg fw --out results/dev_run.csv

Methods that are not written yet (NotImplementedError) are skipped with a message.
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
p.add_argument("--sample", default="data/dev_sample.csv")
p.add_argument("--methods", nargs="+", default=list(METHODS))
p.add_argument("--max-iter", type=int, default=None, help="override each method's default")
p.add_argument("--out", default="results/run.csv")
a = p.parse_args()

df = pd.read_csv(a.sample, dtype={"puzzle": str, "solution": str})
rows = []
for name in a.methods:
    method = METHODS[name]
    kwargs = {} if a.max_iter is None else {"max_iter": a.max_iter}
    try:
        for _, r in df.iterrows():
            t0 = time.perf_counter()
            res = method(parse_grid(r["puzzle"]), **kwargs)
            rows.append({
                "method": name, "id": r.get("id"), "band": r.get("band"),
                "difficulty": r["difficulty"], "solved": res.solved,
                "iterations": res.iterations, "seconds": time.perf_counter() - t0,
                "final_violation": res.history[-1] if res.history else None,
            })
    except NotImplementedError:
        print(f"skipping {name}: not implemented yet")
        continue
    done = [x for x in rows if x["method"] == name]
    print(f"{name}: solved {sum(x['solved'] for x in done)}/{len(done)}")

if rows:
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nsaved {len(out)} rows to {a.out}")
    print(out.pivot_table(index="band", columns="method", values="solved", aggfunc="mean", sort=False).round(2))
