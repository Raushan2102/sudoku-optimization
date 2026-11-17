"""Douglas-Rachford variants on a famous hard puzzle, to probe the gap to Table 3 of
Aragón Artacho, Borwein & Tam (2014) on 'Nasty' (see results/final/README.md).

    python experiments/nasty_dr_variants.py --tie first
    python experiments/nasty_dr_variants.py --givens full --puzzle "AI Escargot"

Every run uses the paper's order, start x0 = (y, ..., y) and at most 10,000 iterations.

--tie     how the one-hot projection picks among tied maxima (within 1e-6):
            raw    np.argmax on the raw floats (what src/methods.py does)
            first  lowest index        last  highest index        random  uniformly at random
--givens  how the clues are handled:
            partial  5 sets, clue projection sets the given digit's entry to 1 (src/methods.py)
            full     5 sets, clue projection resets the whole clue cell (1 for the given digit, 0 else)
            four     4 sets, clues fixed inside the cell projection

With a deterministic tie rule (first, last) a run that revisits a state exactly is stopped and
counted as a cycle: it can never solve. raw is deterministic too, but its float state drifts
in the last bits, so exact repeats are not detected.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.methods import _is_solved_grid, from_groups, project_givens, to_groups  # noqa: E402
from src.sudoku import parse_grid  # noqa: E402

TOL = 1e-6

p = argparse.ArgumentParser()
p.add_argument("--tie", choices=["raw", "first", "last", "random"], default="raw")
p.add_argument("--givens", choices=["partial", "full", "four"], default="partial")
p.add_argument("--puzzle", default="Nasty", help="substring of the band column in data/hard_9.csv")
p.add_argument("--seeds", type=int, default=100, help="seeds 0 .. seeds-1")
p.add_argument("--max-iter", type=int, default=10000)
a = p.parse_args()

hard = pd.read_csv(ROOT / "data" / "hard_9.csv", dtype=str)
puzzle = parse_grid(hard[hard.band.str.contains(a.puzzle, regex=False)].iloc[0].puzzle)
clue_r, clue_c = np.nonzero(puzzle)
clue_d = puzzle[clue_r, clue_c] - 1

if a.givens == "four":
    SETS = ["row", "col", "box", "cell_clamped"]
else:
    SETS = ["row", "col", "box", "givens", "cell"]


def onehot(x, kind, rng, stats):
    G = to_groups(x, kind)
    if a.tie == "raw":
        idx = G.argmax(axis=1)
    else:
        ismax = G >= G.max(axis=1, keepdims=True) - TOL
        stats["groups"] += G.shape[0]
        stats["tied"] += int((ismax.sum(axis=1) > 1).sum())
        if a.tie == "first":
            idx = ismax.argmax(axis=1)
        elif a.tie == "last":
            idx = G.shape[1] - 1 - ismax[:, ::-1].argmax(axis=1)
        else:
            idx = (rng.random(G.shape) * ismax).argmax(axis=1)
    H = np.zeros_like(G)
    H[np.arange(G.shape[0]), idx] = 1.0
    return from_groups(H, kind)


def set_clue_cells(x):
    y = x.copy()
    y[clue_r, clue_c, :] = 0.0
    y[clue_r, clue_c, clue_d] = 1.0
    return y


def project(x, kind, rng, stats):
    if kind == "givens":
        return set_clue_cells(x) if a.givens == "full" else project_givens(x, puzzle)
    if kind == "cell_clamped":
        return set_clue_cells(onehot(x, "cell", rng, stats))
    return onehot(x, kind, rng, stats)


def run(seed):
    y = np.random.default_rng(seed).random((9, 9, 9))
    x = np.stack([y] * len(SETS))
    tie_rng = np.random.default_rng(10_000 + seed)
    stats = {"groups": 0, "tied": 0}
    seen = {}
    deterministic = a.tie in ("first", "last")
    for t in range(1, a.max_iter + 1):
        d = x.mean(axis=0)
        r = 2 * d[None] - x
        pc = np.stack([project(r[i], k, tie_rng, stats) for i, k in enumerate(SETS)])
        x = x + pc - d[None]
        if _is_solved_grid(x.mean(axis=0).argmax(axis=2) + 1, puzzle):
            return "solved", t, None, stats
        if deterministic:
            key = np.rint(x * 5).astype(np.int32).tobytes()   # iterates are multiples of 0.2
            if key in seen:
                return "cycle", t, t - seen[key], stats
            seen[key] = t
    return "cap", a.max_iter, None, stats


rows = []
for s in range(a.seeds):
    outcome, it, period, st = run(s)
    rows.append({"seed": s, "outcome": outcome, "iterations": it, "period": period,
                 "tie_share": st["tied"] / st["groups"] if st["groups"] else np.nan})
df = pd.DataFrame(rows)
solved = df[df.outcome == "solved"]
cyc = df[df.outcome == "cycle"]
print(f"{a.puzzle} tie={a.tie} givens={a.givens}: solved {len(solved)}/{len(df)}  "
      f"outcomes {df.outcome.value_counts().to_dict()}")
if len(solved):
    print(f"  median iterations when solved: {int(solved.iterations.median())}")
if len(cyc):
    print(f"  cycles: median start {cyc.iterations.median():.0f}, median period {cyc.period.median():.0f}")
if df.tie_share.notna().any():
    print(f"  share of group projections with a tie: {100 * df.tie_share.median():.2f} %")
