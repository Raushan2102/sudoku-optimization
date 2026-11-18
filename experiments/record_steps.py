"""Record what each method does at every iteration on one puzzle, for the step-by-step animations.

    python experiments/record_steps.py --id 260897
    python experiments/record_steps.py --id 868871

Runs each method once exactly as in experiment E1 (seed 0, max_iter=3000) through the `trace`
and `callback` hooks of src/methods.py, which observe the run without changing it. Writes
results/animation_data/<id>/:

  <method>.npz   per iteration t = 1..T (index 0 is the start):
                   x               weights after the iteration (DR: the average of the five copies)
                   row_v, col_v, box_v   1/2 sum over digits of (group sum - 1)^2 for each of the
                                   9 rows / columns / boxes: the constraint check of the objective
                   focus_*         the focus cell's 9 weights at each stage of the iteration
                 and the method's own internals:
                   pg   lam, step, halvings
                   apg  lam, beta, used_momentum, restart, step, halvings
                   fw   lam, gamma, halvings, target (digit chosen in every cell, 1..9)
                   dr   projections (the 0/1 copies for rows, columns, boxes and cells,
                        bit-packed), clue_copy (the clue copy, which sets each clue's digit to
                        1 and passes every other value through unchanged, so it is not 0/1)
  events.json    for each method, every placement / revision / retraction of a digit. A cell
                 counts as placed when one digit holds more than 85 % of its weight (for DR:
                 of the average of the five copies).
  summary.json   puzzle, solution, focus cell and per-method totals.

None of the methods selects cells, places digits one at a time or backtracks: each iteration
updates all 729 weights at once. Placements and revisions are read off the weights.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.methods import METHODS  # noqa: E402
from src.objective import box_sums, init_weights  # noqa: E402
from src.sudoku import parse_grid  # noqa: E402

PLACE = 0.85
ORDER = ["pg", "apg", "fw", "dr"]


def group_violation(x):
    rows = 0.5 * ((x.sum(axis=1) - 1) ** 2).sum(axis=1)
    cols = 0.5 * ((x.sum(axis=0) - 1) ** 2).sum(axis=1)
    boxes = 0.5 * ((box_sums(x) - 1) ** 2).sum(axis=2).reshape(9)
    return rows, cols, boxes


def placed_grid(x):
    """Digit 1..9 where one digit holds more than PLACE of the cell, else 0."""
    return np.where(x.max(axis=2) > PLACE, x.argmax(axis=2) + 1, 0)


def run(method, puzzle):
    xs, info = [], []
    res = METHODS[method](puzzle, max_iter=3000, seed=0,
                          callback=lambda t, x: xs.append(x.copy()),
                          trace=lambda t, i: info.append(i))
    if method == "dr":
        x0 = np.random.default_rng(0).random((9, 9, 9))        # the paper start y, in every copy
    else:
        x0, _ = init_weights(puzzle)
    return res, [x0] + xs, info


def events_from(xs, puzzle, solution):
    out, prev = [], placed_grid(xs[0])
    empty = puzzle == 0
    for t in range(1, len(xs)):
        cur = placed_grid(xs[t])
        for r, c in np.argwhere(empty & (cur != prev)):
            a, b = int(prev[r, c]), int(cur[r, c])
            kind = "place" if a == 0 else ("retract" if b == 0 else "revise")
            out.append({"t": t, "row": int(r), "col": int(c), "kind": kind, "from": a, "to": b,
                        "correct": bool(b == solution[r, c]) if b else None})
        prev = cur
    return out


def focus_cell(all_events, puzzle):
    """The empty cell with the most digit changes over all methods; ties go to the cell that
    settles last."""
    score = {}
    for evs in all_events.values():
        for e in evs:
            k = (e["row"], e["col"])
            n, last = score.get(k, (0, 0))
            score[k] = (n + 1, max(last, e["t"]))
    if not score:
        r, c = np.argwhere(puzzle == 0)[0]
        return int(r), int(c)
    return max(score, key=score.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--sample", default=str(ROOT / "data" / "dev_sample.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.sample, dtype={"puzzle": str, "solution": str, "id": str})
    row = df[df["id"] == str(a.id)].iloc[0]
    puzzle, solution = parse_grid(row["puzzle"]), parse_grid(row["solution"])
    out = ROOT / "results" / "animation_data" / str(a.id)
    out.mkdir(parents=True, exist_ok=True)

    runs = {m: run(m, puzzle) for m in ORDER}
    all_events = {m: events_from(xs, puzzle, solution) for m, (_, xs, _) in runs.items()}
    fr, fc = focus_cell(all_events, puzzle)

    summary = {"id": str(a.id), "difficulty": float(row["difficulty"]), "band": row.get("band"),
               "puzzle": row["puzzle"], "solution": row["solution"], "focus_cell": [fr, fc],
               "place_threshold": PLACE, "methods": {}}
    for m, (res, xs, info) in runs.items():
        X = np.stack(xs)                                        # (T+1, 9, 9, 9)
        rv, cv, bv = zip(*(group_violation(x) for x in X))
        data = {"x": X.astype(np.float16), "row_v": np.array(rv, np.float32),
                "col_v": np.array(cv, np.float32), "box_v": np.array(bv, np.float32),
                "solved": res.solved, "iterations": res.iterations}
        if m in ("pg", "apg", "fw"):
            for k in ("lam", "step", "halvings", "beta", "gamma"):
                if k in info[0]:
                    data[k] = np.array([i[k] for i in info], np.float32)
            for k in ("used_momentum", "restart"):
                if k in info[0]:
                    data[k] = np.array([i[k] for i in info], bool)
            data["focus_before"] = np.stack([i["x_before"][fr, fc] for i in info]).astype(np.float32)
            if m == "fw":
                data["target"] = np.stack([i["target"].argmax(axis=2) + 1 for i in info]).astype(np.int8)
                data["focus_target"] = np.stack([i["target"][fr, fc] for i in info]).astype(np.float32)
            else:
                data["focus_gradient_step"] = np.stack([i["gradient_step"][fr, fc]
                                                        for i in info]).astype(np.float32)
            if m == "apg":
                data["focus_y"] = np.stack([i["y"][fr, fc] for i in info]).astype(np.float32)
        else:
            P = np.stack([i["projections"] for i in info])               # (T, 5, 9, 9, 9)
            sets = list(info[0]["sets"])
            binary = [k for k in range(len(sets)) if sets[k] != "givens"]
            g = sets.index("givens")
            data["projections"] = np.packbits(P[:, binary] > 0.5, axis=-1)
            data["sets"] = np.array([sets[k] for k in binary])
            data["clue_copy"] = P[:, g].astype(np.float16)
            data["focus_projections"] = P[:, :, fr, fc].astype(np.float32)
            data["focus_average_before"] = np.stack([i["average_before"][fr, fc]
                                                     for i in info]).astype(np.float32)
        np.savez_compressed(out / f"{m}.npz", **data)
        evs = all_events[m]
        summary["methods"][m] = {
            "solved": bool(res.solved), "iterations": int(res.iterations),
            "placements": sum(e["kind"] == "place" for e in evs),
            "revisions": sum(e["kind"] == "revise" for e in evs),
            "retractions": sum(e["kind"] == "retract" for e in evs),
            "wrong_placements": sum(e["kind"] != "retract" and e["correct"] is False for e in evs),
        }
        print(f"{m}: solved={res.solved} iterations={res.iterations} {summary['methods'][m]}")
    (out / "events.json").write_text(json.dumps(all_events))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"focus cell r{fr + 1}c{fc + 1}; wrote {out}")


if __name__ == "__main__":
    main()
