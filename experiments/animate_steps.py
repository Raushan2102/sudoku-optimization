"""Step-by-step GIF of one method on one puzzle, drawn from experiments/record_steps.py data.

    python experiments/record_steps.py --id 260897
    python experiments/animate_steps.py --id 260897 --method dr

Left: the whole grid. Small digits are each cell's weights (darker = larger); a large digit is a
placed cell (green if it matches the solution, red if not); a ring marks a cell whose digit
changed since the previous frame; rows, columns and boxes are tinted by their constraint
violation. Right: the focus cell's nine weights at every stage of the current iteration, the
method's internals, and the run's timelines. Writes results/figures/animation/steps_<id>_<method>.gif.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sudoku import parse_grid  # noqa: E402

NAMES = {"pg": "Projected gradient", "apg": "Accelerated projected gradient",
         "fw": "Frank–Wolfe", "dr": "Douglas–Rachford"}
ONE_ITERATION = {
    "pg": "Each iteration: step all 729 weights against the gradient of the violation, "
          "then project every cell back onto the simplex.",
    "apg": "Each iteration: extrapolate with momentum, step against the gradient there, "
           "then project every cell back onto the simplex.",
    "fw": "Each iteration: in every cell pick the digit the gradient favours most, "
          "then move part of the way toward that choice.",
    "dr": "Each iteration: reflect five copies of the grid, project each onto one rule "
          "(rows, columns, boxes, clues, cells), then average.",
}
COLOR = {"pg": (42, 120, 214), "apg": (235, 104, 52), "fw": (27, 175, 122), "dr": (179, 122, 0)}
BG, INK, MUTED, GRID = (255, 250, 241), (34, 34, 59), (110, 106, 98), (206, 200, 186)
OK, BAD, CHANGE = (26, 140, 90), (214, 69, 65), (225, 140, 20)
VIOL = (214, 69, 65)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
W, H = 1280, 720


def font(size, bold=False, mono=False):
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


F = {k: font(*v) for k, v in {"title": (28, True), "sub": (15,), "head": (16, True), "txt": (15,),
                              "small": (13,), "mono": (20, False, True), "tiny": (11, False, True)}.items()}


def mix(c, a):
    return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))


def load(pid, method):
    base = ROOT / "results" / "animation_data" / str(pid)
    summary = json.loads((base / "summary.json").read_text())
    events = json.loads((base / "events.json").read_text())[method]
    data = dict(np.load(base / f"{method}.npz"))
    return summary, events, data


def draw_grid(d, x0, y0, size, x, puzzle, solution, rv, cv, bv, focus, changed, accent):
    cell = size / 9
    big, small = font(int(cell * 0.56), bold=True), font(int(cell * 0.25), mono=True)
    for b in range(9):                                       # box violation tint
        a = min(1.0, bv[b] / 2.0) * 0.22
        if a > 0.01:
            br, bc = divmod(b, 3)
            d.rectangle([x0 + bc * 3 * cell, y0 + br * 3 * cell, x0 + (bc + 1) * 3 * cell,
                         y0 + (br + 1) * 3 * cell], fill=mix(VIOL, a))
    for i in range(9):                                       # row / column violation strips
        d.rectangle([x0 - 16, y0 + i * cell + 2, x0 - 6, y0 + (i + 1) * cell - 2],
                    fill=mix(VIOL, min(1.0, rv[i] / 2.0)), outline=GRID)
        d.rectangle([x0 + i * cell + 2, y0 - 16, x0 + (i + 1) * cell - 2, y0 - 6],
                    fill=mix(VIOL, min(1.0, cv[i] / 2.0)), outline=GRID)
    for r in range(9):
        for c in range(9):
            cx, cy = x0 + c * cell, y0 + r * cell
            if puzzle[r, c]:
                d.text((cx + cell / 2, cy + cell / 2), str(puzzle[r, c]), font=big, fill=INK, anchor="mm")
                continue
            w = x[r, c].astype(float)
            k = int(w.argmax())
            if w[k] > 0.85:
                d.text((cx + cell / 2, cy + cell / 2), str(k + 1), font=big,
                       fill=OK if k + 1 == solution[r, c] else BAD, anchor="mm")
            else:
                for dg in range(9):
                    a = float(np.clip(w[dg] / 0.5, 0, 1))
                    if a < 0.06:
                        continue
                    rr, cc = divmod(dg, 3)
                    d.text((cx + (cc + 0.5) * cell / 3, cy + (rr + 0.5) * cell / 3), str(dg + 1),
                           font=small, fill=mix(INK, a), anchor="mm")
            if (r, c) in changed:
                d.ellipse([cx + 3, cy + 3, cx + cell - 3, cy + cell - 3], outline=changed[(r, c)], width=3)
    for i in range(10):
        wd = 3 if i % 3 == 0 else 1
        col = INK if i % 3 == 0 else GRID
        d.line([(x0, y0 + i * cell), (x0 + size, y0 + i * cell)], fill=col, width=wd)
        d.line([(x0 + i * cell, y0), (x0 + i * cell, y0 + size)], fill=col, width=wd)
    fr, fc = focus
    d.rectangle([x0 + fc * cell, y0 + fr * cell, x0 + (fc + 1) * cell, y0 + (fr + 1) * cell],
                outline=accent, width=4)


def bars(d, x0, y0, w, h, vals, title, accent, sol_digit, lo=0.0, hi=1.0, note=None):
    """Nine bars for one cell's weights; the solution digit's label is green."""
    d.text((x0, y0), title, font=F["small"], fill=INK)
    top, bot = y0 + 20, y0 + 20 + h
    def ypos(v):
        return bot - (np.clip(v, lo, hi) - lo) / (hi - lo) * h
    for ref in (0.0, 1.0):
        if lo <= ref <= hi:
            d.line([(x0, ypos(ref)), (x0 + w, ypos(ref))], fill=GRID, width=1)
    bw = w / 9
    for k in range(9):
        v = float(vals[k])
        y_v, y_0 = ypos(v), ypos(0.0)
        if abs(y_v - y_0) >= 1:
            d.rectangle([x0 + k * bw + 2, min(y_v, y_0), x0 + (k + 1) * bw - 2, max(y_v, y_0)], fill=accent)
        if v > hi or v < lo:                                 # clipped: mark the cut end
            yy = top if v > hi else bot
            d.line([(x0 + k * bw + 2, yy), (x0 + (k + 1) * bw - 2, yy)], fill=INK, width=2)
        d.text((x0 + (k + 0.5) * bw, bot + 9), str(k + 1), font=F["tiny"],
               fill=OK if k + 1 == sol_digit else MUTED, anchor="mm")
    if note:
        d.text((x0, bot + 20), note, font=F["tiny"], fill=MUTED)


def timeline(d, x0, y0, w, h, ts, series, t_now, title, log_y, ymax=None, labels=None):
    d.text((x0, y0), title, font=F["small"], fill=INK)
    top, bot = y0 + 20, y0 + 20 + h
    T = ts[-1]
    log_x = T > 600
    def xpos(t):
        if log_x:
            return x0 + np.log1p(t) / np.log1p(T) * w
        return x0 + t / T * w
    allv = np.concatenate([s for s, _ in series])
    if log_y:
        lo, hi = 1e-2, max(1.0, float(allv.max()))
        def ypos(v):
            return bot - (np.log10(max(v, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * h
    else:
        hi = ymax or max(1.0, float(allv.max()))
        def ypos(v):
            return bot - v / hi * h
    d.line([(x0, bot), (x0 + w, bot)], fill=GRID, width=1)
    d.line([(x0, top), (x0, bot)], fill=GRID, width=1)
    for vals, col in series:
        pts = [(xpos(t), ypos(v)) for t, v in zip(ts, vals)]
        d.line(pts, fill=mix(col, 0.35), width=2)
        k = int(np.searchsorted(ts, t_now, side="right"))
        if k > 1:
            d.line(pts[:k], fill=col, width=2)
    xn = xpos(t_now)
    d.line([(xn, top), (xn, bot)], fill=INK, width=1)
    d.text((x0 + w, bot + 4), f"{T}" + (" (log scale)" if log_x else ""), font=F["tiny"], fill=MUTED, anchor="ra")
    d.text((x0, bot + 4), "1" if log_x else "0", font=F["tiny"], fill=MUTED)
    if log_y:
        d.text((x0 - 4, top), f"{hi:.0f}", font=F["tiny"], fill=MUTED, anchor="ra")
        d.text((x0 - 4, bot), "0.01", font=F["tiny"], fill=MUTED, anchor="rm")
    else:
        d.text((x0 - 4, top), f"{hi:.0f}", font=F["tiny"], fill=MUTED, anchor="ra")
        d.text((x0 - 4, bot), "0", font=F["tiny"], fill=MUTED, anchor="rm")
    if labels:
        lx = x0 + 8
        for text, col in labels:
            d.rectangle([lx, top + 2, lx + 10, top + 12], fill=col)
            d.text((lx + 14, top + 7), text, font=F["tiny"], fill=INK, anchor="lm")
            lx += 22 + d.textlength(text, font=F["tiny"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--method", choices=list(NAMES), required=True)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = a.method
    summary, events, D = load(a.id, m)
    puzzle, solution = parse_grid(summary["puzzle"]), parse_grid(summary["solution"])
    fr, fc = summary["focus_cell"]
    accent = COLOR[m]
    X = D["x"].astype(np.float32)                           # index t = 0..T
    T = int(D["iterations"])
    solved = bool(D["solved"])
    empty = puzzle == 0
    n_empty = int(empty.sum())
    placed = (X.max(axis=3) > 0.85) & empty[None]
    digit = X.argmax(axis=3) + 1
    right = (placed & (digit == solution[None])).sum(axis=(1, 2))
    wrong = (placed & (digit != solution[None])).sum(axis=(1, 2))
    viol = D["row_v"].sum(axis=1) + D["col_v"].sum(axis=1) + D["box_v"].sum(axis=1)
    ts_all = np.arange(T + 1)
    if m == "dr":
        P = np.unpackbits(D["projections"], axis=-1, count=9).astype(bool)   # (T, 5, 9, 9, 9)
        set_names = ["rows", "columns", "boxes", "clues", "cells"]

    frame_ts = sorted(set(range(1, min(T, 30) + 1)) |
                      set(np.unique(np.geomspace(1, T, a.frames).astype(int)).tolist()) | {T})
    sol_digit = int(solution[fr, fc])
    frames, durations, prev_t = [], [], 0
    changes = 0
    for t in frame_ts:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 22), f"{NAMES[m]}  ·  step by step", font=F["title"], fill=INK)
        d.text((40, 62), ONE_ITERATION[m], font=F["sub"], fill=MUTED)
        d.text((W - 40, 24), f"iteration {t:>4} / {T}", font=F["mono"], fill=INK, anchor="ra")
        d.text((W - 40, 50), f"puzzle {summary['id']} · difficulty {summary['difficulty']:g}",
               font=F["small"], fill=MUTED, anchor="ra")

        changed = {}
        for e in events:
            if prev_t < e["t"] <= t:
                col = CHANGE if e["kind"] != "place" else (OK if e["correct"] else BAD)
                changed[(e["row"], e["col"])] = col
                changes += e["kind"] != "place"
        gx, gy, gs = 62, 122, 486
        draw_grid(d, gx, gy, gs, X[t], puzzle, solution, D["row_v"][t], D["col_v"][t], D["box_v"][t],
                  (fr, fc), changed, accent)
        status = (f"solved at iteration {T}" if solved and t == T else
                  ("stopped at the 3000-iteration cap, not solved" if t == T else
                   f"placed {right[t] + wrong[t]} of {n_empty} empty cells: "
                   f"{right[t]} correct, {wrong[t]} wrong"))
        d.text((gx, gy + gs + 12), status, font=F["head"],
               fill=OK if (solved and t == T) else (BAD if t == T else INK))
        legend = ["large digit: placed (green right, red wrong); ring: changed since last frame",
                  "red tint: rule violated; framed: focus cell"]
        if solved and t == T:
            legend[1] = "solved: the leading digits form a valid grid (some not yet placed)"
        for k, line in enumerate(legend):
            d.text((gx, gy + gs + 36 + 15 * k), line, font=F["tiny"], fill=MUTED)

        px = 600
        d.text((px, 112), f"Focus cell r{fr + 1}c{fc + 1}: its nine weights through iteration {t}",
               font=F["head"], fill=INK)
        i = t - 1
        after = X[t][fr, fc]
        bw, bh, by = 132, 92, 140
        if m == "pg":
            bars(d, px, by, bw, bh, D["focus_before"][i], "1. before", accent, sol_digit)
            bars(d, px + 165, by, bw, bh, D["focus_gradient_step"][i], "2. gradient step x − α∇f",
                 accent, sol_digit, -0.5, 1.5, "may leave [0, 1]")
            bars(d, px + 330, by, bw, bh, after, "3. projected onto simplex", accent, sol_digit,
                 note="≥ 0, sums to 1")
            info = (f"λ (0/1 push) = {D['lam'][i]:.3f}    step α = {D['step'][i]:.3g}    "
                    f"backtracking halvings = {int(D['halvings'][i])}")
        elif m == "apg":
            bars(d, px, by, 112, bh, D["focus_before"][i], "1. before", accent, sol_digit)
            bars(d, px + 150, by, 112, bh, D["focus_y"][i], "2. momentum point y", accent, sol_digit)
            bars(d, px + 300, by, 112, bh, D["focus_gradient_step"][i], "3. gradient step",
                 accent, sol_digit, -0.5, 1.5)
            bars(d, px + 450, by, 112, bh, after, "4. projected", accent, sol_digit)
            mom = "used" if D["used_momentum"][i] else "rejected (worse): stepped from x"
            info = (f"λ = {D['lam'][i]:.3f}   momentum β = {D['beta'][i]:.2f} {mom}   "
                    f"step α = {D['step'][i]:.3g}" + ("   MOMENTUM RESET" if D["restart"][i] else ""))
        elif m == "fw":
            bars(d, px, by, bw, bh, D["focus_before"][i], "1. before", accent, sol_digit)
            bars(d, px + 165, by, bw, bh, D["focus_target"][i], "2. target: best digit", accent, sol_digit,
                 note="all weight on one digit")
            bars(d, px + 330, by, bw, bh, after, "3. x + γ (target − x)", accent, sol_digit)
            info = (f"λ = {D['lam'][i]:.3f}    γ (share of the way to the target) = {D['gamma'][i]:.3g}    "
                    f"halvings = {int(D['halvings'][i])}")
            tx, ty, ts_ = 1100, 140, 140                     # every cell's chosen target digit
            d.text((tx, ty - 2), "targets, all cells", font=F["small"], fill=INK)
            tc = ts_ / 9
            tgt = D["target"][i]
            for r in range(9):
                for c in range(9):
                    col = INK if puzzle[r, c] else (OK if tgt[r, c] == solution[r, c] else BAD)
                    d.text((tx + (c + 0.5) * tc, ty + 22 + (r + 0.5) * tc), str(int(tgt[r, c])),
                           font=F["tiny"], fill=col, anchor="mm")
            for k in range(0, 10, 3):
                d.line([(tx, ty + 22 + k * tc), (tx + ts_, ty + 22 + k * tc)], fill=GRID)
                d.line([(tx + k * tc, ty + 22), (tx + k * tc, ty + 22 + ts_)], fill=GRID)
        else:
            bars(d, px, by, 120, bh, D["focus_average_before"][i], "1. average before", accent, sol_digit)
            vx, vy, cw = px + 160, by, 22                    # vote table: one row per copy
            d.text((vx, vy), "2. each copy projected onto its rule", font=F["small"], fill=INK)
            votes = P[i][:, fr, fc, :]
            for s in range(5):
                yy = vy + 22 + s * 19
                d.text((vx + 62, yy + 8), set_names[s], font=F["tiny"], fill=MUTED, anchor="rm")
                for k in range(9):
                    box = [vx + 70 + k * cw, yy, vx + 70 + (k + 1) * cw - 3, yy + 16]
                    d.rectangle(box, fill=accent if votes[s, k] else BG, outline=GRID)
            for k in range(9):
                d.text((vx + 70 + (k + 0.5) * cw - 1, vy + 22 + 5 * 19 + 6), str(k + 1), font=F["tiny"],
                       fill=OK if k + 1 == sol_digit else MUTED, anchor="mm")
            bars(d, px + 440, by, 120, bh, after, "3. new average", accent, sol_digit,
                 note="moves in steps of 0.2")
            agree = int(votes[:, int(after.argmax())].sum())
            info = (f"copies voting for the leading digit {int(after.argmax()) + 1}: {agree} of 5    "
                    "(placed only when all five agree)")
        d.text((px, 290), info, font=F["txt"], fill=INK)

        tl = ts_all[1:]
        timeline(d, px + 30, 330, 610, 110, tl, [(viol[1:], accent)], t,
                 "Constraint violation (rows + columns + boxes, log scale)", log_y=True)
        timeline(d, px + 30, 488, 610, 110, tl, [(right[1:].astype(float), OK), (wrong[1:].astype(float), BAD)],
                 t, "Placed cells", log_y=False, ymax=n_empty, labels=[("correct", OK), ("wrong", BAD)])
        d.text((px, 650), f"Digit changes so far: {changes} retracted or revised", font=F["txt"], fill=INK)
        d.text((px, 672), "None of the methods picks cells or backtracks: every iteration updates all "
               "729 weights at once.", font=F["tiny"], fill=MUTED)

        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=96))
        durations.append(200 if t <= 12 else 120)
        prev_t = t
    durations[-1] = 3500
    out = Path(a.out) if a.out else ROOT / "results" / "figures" / "animation" / f"steps_{a.id}_{m}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames")


if __name__ == "__main__":
    main()
