"""Animated GIF: the four methods filling in the same Sudoku, side by side.

    python experiments/animate.py --id 35099 --out results/figures/animation/race_35099.gif
    python experiments/animate.py --id 272592 --iters 800 --out results/figures/animation/stuck_272592.gif

Each empty cell shows its nine weights as pencil marks (darker = larger weight). When one
digit holds more than 85 % of the weight the cell shows that digit in large type: green if it
matches the solution, red if not. Clues are dark. A method that solves the puzzle freezes and
says at which iteration. All methods run with max_iter=3000, exactly as in experiment E1; the
animation shows the first --iters iterations on a shared, roughly logarithmic time line.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods import METHODS  # noqa: E402
from src.sudoku import parse_grid  # noqa: E402

NAMES = {"pg": "Projected gradient", "apg": "Accelerated PG", "fw": "Frank–Wolfe", "dr": "Douglas–Rachford"}
COLORS = {"pg": (42, 120, 214), "apg": (235, 104, 52), "fw": (27, 143, 102), "dr": (179, 122, 0)}
BG, INK, GRID, MUTED = (255, 250, 241), (34, 34, 59), (196, 190, 176), (110, 106, 98)
OK, BAD = (26, 140, 90), (214, 69, 65)
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold.ttf" if bold else ".ttf"
    try:
        return ImageFont.truetype(FONT_DIR + name, size)
    except OSError:
        return ImageFont.load_default()


def record(method, puzzle, keep):
    """Run a method once (as in E1) and keep the weights at the iterations in `keep`."""
    snaps = {}

    def cb(t, x):
        if t in keep:
            snaps[t] = x.copy()

    res = METHODS[method](puzzle, max_iter=3000, callback=cb)
    return snaps, res


def draw_grid(d, x0, y0, size, x, puzzle, solution):
    N = puzzle.shape[0]
    n = int(round(N ** 0.5))
    cell = size / N
    big, small = font(int(cell * 0.6), bold=True), font(int(cell * 0.27), mono=True)
    for r in range(N):
        for c in range(N):
            cx, cy = x0 + c * cell, y0 + r * cell
            if puzzle[r, c]:
                d.text((cx + cell / 2, cy + cell / 2), str(puzzle[r, c]), font=big, fill=INK, anchor="mm")
                continue
            w = x[r, c]
            k = int(w.argmax())
            if w[k] > 0.85:
                col = OK if k + 1 == solution[r, c] else BAD
                d.text((cx + cell / 2, cy + cell / 2), str(k + 1), font=big, fill=col, anchor="mm")
            else:
                for dgt in range(N):
                    a = float(min(1.0, w[dgt] / 0.5))
                    if a < 0.06:
                        continue
                    col = tuple(int(BG[i] + (INK[i] - BG[i]) * a) for i in range(3))
                    rr, cc = divmod(dgt, n)
                    d.text((cx + (cc + 0.5) * cell / n, cy + (rr + 0.5) * cell / n), str(dgt + 1),
                           font=small, fill=col, anchor="mm")
    for i in range(N + 1):
        wdt = 4 if i % n == 0 else 1
        d.line([(x0, y0 + i * cell), (x0 + size, y0 + i * cell)], fill=INK if i % n == 0 else GRID, width=wdt)
        d.line([(x0 + i * cell, y0), (x0 + i * cell, y0 + size)], fill=INK if i % n == 0 else GRID, width=wdt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/dev_sample.csv")
    ap.add_argument("--id", required=True)
    ap.add_argument("--methods", nargs="+", default=["pg", "apg", "fw", "dr"])
    ap.add_argument("--iters", type=int, default=None, help="last iteration shown (default: slowest solve)")
    ap.add_argument("--frames", type=int, default=70)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.sample, dtype={"puzzle": str, "solution": str, "id": str})
    row = df[df["id"] == str(a.id)].iloc[0]
    puzzle, solution = parse_grid(row["puzzle"]), parse_grid(row["solution"])

    full = set(range(1, 3001))
    runs = {m: record(m, puzzle, full) for m in a.methods}
    last = a.iters or max(res.iterations for _, res in runs.values())
    ts = sorted(set(np.unique(np.geomspace(1, last, a.frames).astype(int)))
                | {res.iterations for _, res in runs.values() if res.solved and res.iterations <= last})

    panel, gap, top, side = 360, 44, 132, 40
    W = side * 2 + len(a.methods) * panel + (len(a.methods) - 1) * gap
    H = top + panel + 70
    title = f"Same puzzle, four methods  ·  difficulty rating {row['difficulty']}"
    frames = []
    for t in ts + [ts[-1]] * 12:                              # hold the last frame
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((side, 26), title, font=font(26, bold=True), fill=INK)
        d.text((W - side, 30), f"iteration {t}", font=font(24, mono=True), fill=MUTED, anchor="ra")
        for i, m in enumerate(a.methods):
            snaps, res = runs[m]
            x0 = side + i * (panel + gap)
            done = res.solved and res.iterations <= t
            tt = min(t, res.iterations)
            x = snaps.get(tt, snaps[max(snaps)])
            d.rounded_rectangle([x0, 76, x0 + 22, 98], radius=5, fill=COLORS[m])
            d.text((x0 + 32, 87), NAMES[m], font=font(22, bold=True), fill=INK, anchor="lm")
            draw_grid(d, x0, top, panel, x, puzzle, solution)
            if done:
                msg, col = f"solved at iteration {res.iterations}", OK
            elif t >= res.iterations:
                msg, col = "stopped without a solution", BAD
            else:
                wrong = int(((x.argmax(axis=2) + 1 != solution) & (puzzle == 0) & (x.max(axis=2) > 0.85)).sum())
                msg, col = (f"{wrong} confident cells wrong" if wrong else "working…"), (BAD if wrong else MUTED)
            d.text((x0 + panel / 2, top + panel + 34), msg, font=font(21, bold=done), fill=col, anchor="mm")
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    durations = [140] * len(ts) + [250] * 12
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {a.out}: {len(frames)} frames, {W}x{H}, iterations 1..{last}")
    for m in a.methods:
        print(m, "solved" if runs[m][1].solved else "not solved", runs[m][1].iterations)


if __name__ == "__main__":
    main()
