"""Extra analysis figures from the final 9x9 results (run after experiments/run_final.sh).

    python experiments/analysis_plots.py

Reads results/final/{main_9,restarts_9,hard_dr,hard_grad}.csv and writes PNGs to
results/figures/analysis/. Colors match experiments/plots.py; hatching repeats each
method's identity so the figures also work in grayscale.
"""
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / "results" / "figures" / "analysis"
out.mkdir(parents=True, exist_ok=True)

M = ["pg", "apg", "fw", "dr"]
LABEL = {"pg": "Projected gradient", "apg": "Accelerated PG", "fw": "Frank–Wolfe", "dr": "Douglas–Rachford"}
COLOR = {"pg": "#2a78d6", "apg": "#eb6834", "fw": "#1baf7a", "dr": "#b37a00"}
HATCH = {"pg": "", "apg": "//", "fw": "..", "dr": "xx"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BANDS = ["0", "0.1-1.9", "2-2.9", "3-3.9", "4-4.9", "5-8.5"]
plt.rcParams.update({"font.size": 10, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "hatch.color": "#ffffff", "hatch.linewidth": 0.8})


def axes_style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print("wrote", out / name)


e1 = pd.read_csv(root / "results/final/main_9.csv", dtype={"band": str})
e2 = pd.read_csv(root / "results/final/restarts_9.csv", dtype={"band": str})

# 1. Overall solve rate with 95% CI, one start vs up to 5 starts
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(M))
w = 0.36
for j, (d, lab, alpha) in enumerate([(e1, "1 start × 3000 iterations", 1.0), (e2, "up to 5 starts × 1000", 0.55)]):
    for i, m in enumerate(M):
        s = d[d.method == m].solved
        k, n = int(s.sum()), len(s)
        lo, hi = wilson(k, n)
        p = 100 * k / n
        xi = x[i] + (j - 0.5) * w
        ax.bar(xi, p, w - 0.03, color=COLOR[m], alpha=alpha, hatch=HATCH[m], edgecolor=SURF, linewidth=0)
        ax.errorbar(xi, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=3)
        ax.text(xi, hi + 1.5, f"{k}", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(x, [LABEL[m] for m in M])
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.legend(handles=[Patch(facecolor="#898781", label="1 start × 3000 iterations"),
                   Patch(facecolor="#898781", alpha=0.55, label="up to 5 starts × 1000")],
          frameon=False, loc="upper left")
axes_style(ax, "Puzzles solved out of 300, with 95% confidence intervals", "", "solved")
save(fig, "overall_solve_rate.png")

# 2. Effect of restarts per band: small multiples
fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
for ax, m in zip(axes, M):
    a = [100 * e1[(e1.method == m) & (e1.band == b)].solved.mean() for b in BANDS]
    b_ = [100 * e2[(e2.method == m) & (e2.band == b)].solved.mean() for b in BANDS]
    xs = np.arange(len(BANDS))
    ax.fill_between(xs, a, b_, color=COLOR[m], alpha=0.15, linewidth=0)
    ax.plot(xs, a, color=COLOR[m], lw=2, ls="--", marker="o", ms=5, label="1 start")
    ax.plot(xs, b_, color=COLOR[m], lw=2, marker="o", ms=6, mfc=COLOR[m], label="up to 5 starts")
    ax.set_xticks(xs, BANDS, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    tot1, tot2 = int(e1[e1.method == m].solved.sum()), int(e2[e2.method == m].solved.sum())
    axes_style(ax, f"{LABEL[m]}\n{tot1} → {tot2} solved", "difficulty band", "solved" if m == "pg" else "")
    ax.title.set_fontsize(10.5)
axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
save(fig, "restart_gain_by_band.png")

# 3. Cost per iteration and total wall time (two charts, one axis each)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
msit = [float((1000 * e1[e1.method == m].seconds / e1[e1.method == m].iterations.clip(lower=1)).median()) for m in M]
tot = [float(e1[e1.method == m].seconds.sum()) for m in M]
for ax, vals, title, fmt, yl in [(axes[0], msit, "Cost of one iteration (median)", "{:.2f} ms", "milliseconds"),
                                 (axes[1], tot, "Total time for all 300 puzzles, 1 start", "{:.0f} s", "seconds")]:
    for i, m in enumerate(M):
        ax.bar(i, vals[i], 0.6, color=COLOR[m], hatch=HATCH[m], edgecolor=SURF, linewidth=0)
        ax.text(i, vals[i] * 1.02, fmt.format(vals[i]), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(M)), ["Projected\ngradient", "Accelerated\nPG", "Frank–\nWolfe", "Douglas–\nRachford"])
    ax.set_ylim(0, max(vals) * 1.18)
    axes_style(ax, title, "", yl)
save(fig, "cost_per_method.png")

# 4. Iterations needed when a method succeeds (1 start), box plot
fig, ax = plt.subplots(figsize=(8, 4))
data = [e1[(e1.method == m) & e1.solved].iterations.values for m in M]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                medianprops={"color": INK, "lw": 1.5},
                flierprops={"marker": "o", "ms": 3, "mfc": MUTED, "mec": "none", "alpha": 0.5},
                whiskerprops={"color": MUTED}, capprops={"color": MUTED})
for patch, m in zip(bp["boxes"], M):
    patch.set_facecolor(COLOR[m]); patch.set_alpha(0.75); patch.set_edgecolor(COLOR[m])
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
ax.set_xticks(range(1, 5), [f"{LABEL[m]}\nmedian {int(np.median(d))}, n = {len(d)}" for m, d in zip(M, data)])
axes_style(ax, "Iterations to solve, solved puzzles only (1 start, log scale)", "", "iterations")
save(fig, "iterations_to_solve.png")

# 5. Who solves what (1 start): overlap of solved sets
piv = e1.pivot_table(index="id", columns="method", values="solved", aggfunc="first").astype(bool)
grad = piv[["pg", "apg", "fw"]]
cats = [("All four methods", int((grad.all(axis=1) & piv.dr).sum())),
        ("DR + one or two\ngradient methods", int((grad.any(axis=1) & ~grad.all(axis=1) & piv.dr).sum())),
        ("Only DR", int((~grad.any(axis=1) & piv.dr).sum())),
        ("A gradient method\nbut not DR", int((grad.any(axis=1) & ~piv.dr).sum())),
        ("No method", int((~piv.any(axis=1)).sum()))]
fig, ax = plt.subplots(figsize=(8, 3.6))
ys = np.arange(len(cats))[::-1]
for y, (lab, v) in zip(ys, cats):
    ax.barh(y, v, 0.6, color="#2a78d6" if v else GRID)
    ax.text(v + 3, y, f"{v}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(ys, [c[0] for c in cats])
ax.set_xlim(0, 300)
ax.grid(axis="x", color=GRID, linewidth=0.8)
axes_style(ax, "Which methods solve each of the 300 puzzles (1 start)", "puzzles", "")
ax.grid(axis="y", visible=False)
save(fig, "solved_overlap.png")

# 6. Famous hard puzzles
h = pd.concat([pd.read_csv(root / "results/final/hard_dr.csv"), pd.read_csv(root / "results/final/hard_grad.csv")])
P = ["AI Escargot", "Inkala 2012", "Easter Monster", "Nasty (paper Fig. 11)"]
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.2
for i, m in enumerate(M):
    for j, p in enumerate(P):
        d = h[(h.method == m) & (h.puzzle == p)]
        k, n = int(d.solved.sum()), len(d)
        v = 100 * k / n
        xi = j + (i - 1.5) * w
        ax.bar(xi, max(v, 0.6), w - 0.03, color=COLOR[m], hatch=HATCH[m], edgecolor=SURF, linewidth=0,
               label=LABEL[m] if j == 0 else None)
        ax.text(xi, max(v, 0.6) + 1.5, f"{k}/{n}", ha="center", va="bottom", fontsize=7.5, color=INK, rotation=90)
ax.set_xticks(range(len(P)), P)
ax.set_ylim(0, 125)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=8.5)
axes_style(ax, "Famous hard puzzles: share of random starts that solve them", "",
           "starts that solve")
save(fig, "hard_puzzles.png")

# 7. Which start solved the puzzle (restarts run)
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.2
for i, m in enumerate(M):
    d = e2[(e2.method == m) & e2.solved]
    counts = [int((d.starts == s).sum()) for s in range(1, 6)]
    xs = np.arange(5) + (i - 1.5) * w
    ax.bar(xs, counts, w - 0.03, color=COLOR[m], hatch=HATCH[m], edgecolor=SURF, linewidth=0, label=LABEL[m])
ax.set_xticks(range(5), [f"start {s}" for s in range(1, 6)])
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
ax.legend(frameon=False, ncol=2, fontsize=8.5)
axes_style(ax, "On which start each solved puzzle was solved (up to 5 starts, log scale)", "",
           "puzzles")
save(fig, "solved_on_start.png")
