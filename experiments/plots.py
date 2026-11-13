"""Shared figures for the report and slides, so every method is drawn the same way.

    python experiments/plots.py --results "results/*.csv" --out results/figures

Reads the CSV files written by run_benchmark.py (any number, any methods, any grid sizes).
Writes PNG figures:

  solved_by_band.png       share of puzzles solved per difficulty band, one line per method (9x9)
  iterations_by_band.png   median iterations to solve (solved puzzles only), per band (9x9)
  solved_vs_iterations.png share of puzzles solved within t iterations, one line per method (9x9)
  scaling.png              time per iteration and median time to solve vs grid size N

The last-but-one figure compares speed of convergence across methods on one scale; the
methods' own objectives (penalty, DR consensus) are not comparable with each other.

A figure is skipped when the data it needs is not there (e.g. scaling with only 9x9 results).
Every method keeps the same color and marker in every figure (METHOD_STYLE).
"""
import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed colors and markers per method (color follows the method, never its rank).
# Markers and line styles repeat the identity so the figures also work in grayscale.
METHOD_STYLE = {
    "pg":  {"label": "Projected gradient", "color": "#2a78d6", "marker": "o", "ls": "-"},
    "apg": {"label": "Accelerated PG",     "color": "#eb6834", "marker": "s", "ls": "--"},
    "fw":  {"label": "Frank–Wolfe",        "color": "#1baf7a", "marker": "^", "ls": "-."},
    "dr":  {"label": "Douglas–Rachford",   "color": "#b37a00", "marker": "D", "ls": ":"},
}
ORDER = list(METHOD_STYLE)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(m):
    return METHOD_STYLE.get(m, {"label": m, "color": "#6250d6", "marker": "v", "ls": "-"})


def setup_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def methods_in(df):
    present = list(df["method"].unique())
    return [m for m in ORDER if m in present] + [m for m in present if m not in ORDER]


def band_order(df):
    """Bands in the order they first appear (the sampling scripts write them easy -> hard)."""
    return list(dict.fromkeys(df["band"]))


def load_results(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        if "N" not in d.columns:      # files from before the N column were all 9x9
            d["N"] = 9
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def line_by_band(df, value, ylabel, title, path, percent=False):
    bands = band_order(df)
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods_in(df):
        s = style(m)
        y = [value(df[(df.method == m) & (df.band == b)]) for b in bands]
        ax.plot(x, y, color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=7, label=s["label"])
    ax.set_xticks(x, bands)
    if percent:
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    setup_axes(ax, title, "difficulty band", ylabel)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def scaling(df, path):
    if df["N"].nunique() < 2:
        print("skip scaling: results for only one grid size")
        return
    df = df.assign(ms_per_iter=1000 * df.seconds / df.iterations.clip(lower=1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    Ns = sorted(df["N"].unique())
    for m in methods_in(df):
        s = style(m)
        d = df[df.method == m]
        per_it = [d[d.N == n].ms_per_iter.median() for n in Ns]
        solved = d[d.solved]
        to_solve = [solved[solved.N == n].seconds.median() for n in Ns]
        kw = dict(color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=7, label=s["label"])
        axes[0].plot(Ns, per_it, **kw)
        axes[1].plot(Ns, to_solve, **kw)
    for ax, (t, yl) in zip(axes, [("Cost of one iteration", "milliseconds per iteration"),
                                   ("Time to solve (solved puzzles)", "median seconds")]):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(Ns, [f"{n}×{n}" for n in Ns])
        setup_axes(ax, t, "grid size", yl)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def solved_vs_iterations(df, path):
    """Empirical CDF: share of all puzzles solved within t iterations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods_in(df):
        s = style(m)
        d = df[df.method == m]
        it = np.sort(d[d.solved].iterations.to_numpy())
        if len(it) == 0:
            continue
        share = 100 * np.arange(1, len(it) + 1) / len(d)
        ax.step(it, share, where="post", color=s["color"], ls=s["ls"], lw=2, label=s["label"])
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    setup_axes(ax, "Puzzles solved within a given number of iterations (9×9)",
               "iterations (log scale)", "solved")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True, help="CSV files or globs from run_benchmark.py")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    paths = sorted({f for pat in a.results for f in glob.glob(pat)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(paths)
    nine = df[df.N == 9]
    if len(nine):
        line_by_band(nine, lambda d: 100 * d.solved.mean() if len(d) else np.nan,
                     "solved", "Share of puzzles solved (9×9)", out / "solved_by_band.png", percent=True)
        line_by_band(nine, lambda d: d[d.solved].iterations.median() if d.solved.any() else np.nan,
                     "median iterations", "Iterations to solve (9×9, solved puzzles)",
                     out / "iterations_by_band.png")
        solved_vs_iterations(nine, out / "solved_vs_iterations.png")
    scaling(df, out / "scaling.png")


if __name__ == "__main__":
    main()
