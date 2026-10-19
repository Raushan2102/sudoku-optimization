"""Make a small stratified sample from the full Kaggle CSV.

    python scripts/make_sample.py --data data/sudoku-3m.csv --per-band 50 --out data/dev_sample.csv

The sample is small enough to commit; the full dataset stays out of git.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sudoku import load_puzzles, stratified_sample  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="path to the full dataset CSV")
p.add_argument("--per-band", type=int, default=50)
p.add_argument("--rows", type=int, default=None, help="read only the first N rows (faster)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="data/dev_sample.csv")
a = p.parse_args()

df = load_puzzles(a.data, n=a.rows)
s = stratified_sample(df, a.per_band, seed=a.seed)
s.to_csv(a.out, index=False)
print(f"wrote {len(s)} puzzles to {a.out}")
print(s.groupby("band", sort=False).size().to_string())
