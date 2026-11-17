#!/usr/bin/env bash
# Final experiments for the slides and report (9x9 only). Same iteration budget for every method.
#   bash experiments/run_final.sh
# Writes results/final/*.csv and figures into results/figures/{main,restarts}/.
set -e
cd "$(dirname "$0")/.."
OUT=results/final
mkdir -p "$OUT"

# E1  main comparison, 9x9 Kaggle sample: one start, at most 3000 iterations
python experiments/run_benchmark.py --sample data/dev_sample.csv    --max-iter 3000 --label "1 start x 3000" --out $OUT/main_9.csv
# E2  same total budget as 5 x 1000, spent on random restarts
python experiments/run_benchmark.py --sample data/dev_sample.csv    --max-iter 1000 --restarts 5 --label "5 starts x 1000" --out $OUT/restarts_9.csv

python experiments/plots.py --results "$OUT/main_9.csv" --out results/figures/main
python experiments/plots.py --results "$OUT/restarts_9.csv" --out results/figures/restarts

# E3  famous hard 9x9 puzzles, repeated random starts (as in Table 3 of the DR paper)
python experiments/run_hard.py --methods dr --reps 100 --out $OUT/hard_dr.csv
python experiments/run_hard.py --methods pg apg fw --reps 20 --max-iter 3000 --out $OUT/hard_grad.csv
