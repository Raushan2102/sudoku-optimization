#!/usr/bin/env bash
# Final experiments for the slides and report. Same iteration budget for every method.
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
# E3  scaling: generated 16x16 and 25x25 puzzles, one start, at most 3000 iterations
python experiments/run_benchmark.py --sample data/dev_sample_16.csv --max-iter 3000 --label "1 start x 3000" --out $OUT/scale_16.csv
python experiments/run_benchmark.py --sample data/dev_sample_25.csv --max-iter 3000 --label "1 start x 3000" --out $OUT/scale_25.csv

python experiments/plots.py --results "$OUT/main_9.csv" "$OUT/scale_*.csv" --out results/figures/main
python experiments/plots.py --results "$OUT/restarts_9.csv" --out results/figures/restarts

# E4  famous hard 9x9 puzzles, repeated random starts (as in Table 3 of the DR paper)
python experiments/run_hard.py --methods dr --reps 100 --out $OUT/hard_dr.csv
python experiments/run_hard.py --methods pg apg fw --reps 20 --max-iter 3000 --out $OUT/hard_grad.csv
