# Solving Sudoku with Continuous Optimization

Course project for ISE/CSCI 633 (Large Scale Optimization for Machine Learning), USC, Fall 2026.

We solve Sudoku with constrained first-order methods instead of combinatorial search. Each cell gets nine weights, one per digit, that lie on a probability simplex. The objective penalizes rows, columns and boxes that do not contain each digit exactly once, plus a term that pushes weights toward 0 or 1. We compare how often each method solves a puzzle and how this depends on puzzle difficulty.

**Main question:** how well can the course's constrained first-order methods solve Sudoku through a continuous relaxation, and how does their success depend on how hard the puzzle is for combinatorial search?

## Methods

| Method | Idea | Status |
| --- | --- | --- |
| Projected gradient | gradient step, then project each cell onto the simplex | to do |
| Accelerated projected gradient | Nesterov extrapolation before the projected step | to do |
| Frank–Wolfe | move toward the best digit per cell under the gradient | to do |
| Douglas–Rachford | projections onto the exact 0/1 rules with reflections (outside the course, for comparison) | to do |

## Data

Kaggle dataset **"3 million Sudoku puzzles with ratings"** (CC0). Each row has a puzzle, its solution, the number of clues and a difficulty rating from 0 to 8.5.

The full file is not in this repository. Download it from Kaggle into `data/`, then make a small stratified sample:

```bash
python scripts/make_sample.py --data data/<dataset file>.csv --per-band 50 --out data/dev_sample.csv
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # checker tests pass; method tests are skipped until written
python experiments/run_benchmark.py --sample data/dev_sample.csv --out results/dev_run.csv
```

## Layout

```
src/sudoku.py        loading, parsing, solution checks, stratified sampling
src/objective.py     relaxation helpers; project_simplex and objective_and_grad (to do)
src/methods.py       the four methods (to do), shared Result type
experiments/         benchmark runs
scripts/             data preparation
tests/               unit tests
notebooks/           exploration and figures
results/             tables and figures for the report
```

## Team

- [Name 1]
- [Name 2]
