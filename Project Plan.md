# Project Plan — Work Split Proposal

**Status: draft proposal, not final.** For Gracy and Raushan to discuss and adjust together. Once we agree on the split, turn the items below into GitHub Issues (one issue per `### Issue:` block).

## Current state of the repo

**Done:** data loading/parsing/checking (`src/sudoku.py`), weight init/rounding/constraint-violation metric (`src/objective.py`), the `Result` type and method registry (`src/methods.py`), the benchmark and sampling scripts, and the test scaffolding.

**Not done:** `project_simplex`, `objective_and_grad`, and all four methods (`pg`, `apg`, `fw`, `dr`) — all currently `raise NotImplementedError`. No dataset downloaded yet, no experiments run, no report written.

## Why this split

- `project_simplex` + `objective_and_grad` are a hard dependency for **PG, APG, and FW**. Douglas–Rachford is the one method that needs neither (it projects directly onto exact 0/1 rules) — so it's the natural task to run in parallel with the core math instead of waiting on it.
- PG and APG share the same update loop (APG is PG plus Nesterov extrapolation), so they belong together.
- FW needs the gradient but not `project_simplex` (its subproblem just picks the digit with smallest partial derivative per cell), so it pairs naturally with DR as the other track.
- Net result: only two points in the whole project genuinely require both people — the small Phase 1 core, and the Phase 4 final comparison. Everything else is two clean, independent tracks.

## Phase 1 — Foundations

### Issue: Implement `project_simplex` and `objective_and_grad`
Core math in `src/objective.py` — blocking item for PG, APG, and FW. Verify with the existing tests: `test_projection_lands_on_simplex`, `test_projection_keeps_points_already_on_simplex`, `test_gradient_matches_finite_differences`.
Suggested owner: Track 1 (whoever prefers the optimization math).

### Issue: Download dataset and build dev sample
Download the Kaggle dataset ("3 million Sudoku puzzles with ratings", CC0), run `scripts/make_sample.py` to produce `data/dev_sample.csv`, confirm `load_puzzles` / `stratified_sample` work end-to-end. Independent of the core math above.
Suggested owner: Track 2.

### Issue: Implement `douglas_rachford`
In `src/methods.py` — doesn't depend on `project_simplex` or `objective_and_grad`, so it can start immediately in parallel with the core-math issue.
Suggested owner: Track 2.

## Phase 2 — The four methods

### Issue: Implement `projected_gradient`
Armijo backtracking line search. Depends on Phase 1 core math.
Suggested owner: Track 1.

### Issue: Implement `accelerated_projected_gradient`
Add Nesterov extrapolation on top of `projected_gradient`. Depends on the PG issue above.
Suggested owner: Track 1.

### Issue: Implement `frank_wolfe`
Linear minimization oracle per cell + Armijo or `2/(t+2)` step size. Depends on Phase 1's `objective_and_grad` only (no simplex projection needed).
Suggested owner: Track 2.

(No file-level conflicts — different functions in the same `src/methods.py`, merges cleanly.)

## Phase 3 — Benchmarking

### Issue: Run full benchmark across all four methods
One run of `experiments/run_benchmark.py` over the full stratified sample. Shared step, doesn't need splitting.

### Issue: Analyze PG/APG results
Convergence curves, effect of the `lam_schedule`, iteration counts to solve, by difficulty band.
Suggested owner: Track 1.

### Issue: Analyze FW/DR results
Same analysis for Frank–Wolfe and Douglas–Rachford.
Suggested owner: Track 2.

## Phase 4 — Comparison & report

### Issue: Write methodology + results sections
Each track writes up its own two methods, reusing the Phase 3 analysis.

### Issue: Final comparison and conclusion (joint)
Solved-rate vs. difficulty band, all four methods side by side — the one necessary sync point since it merges both tracks' results.

### Issue: Update README
Flip the methods table (`to do` → `done`) and fill in any remaining placeholder sections.

## Open questions for discussion

- Who takes Track 1 vs. Track 2? Either can do either — suggest picking based on comfort with optimization math vs. scripting/data work.
- Any changes to phase boundaries or task granularity before we convert this into GitHub Issues?