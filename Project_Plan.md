# Project Plan — Work Split Proposal

**Status: draft proposal, not final.** For Gracy and Raushan to discuss and adjust together. Once we agree on the split, turn the items below into GitHub Issues (one issue per `### Issue:` block).

## Deadlines

| Date | Milestone |
| --- | --- |
| Fri Nov 13 | Phase 1 done: core math + N×N generalization merged, all tests pass |
| Tue Nov 17 | Phase 2 done: all four methods merged |
| Thu Nov 19 | Phase 3 done: benchmark + scaling runs finished, figures ready |
| Sat Nov 21 | Slides done and rehearsed (presentation slots: Nov 23, Nov 30, Dec 2 — confirm ours) |
| Fri Dec 4 | Final report submitted |

If we fall behind, cut in this order: 25×25 runs (mention as future work) → APG variants (keep one, no restart comparison) → size of the full sample. Keep Douglas–Rachford and 16×16: the professor asked for both.

## Current state of the repo

**Done:** data loading/parsing/checking (`src/sudoku.py`), weight init/rounding/constraint-violation metric (`src/objective.py`), the `Result` type and method registry (`src/methods.py`), the benchmark and sampling scripts, the test scaffolding, and the dev sample (`data/dev_sample.csv`, 50 puzzles per difficulty band).

**Not done:** `project_simplex`, `objective_and_grad`, and all four methods (`pg`, `apg`, `fw`, `dr`) — all currently `raise NotImplementedError`. No experiments run, no report written.

## Why this split

- `project_simplex` + `objective_and_grad` are a hard dependency for **PG, APG, and FW**. Douglas–Rachford is the one method that needs neither (it projects directly onto exact 0/1 rules) — so it's the natural task to run in parallel with the core math instead of waiting on it.
- PG and APG share the same update loop (APG is PG plus Nesterov extrapolation), so they belong together.
- FW needs the gradient but not `project_simplex` (its subproblem just picks the digit with smallest partial derivative per cell), so it pairs naturally with DR as the other track.
- Net result: only two points in the whole project genuinely require both people — the small Phase 1 core, and the Phase 5 final comparison. Everything else is two clean, independent tracks.

## Phase 1 — Foundations (by Nov 13)

### Issue: Implement `project_simplex` and `objective_and_grad`
Core math in `src/objective.py` — blocking item for PG, APG, and FW. Verify with the existing tests: `test_projection_lands_on_simplex`, `test_projection_keeps_points_already_on_simplex`, `test_gradient_matches_finite_differences`. Write both for a general side length `N = x.shape[0]`, not a hard-coded 9 (see the N×N issue below).
Suggested owner: Track 1 (whoever prefers the optimization math).

### Issue: Download dataset and build dev sample — DONE
`data/dev_sample.csv` is committed: 300 puzzles, 50 per difficulty band.

### Issue: Generalize the code from 9×9 to N×N grids
Professor's feedback: study how the methods scale on 16×16 and 25×25 grids. Replace the hard-coded `9` and `3` with `N` (side length) and `n = sqrt(N)` (box size) in `init_weights`, `box_sums`, `round_grid`, `is_valid_solution` and anything else that assumes 9×9. Add a test that a solved 16×16 grid passes `is_solved`. Do this now, while the code is small.
Suggested owner: Track 2.

### Issue: Generator for 16×16 and 25×25 puzzles
The Kaggle dataset has only 9×9 puzzles. Build a full solved N×N grid from a pattern, shuffle digits, rows within bands and columns within stacks, then blank out a fraction of cells (e.g. 40%, 50%, 60%) as the difficulty knob. No uniqueness check needed: `is_solved` accepts any grid that satisfies the rules and keeps the clues. Save samples to `data/`.
Suggested owner: Track 2.

### Issue: Implement `douglas_rachford`
In `src/methods.py` — doesn't depend on `project_simplex` or `objective_and_grad`, so it can start immediately in parallel with the core-math issue. Follow the formulation in Aragón Artacho, Borwein & Tam (2014), as the professor suggested, and cite it in the report.
Suggested owner: Track 2.

## Phase 2 — The four methods (by Nov 17)

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

## Phase 3 — Benchmarking (by Nov 19)

### Issue: Shared plotting script
One script (e.g. `experiments/plots.py`) that reads `results/*.csv` and draws the same figures for every method: solved rate vs. difficulty band, iterations and time to solve, constraint violation over iterations. Write it first, so both tracks produce figures in one style that can go side by side in the slides and report.
Suggested owner: either; agree before the analysis issues start.

### Issue: Run full benchmark across all four methods
One run of `experiments/run_benchmark.py` over the full stratified sample (9×9). Shared step, doesn't need splitting.

### Issue: Scaling experiment (16×16, 25×25)
Run all four methods on generated N×N puzzles. Report time per iteration, iterations to solve and solved rate vs. N. Expect the per-iteration cost to differ: FW picks a minimum per cell, PG sorts per cell, DR does several projections per iteration.
Suggested owner: Track 2.

### Issue: Analyze PG/APG results
Convergence curves, effect of the `lam_schedule`, iteration counts to solve, by difficulty band.
Suggested owner: Track 1.

### Issue: Analyze FW/DR results
Same analysis for Frank–Wolfe and Douglas–Rachford.
Suggested owner: Track 2.

## Phase 4 — Slides and presentation (by Nov 21)

### Issue: Slides
Problem and relaxation (1–2 slides), the four methods (1 slide each or one comparison slide), main result (solved rate vs. difficulty), scaling result, what fails and why, conclusion. Upload to Brightspace before our slot.
Owner: both.

### Issue: Rehearsal
Time the talk once end to end and prepare answers for likely questions (why a relaxation, why DR does better, how this compares with backtracking). Each of us also asks at least one question at other teams' talks (participation grade).
Owner: both.

## Phase 5 — Report (by Dec 4)

### Issue: Write methodology + results sections
Each track writes up its own two methods, reusing the Phase 3 analysis.

### Issue: Final comparison and conclusion (joint)
Solved-rate vs. difficulty band, all four methods side by side, plus the scaling result — the one necessary sync point since it merges both tracks' results.

### Issue: Report requirements check
Main text at most 3 pages (appendix and references don't count; up to 30 pages of appendix), self-contained for a classmate, a section on each member's contributions, and a note on how AI tools were used.
Owner: both.

### Issue: Update README
Flip the methods table (`to do` → `done`) and fill in any remaining placeholder sections.

## Open questions for discussion

- Who takes Track 1 vs. Track 2? Either can do either. Raushan already did the dataset work, so Track 2 (DR, FW, N×N generalization and generator) may fit naturally; Track 1 is the core math plus PG/APG.
- Which presentation slot do we have? If it is Nov 30 or Dec 2, Phase 4 can move later.
- Any changes to phase boundaries or task granularity before we convert this into GitHub Issues?
