# Project Plan — Work Split Proposal

**Status: draft proposal, not final.** For Gracy and Raushan to discuss and adjust together. Once we agree on the split, turn the items below into GitHub Issues (one issue per `### Issue:` block).

## Deadlines

| Date | Milestone |
| --- | --- |
| Fri Nov 13 | Phase 1 done: core math + N×N generalization merged, all tests pass |
| Tue Nov 17 | Phase 2 done: all four methods merged |
| Thu Nov 19 | Phase 3 done: benchmark + scaling runs finished, figures ready |
| **Fri Nov 20, 11:59 PM** | **Slides submitted on Brightspace** (same deadline for every team) |
| Mon Nov 30, 3:28–3:48 PM | Our presentation (20-minute slot) |
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

## Phase 4 — Slides and presentation (slides due Nov 20, talk Nov 30)

### Issue: Slides
Problem and relaxation (1–2 slides), the four methods (1 slide each or one comparison slide), main result (solved rate vs. difficulty), scaling result, what fails and why, conclusion. Upload to Brightspace by Fri Nov 20, 11:59 PM.
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
- Presentation slot: Mon Nov 30, 3:28–3:48 PM. Slides are still due on Brightspace by Fri Nov 20, so Phases 2–4 cannot slip; Nov 21–29 can go to rehearsal and to the report.
- Any changes to phase boundaries or task granularity before we convert this into GitHub Issues?

---

## Status update (2026-11-12)

The section below is Gracy's Nov 10 write-up of the professor's feedback, restored verbatim: it was accidentally dropped when the deadlines version of this plan (commit 9f90802) replaced the file. Answers to its open questions so far:

- **Douglas–Rachford is done for any N = k²** (PR #3), and the paper's exact 5-set model is now the default (PR #4), with the 4-set masked-clue version kept as an option. Dev sample, 10,000 iterations: 5 sets 297/300, 4 sets 296/300; 4 sets needs fewer iterations (median 365 vs 507).
- **N×N generalization:** done inside `src/methods.py` for DR; `src/sudoku.py` and `src/objective.py` still hard-code 9×9 (issue "Generalize the code from 9×9 to N×N grids" above).
- **Puzzle source for 16×16 / 25×25:** still open. Suggestion: generator first (no dependency), and add the paper's ksudoku16/ksudoku25 sets if we can find them, so we can compare with the paper's 92% / 100%.
- **Phase timing:** scaling runs are planned for Phase 3 (by Nov 19), see the deadlines table.

## Professor feedback (2026-11-10)

**Status: new, appended below — does not replace or change Phases 1-4 above.**

The professor reviewed this plan and suggested two additions:

1. For Douglas–Rachford, read Aragón Artacho, Borwein & Tam, *"Recent Results on Douglas–Rachford Methods for Combinatorial Optimization Problems"* (Serdica Math. J., 2013 / JOTA, 2014).
2. Consider 16×16 and 25×25 grids (or larger) to study how the methods scale.

Paper downloaded to [`references/papers/aragon-artacho-borwein-tam-2014-douglas-rachford-combinatorial.pdf`](references/papers/aragon-artacho-borwein-tam-2014-douglas-rachford-combinatorial.pdf) ([arXiv:1305.2657](https://arxiv.org/abs/1305.2657)).

### What the paper actually does (relevant to our `douglas_rachford` stub)

Its Sudoku formulation uses a binary one-hot tensor `B[i,j,k] ∈ {0,1}` and **5** constraint sets, each with its own copy in the Douglas–Rachford product-space construction:
- C1: rows — each `B[i,:,k]` is a basis vector
- C2: columns — each `B[:,j,k]` is a basis vector
- C3: boxes — each box's `B[box,k]` is a basis vector
- C4: clues — `B[i,j,k] = 1` fixed at the given clue positions
- C5: cells — each `B[i,j,:]` is a basis vector (exactly one digit per cell)

Update rule: `x_{n+1} = T_{D,C} x_n` with `T_{A,B} = (I + R_B R_A) / 2` and `R_A = 2P_A - I`. `P_D` averages across the 5 copies (consensus); `P_C` projects each copy independently onto its own basis-vector constraint (snap each group to one-hot). This is essentially what our existing `douglas_rachford` docstring in `src/methods.py` already sketches ("one copy per rule family, project to exact 0/1, average, reflect") — the one real difference is that the paper keeps **clues as an explicit 5th DR set**, instead of a mask applied outside the loop like our sketch implies. Both should reach the same fixed points; the masked version is simpler to implement first, and the exact 5-set version is a natural refinement to try afterward.

The paper reports running exactly this on 16×16 and 25×25 puzzles ("ksudoku16"/"ksudoku25", from the KSudoku KDE game's puzzle sets): DR solved 92% of 16×16 puzzles (mean ≈5.06s) and 100% of 25×25 puzzles (mean ≈4.01s) across repeated random-restart trials. Useful as a target/sanity-check for our own scaling results once we have them.

### New issue: upgrade `douglas_rachford` to the paper's exact 5-set model (optional refinement)
Add clues as an explicit 5th projected copy instead of (or alongside) the current clue-mask approach, matching the paper precisely. Not blocking — implement the simpler masked version first per the existing Phase 1 issue; treat this as a follow-up comparison.

## Phase 5 — Generalize to N×N and study scaling (new, from professor feedback)

**Not started — no code changed for this yet.** Appended to, not a replacement for, Phases 1-4.

### Issue: Generalize core data structures from fixed 9×9 to any N = k²
`src/sudoku.py` and `src/objective.py` currently hardcode 9×9 grids and 3×3 boxes (e.g. `box_sums`'s `reshape(3, 3, 3, 3, 9)`, `init_weights`'s `(9, 9, 9)` shape). Generalize these to take the grid size `N` (box size `sqrt(N)`) as a parameter, so 9×9, 16×16, and 25×25 puzzles all run through the same code path. `project_simplex` and `objective_and_grad` should already be size-agnostic once written (they operate along the last axis) but should be explicitly tested at each size once they exist.

### Issue: Dataset/puzzle source for 16×16 and 25×25 — new database plan
We currently have **no 16×16 or 25×25 puzzles**: the Kaggle "3 million Sudoku puzzles" dataset (our only data source so far) is 9×9-only. Options to decide between, as a new data-sourcing track alongside the existing Kaggle 9×9 pipeline:
- **Generate synthetically**: a generator script (backtracking solver + random clue removal), parameterized by `N`, following the same pattern as `scripts/make_sample.py` (e.g. `scripts/make_sample_nxn.py`). No external dependency, full control over difficulty bands and sample size.
- **Source existing puzzles**: look for the paper's own "ksudoku16"/"ksudoku25" test sets (KSudoku KDE game puzzle files) or another public 16×16/25×25 dataset, so our results can be compared directly against the paper's reported 92%/100% success rates.
- Either way, the full puzzle files (if large) stay out of git per the existing `.gitignore` convention (`data/*` except `.gitkeep` and `dev_sample.csv`); any new dev-sample files for N=16/25 should follow the same naming convention, e.g. `data/dev_sample_16.csv`, `data/dev_sample_25.csv`.

Decision between the two options, and the generator/download script itself, are still open — nothing built yet.

### Issue: Scaling-study experiment design
Once the generalization above is done, extend `experiments/run_benchmark.py` to run all methods across N ∈ {9, 16, 25} (and potentially larger), reporting success rate and iterations/time-to-solve vs. N, per method. Compare our Douglas–Rachford numbers against the paper's reported 92% (16×16) / 100% (25×25) success rates as a sanity check.

## Open questions for discussion (added 2026-11-10)

- Should the N×N generalization (Phase 5) cover all four methods, or start with Douglas–Rachford only (the method the professor named specifically) and extend to PG/APG/FW later?
- Synthetic puzzle generator vs. sourcing the paper's own ksudoku16/ksudoku25 test sets for the new dataset — which do we want, and who owns it?
- Should the paper's exact 5-set Douglas–Rachford model (explicit clue projection) replace our masked-clue sketch, or do we implement the masked version first and treat the 5-set version as a stretch goal?
- Where does Phase 5 sit relative to Phases 1-4 — done in parallel once Phase 1's core math lands, or only after Phase 3's benchmarking is complete?
