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

---

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